"""Unified privacy and over-removal scorer against silver labels (`labels/merged/<split>/`).

Combines the two scorers the unified harness descends from, so one run is comparable with both:

- the Alind+Davide scorer: removal by risk tier (high / medium / low), category, field and dataset;
  keep-trap survival; edit precision against the decisions sidecar;
- the Ketai scorer: removal by identifier tier (direct / quasi), entity-level removal, name-token
  removal (a surname left after "[PERSON] Orlov" counts as a leak) and labeled medical concept recall.

Removal = for every gold surface string, the share of its occurrences in the original field that no
longer occur in the anonymized field. Gold strings absent from the original are skipped. Two matching
rules are available:

- `word` (default): whole word, case insensitive, on the raw text (Alind+Davide);
- `normalized`: lower case, hyphens and slashes as spaces, punctuation spaced out (Ketai), so a
  string matches across the tokenized dialogue and the untokenized note.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .labels import tier as category_tier

MATCHERS = ("word", "normalized")
NAME_CATEGORIES = {"PATIENT_NAME", "PERSON_NAME", "CLINICIAN_NAME", "OTHER_PERSON_NAME"}
TITLES = {"dr", "mr", "mrs", "ms", "miss", "doctor", "nurse", "md", "pa", "np", "jr", "sr"}
_WS = re.compile(r"\s+")


def pattern(text: str) -> re.Pattern:
    left = r"(?<!\w)" if text[:1].isalnum() or text[:1] == "_" else ""
    right = r"(?!\w)" if text[-1:].isalnum() or text[-1:] == "_" else ""
    return re.compile(left + re.escape(text) + right, re.IGNORECASE)


def normalize(text: str) -> str:
    """Case- and spacing-insensitive form of the Ketai scorer."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"([.,;:!?()\[\]\"])", r" \1 ", t)
    return _WS.sub(" ", t).strip()


def counter(matcher: str) -> Callable[[str, str], int]:
    """count(needle, haystack) for a matching rule; haystacks are cached per object identity."""
    if matcher == "word":
        return lambda needle, hay: len(pattern(needle).findall(hay))
    if matcher != "normalized":
        raise ValueError(f"unknown matcher {matcher!r}; expected one of {MATCHERS}")
    cache: dict[int, tuple[str, str]] = {}

    def count(needle: str, hay: str) -> int:
        hit = cache.get(id(hay))
        if hit is None or hit[0] is not hay:
            hit = cache[id(hay)] = (hay, normalize(hay))
        n = normalize(needle)
        return len(re.findall(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", hit[1])) if n else 0

    return count


def name_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z']+", normalize(text)) if len(t) >= 3 and t not in TITLES]


def load_gold(gold_dir: str | Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(gold_dir).glob("*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        out[g["encounter_id"]] = g
    return out


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def score(
    original: list[dict],
    anonymized: list[dict],
    gold: dict[str, dict],
    decisions: list[dict] | None = None,
    matcher: str = "word",
) -> dict:
    count = counter(matcher)
    by_id = {r["encounter_id"]: r for r in anonymized}
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # key -> [removed, total]
    entity: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # key -> [fully removed, items]
    keep = defaultdict(lambda: [0.0, 0.0])
    tokens = [0.0, 0.0]
    concepts = {"ALL": [0.0, 0.0], "icd_relevant": [0.0, 0.0]}
    leaks: list[dict] = []
    over: list[dict] = []
    gold_occ: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    n_enc = 0
    for rec in original:
        eid = rec["encounter_id"]
        g = gold.get(eid)
        if g is None or eid not in by_id:
            continue
        n_enc += 1
        anon = by_id[eid]
        seen_tokens: set[tuple[str, str]] = set()
        for it in g["pii"]:
            f = it["field"]
            gold_occ[(eid, f)] += [(m.start(), m.end()) for m in pattern(it["text"]).finditer(rec[f])]
            n0 = count(it["text"], rec[f])
            if n0 == 0:
                continue
            n1 = min(count(it["text"], anon[f]), n0)
            risk = it.get("risk", "high")
            tier = it.get("tier") or category_tier(it["category"])
            keys = ["ALL", f"risk:{risk}", f"tier:{tier}", f"cat:{it['category']}", f"field:{f}"]
            keys.append(f"dataset:{rec['dataset']}")
            if risk in ("high", "medium"):
                keys.append("risk:high+medium")
            for key in keys:
                agg[key][0] += n0 - n1
                agg[key][1] += n0
                entity[key][0] += 1.0 if n1 == 0 else 0.0
                entity[key][1] += 1
            agg["items_fully_removed"][0] += 1.0 if n1 == 0 else 0.0
            agg["items_fully_removed"][1] += 1
            if n1:
                leaks.append(
                    {
                        "encounter_id": eid,
                        "field": f,
                        "text": it["text"],
                        "category": it["category"],
                        "risk": risk,
                        "tier": tier,
                        "left": n1,
                        "of": n0,
                    }
                )
            if it["category"] in NAME_CATEGORIES:
                for t in name_tokens(it["text"]):
                    if (f, t) in seen_tokens:
                        continue
                    seen_tokens.add((f, t))
                    t0 = count(t, rec[f])
                    if t0:
                        tokens[0] += t0 - min(count(t, anon[f]), t0)
                        tokens[1] += t0
        for it in g.get("keep", []):
            f = it["field"]
            n0 = count(it["text"], rec[f])
            if n0 == 0:
                continue
            n1 = min(count(it["text"], anon[f]), n0)
            for key in ("ALL", f"cat:{it['category']}"):
                keep[key][0] += n1
                keep[key][1] += n0
            if n1 < n0:
                over.append(
                    {
                        "encounter_id": eid,
                        "field": f,
                        "text": it["text"],
                        "category": it["category"],
                        "kept": n1,
                        "of": n0,
                    }
                )
        for c in g.get("concepts", []):
            f = c["field"]
            if count(c["text"], rec[f]) == 0:
                continue
            kept = 1.0 if count(c["text"], anon[f]) else 0.0
            for key in ("ALL", "icd_relevant") if c.get("icd_relevant") else ("ALL",):
                concepts[key][0] += kept
                concepts[key][1] += 1
    res = {
        "matcher": matcher,
        "n_encounters": n_enc,
        "removed": {k: _ratio(*v) for k, v in sorted(agg.items())},
        "counts": {k: [int(v[0]), int(v[1])] for k, v in sorted(agg.items())},
        "entity_removed": {k: _ratio(*v) for k, v in sorted(entity.items())},
        "name_token_removed": _ratio(*tokens),
        "name_token_counts": [int(tokens[0]), int(tokens[1])],
        "keep_survival": {k: _ratio(*v) for k, v in sorted(keep.items())},
        "labeled_concept_recall": {k: _ratio(*v) for k, v in concepts.items()},
        "leaks": leaks,
        "over_removed_keep_items": over,
    }
    if decisions is not None:
        tp = fp = 0
        fps: list[dict] = []
        family: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # detector family -> [matching, not]
        for d in decisions:
            if d.get("action") in (None, "KEEP") or d["encounter_id"] not in gold:
                continue
            occ = gold_occ.get((d["encounter_id"], d["field"]), [])
            fam = str(d.get("source") or "?").split(":")[0]
            if any(s < d["end"] and d["start"] < e for s, e in occ):
                tp += 1
                family[fam][0] += 1
            else:
                fp += 1
                family[fam][1] += 1
                fps.append({k: d.get(k) for k in ("encounter_id", "field", "category", "source", "text", "start")})
        res["edit_precision"] = _ratio(tp, tp + fp)
        res["edits_matching_gold"] = tp
        res["edits_not_in_gold"] = fp
        res["edits_not_in_gold_examples"] = fps
        res["edits_by_family"] = {
            k: {"matching_gold": v[0], "not_in_gold": v[1], "precision": _ratio(v[0], v[0] + v[1])}
            for k, v in sorted(family.items())
        }
    return res
