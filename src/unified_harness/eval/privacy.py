"""Privacy metrics against silver labels (`labels/upstream/<split>/<encounter_id>.json`).

Proportion of removed privacy-related information = for every gold surface string, the share of
its occurrences in the original field that no longer occur in the anonymized field (case
insensitive, whole word). Reported overall, by risk tier, category, field and dataset.

Also reported:
- keep-trap survival: gold `keep` strings (eponyms, code evidence, clinical look-alike numbers) that
  must still be present after anonymization (over-removal check);
- edit precision: share of applied edits (decisions sidecar) that overlap a gold PII occurrence.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


def pattern(text: str) -> re.Pattern:
    left = r"(?<!\w)" if text[:1].isalnum() or text[:1] == "_" else ""
    right = r"(?!\w)" if text[-1:].isalnum() or text[-1:] == "_" else ""
    return re.compile(left + re.escape(text) + right, re.IGNORECASE)


def load_gold(gold_dir: str | Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(gold_dir).glob("*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        out[g["encounter_id"]] = g
    return out


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def score(
    original: list[dict], anonymized: list[dict], gold: dict[str, dict], decisions: list[dict] | None = None
) -> dict:
    by_id = {r["encounter_id"]: r for r in anonymized}
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # key -> [removed, total]
    keep = defaultdict(lambda: [0.0, 0.0])
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
        for it in g["pii"]:
            f = it["field"]
            rx = pattern(it["text"])
            occ = [(m.start(), m.end()) for m in rx.finditer(rec[f])]
            gold_occ[(eid, f)] += occ
            n0 = len(occ)
            if n0 == 0:
                continue
            n1 = min(len(rx.findall(anon[f])), n0)
            removed = (n0 - n1) / n0  # per-item share; aggregate below is occurrence weighted
            risk = it.get("risk", "high")
            for key in ("ALL", f"risk:{risk}", f"cat:{it['category']}", f"field:{f}", f"dataset:{rec['dataset']}"):
                agg[key][0] += n0 - n1
                agg[key][1] += n0
            if risk in ("high", "medium"):
                agg["risk:high+medium"][0] += n0 - n1
                agg["risk:high+medium"][1] += n0
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
                        "left": n1,
                        "of": n0,
                    }
                )
            _ = removed
        for it in g.get("keep", []):
            f = it["field"]
            rx = pattern(it["text"])
            n0 = len(rx.findall(rec[f]))
            if n0 == 0:
                continue
            n1 = min(len(rx.findall(anon[f])), n0)
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
    res = {
        "n_encounters": n_enc,
        "removed": {k: _ratio(*v) for k, v in sorted(agg.items())},
        "counts": {k: [int(v[0]), int(v[1])] for k, v in sorted(agg.items())},
        "keep_survival": {k: _ratio(*v) for k, v in sorted(keep.items())},
        "leaks": leaks,
        "over_removed_keep_items": over,
    }
    if decisions is not None:
        tp = fp = 0
        fps: list[dict] = []
        for d in decisions:
            if d.get("action") in (None, "KEEP") or d["encounter_id"] not in gold:
                continue
            occ = gold_occ.get((d["encounter_id"], d["field"]), [])
            if any(s < d["end"] and d["start"] < e for s, e in occ):
                tp += 1
            else:
                fp += 1
                fps.append({k: d.get(k) for k in ("encounter_id", "field", "category", "source", "text", "start")})
        res["edit_precision"] = _ratio(tp, tp + fp)
        res["edits_matching_gold"] = tp
        res["edits_not_in_gold"] = fp
        res["edits_not_in_gold_examples"] = fps
    return res
