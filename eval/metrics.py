"""Model-agnostic metrics: they compare the original and the anonymized record and never look at how a model works.

- validity:   same records, ids, order and keys; `icd10_codes` byte-identical (a hard challenge constraint). One
              optional extra key is allowed: `explanations`, a list (the explainability bonus may ride on the output).
- privacy:    share of gold PII occurrences no longer present (the organizers' "proportion of removed
              privacy-related information"). An occurrence also counts as leaked when a distinctive token of a
              multi-word item survives (`[PATIENT] Campbell` leaks `Andrew Campbell`). On a merged gold the headline
              is the consensus level, with the union and direct levels, entity-level removal and each source gold
              beside it (eval/docs/GOLD.md).
- keep:       share of must-survive gold strings still present (eponyms, clinical numbers, code evidence, ...).
- edits:      token alignment of original vs anonymized: how much non-PII text survived, and the share of changed
              tokens that were gold PII (edit precision, only on exhaustive gold).
- concepts:   share of gold medical-concept occurrences still present, overall and ICD-relevant only.
- terms:      the same with an independent vocabulary (ICD-10-CM 2023 description words), for suites without
              concept gold. Words inside gold PII are not counted: removing them is privacy, not lost medicine.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from functools import cache, lru_cache
from pathlib import Path

from eval.common import CACHE_DIR, CONTAINER, DATA, ENCOUNTER_KEYS, FIELDS, count, occurrences, ratio, words
from eval.gold.taxonomy import LEVELS, SOURCES, in_level

# Tokens that occur in placeholders and generalizations, so their survival says nothing about a leak.
CUE_WORDS = frozenset(
    """the and of for with from that this mr mrs ms miss dr doctor prof patient pt person name names date dates birth
    year years old age aged month months week weeks day days hospital clinic center centre medical health care
    university college school street road avenue lane drive city state county country park lake mount north south east
    west new saint company inc ltd group office department unit ward room phone email number address location""".split()
)
NUMERIC_CATS = {"ID", "DOB", "DATE", "AGE", "YEAR"}


def validity(original: list[dict], anonymized: list[dict]) -> dict:
    errors: list[str] = []
    if len(original) != len(anonymized):
        errors.append(f"record count {len(anonymized)} != {len(original)}")
    by_id = {r.get("encounter_id"): r for r in anonymized}
    if [r.get("encounter_id") for r in anonymized] != [r["encounter_id"] for r in original]:
        errors.append("encounter ids or their order differ from the input")
    bad = changed = 0
    for r in original:
        a = by_id.get(r["encounter_id"])
        why = []
        if a is None:
            why.append("missing")
        else:
            if "explanations" in a and not isinstance(a["explanations"], list):
                why.append("explanations not a list")
            keys = set(a) - {"explanations"}
            if keys != set(ENCOUNTER_KEYS):
                why.append(f"keys {sorted(keys ^ set(ENCOUNTER_KEYS))}")
            if json.dumps(a.get("icd10_codes"), sort_keys=True) != json.dumps(r["icd10_codes"], sort_keys=True):
                why.append("icd10_codes changed")
            if a.get("dataset") != r["dataset"]:
                why.append("dataset changed")
            if not all(isinstance(a.get(f), str) for f in FIELDS):
                why.append("dialogue/note not strings")
            changed += any(a.get(f) != r[f] for f in FIELDS)
        if why:
            bad += 1
            if len(errors) < 12:
                errors.append(f"{r['encounter_id']}: {', '.join(why)}")
    return {"valid_rate": ratio(len(original) - bad, len(original)), "records_changed": ratio(changed, len(original)), "errors": errors}


NAME_CATS = {"PATIENT_NAME", "PERSON_NAME"}
WORDLISTS = (CONTAINER / "models" / "anonymous-deps" / "american-english-large", Path("/usr/share/dict/american-english-large"), Path("/usr/share/dict/words"))


@cache
def common_words() -> tuple[frozenset[str], str | None]:
    """Lowercase entries of an English word list (Debian wamerican-large): `supermarket`, `classes`. Capitalized
    entries are proper names and stay distinctive. Without a list every long token counts as distinctive."""
    for p in WORDLISTS:
        if p.exists():
            return frozenset(w.strip() for w in p.open(encoding="utf-8", errors="ignore") if w[:1].islower() and "'" not in w), str(p)
    return frozenset(), None


def _distinctive(item: str, category: str) -> list[str]:
    """Tokens whose survival betrays the item: digit runs of 3+, name tokens, and rare words of other items
    (`[OCCUPATION] in a supermarket` does not leak `cashier in a supermarket`, `exercise classes` not `barre classes`)."""
    common = common_words()[0]
    toks = []
    for t, _, _ in words(item):
        if t.isdigit():
            if len(t) >= 3:
                toks.append(t)
        elif category not in NUMERIC_CATS and len(t) >= 4 and t not in CUE_WORDS and (category in NAME_CATS or t not in common):
            toks.append(t)
    return toks


@lru_cache(maxsize=256)
def _kept_tokens(orig: str, anon: str) -> tuple[tuple[str, int, int], ...]:
    """The original's word tokens that survive in the output, by original position (`align`, cached per field)."""
    return tuple((t, s, e) for t, s, e, kept in align(orig, anon) if kept)


def leaked(item: str, category: str, orig: str, anon: str, n0: int) -> int:
    """Leaked occurrences of a gold string: the whole string still present, or an occurrence in which a distinctive
    token survives at its own position. Counting the token's survivors elsewhere in the field would credit them to
    its other uses (`HORNER, Dennis` is leaked even when `Horner's syndrome` is rightly kept), so the check is
    positional: the token alignment of `align` says which original tokens the output kept."""
    worst = min(count(anon, item), n0)
    toks = _distinctive(item, category)
    if len(toks) > 1 or (toks and toks[0] != item.lower()):
        kept = [(s, e) for t, s, e in _kept_tokens(orig, anon) if t in toks]
        hit = sum(any(s0 <= s and e <= e0 for s, e in kept) for s0, e0 in occurrences(orig, item))
        worst = max(worst, min(n0, hit))
    return max(0, worst)


def _outcomes(original: list[dict], anonymized: list[dict], labels: dict[str, dict], items_of) -> tuple[list[tuple[dict, dict, int, int]], int]:
    """(gold dict, item, occurrences n0, leaked n1) of every gold item found in its field, and the number of encounters
    scored. `items_of(gold)` gives an encounter's items, or None to skip it."""
    by_id = {r["encounter_id"]: r for r in anonymized}
    out, n_enc = [], 0
    for r in original:
        g, a = labels.get(r["encounter_id"]), by_id.get(r["encounter_id"])
        items = items_of(g) if g is not None and a is not None else None
        if items is None:
            continue
        n_enc += 1
        for it in items:
            f = it["field"]
            n0 = count(r[f], it["text"])
            if n0:
                out.append((g, it, n0, leaked(it["text"], it["category"], r[f], a.get(f) or "", n0)))
    return out, n_enc


def _summary(rows: list[tuple[dict, dict, int, int]]) -> dict:
    """Occurrence-level removal (all, high risk), entity-level removal (items fully removed), items, occurrences."""
    occ, high, ent = [0, 0], [0, 0], [0, 0]
    for _, it, n0, n1 in rows:
        occ[0], occ[1], ent[0], ent[1] = occ[0] + n0 - n1, occ[1] + n0, ent[0] + (n1 == 0), ent[1] + 1
        if (it.get("risk") or "high") == "high":
            high[0], high[1] = high[0] + n0 - n1, high[1] + n0
    return {"removed": ratio(*occ), "removed_high_risk": ratio(*high), "entity_removed": ratio(*ent), "items": ent[1], "occurrences": occ[1]}


def privacy(original: list[dict], anonymized: list[dict], labels: dict[str, dict], key: str = "pii") -> dict | None:
    """Removal of gold PII. The classic keys (removed, removed_high_risk, breakdown, counts) are on the consensus level
    of a merged gold; an item without vote metadata belongs to every level, so on other suites the levels coincide.
    `levels` and the breakdown keys `level:*`/`entity:*` give each level, `votes:*`/`src:*` split the union by votes and
    supporting gold, and `by_source` scores each gold's own list (`pii_by_source`), so the old per-gold numbers stay
    reproducible. `entity_removed`: share of items with no occurrence and no distinctive token left (strict matching)."""
    rows, n_enc = _outcomes(original, anonymized, labels, lambda g: g.get(key))
    if not n_enc or not rows:
        return None
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # key -> [removed, total]
    leaks: list[dict] = []
    for g, it, n0, n1 in rows:
        f, risk = it["field"], it.get("risk") or "high"
        keys = [f"level:{lvl}" for lvl in LEVELS if in_level(it, lvl)]
        if "votes" in it:
            keys += [f"votes:{it['votes']}"] + [f"src:{s}" for s in it.get("sources", [])]
        if in_level(it, "consensus"):
            keys += ["ALL", f"risk:{risk}", f"cat:{it['category']}", f"field:{f}", f"lang:{g.get('language', 'en')}"]
            agg["items_fully_removed"][0] += n1 == 0
            agg["items_fully_removed"][1] += 1
        for k in keys:
            agg[k][0] += n0 - n1
            agg[k][1] += n0
        for lvl in LEVELS:
            if in_level(it, lvl):
                agg[f"entity:{lvl}"][0] += n1 == 0
                agg[f"entity:{lvl}"][1] += 1
        if n1:
            meta = {k: it[k] for k in ("votes", "direct") if k in it}
            leaks.append({"encounter_id": g["encounter_id"], "field": f, "text": it["text"], "category": it["category"], "risk": risk, **meta,
                          "left": n1, "of": n0})  # fmt: skip
    sources = sorted({s for g in labels.values() for s in g.get("pii_by_source") or {}}, key=lambda s: (SOURCES.index(s) if s in SOURCES else len(SOURCES), s))
    by_source = {s: _summary(_outcomes(original, anonymized, labels, lambda g, s=s: (g.get("pii_by_source") or {}).get(s))[0]) for s in sources}
    return {
        "removed": ratio(*agg["ALL"]),
        "removed_high_risk": ratio(*agg["risk:high"]) if agg["risk:high"][1] else None,
        "entity_removed": ratio(*agg["items_fully_removed"]),
        "levels": {lvl: _summary([x for x in rows if in_level(x[1], lvl)]) for lvl in LEVELS},
        "by_source": by_source,
        "breakdown": {k: ratio(*v) for k, v in sorted(agg.items())},
        "counts": {k: v for k, v in sorted(agg.items())},
        "n_encounters": n_enc,
        "token_wordlist": common_words()[1],
        "leaks": sorted(leaks, key=lambda x: (x["risk"] != "high", x["encounter_id"])),
    }


def survival(original: list[dict], anonymized: list[dict], labels: dict[str, dict], key: str, group: str) -> dict | None:
    """Share of occurrences of must-survive gold strings (`keep`, `concepts`) still present, by `group`."""
    by_id = {r["encounter_id"]: r for r in anonymized}
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    lost: dict[str, int] = defaultdict(int)
    for r in original:
        g, a = labels.get(r["encounter_id"]), by_id.get(r["encounter_id"])
        if g is None or a is None:
            continue
        for it in g.get(key, []):
            f = it["field"]
            n0 = count(r[f], it["text"])
            if not n0:
                continue
            n1 = min(n0, count(a.get(f) or "", it["text"]))
            keys = ["ALL", f"{group}:{it.get(group)}"]
            if it.get("icd_relevant"):
                keys.append("icd_relevant")
            for k in keys:
                agg[k][0] += n1
                agg[k][1] += n0
            if n1 < n0:
                lost[f"{it.get(group)}: {it['text'].lower()}"] += n0 - n1
    if not agg["ALL"][1]:
        return None
    return {
        "kept": ratio(*agg["ALL"]),
        "breakdown": {k: ratio(*v) for k, v in sorted(agg.items())},
        "counts": {k: v for k, v in sorted(agg.items())},
        "top_lost": sorted(lost.items(), key=lambda x: -x[1])[:30],
    }


def _pii_mask(text: str, items: list[dict], field: str) -> list[tuple[int, int]]:
    return [o for it in items if it["field"] == field for o in occurrences(text, it["text"])]


def align(orig: str, anon: str) -> list[tuple[str, int, int, bool]]:
    """Word tokens of `orig` as (token, start, end, kept): kept when the token-level alignment (difflib, no autojunk)
    matches it to a token of `anon`. Shared by `edits` and the explainability score (bonus/explain.py)."""
    ow = words(orig)
    if not ow:
        return []
    sm = difflib.SequenceMatcher(None, [t for t, _, _ in ow], [t for t, _, _ in words(anon)], autojunk=False)
    kept = [False] * len(ow)
    for blk in sm.get_matching_blocks():
        for i in range(blk.a, blk.a + blk.size):
            kept[i] = True
    return [(t, s, e, k) for (t, s, e), k in zip(ow, kept)]


def edits(original: list[dict], anonymized: list[dict], labels: dict[str, dict]) -> dict | None:
    """Token-level alignment (difflib) of each field. Only on exhaustive gold, where unlabelled = not private."""
    by_id = {r["encounter_id"]: r for r in anonymized}
    kept_other = total_other = changed_pii = total_pii = changed = 0
    for r in original:
        g, a = labels.get(r["encounter_id"]), by_id.get(r["encounter_id"])
        if g is None or a is None or g.get("pii_gold") != "exhaustive":
            continue
        for f in FIELDS:
            toks = align(r[f], a.get(f) or "")
            if not toks:
                continue
            spans = _pii_mask(r[f], g["pii"], f)
            is_pii = [any(s < e2 and s2 < e for s2, e2 in spans) for _, s, e, _ in toks]
            for (_, _, _, k), p in zip(toks, is_pii):
                total_pii += p
                total_other += not p
                kept_other += k and not p
                changed_pii += p and not k
                changed += not k
    if not total_other:
        return None
    return {
        "non_pii_tokens_kept": ratio(kept_other, total_other),
        "pii_tokens_changed": ratio(changed_pii, total_pii),
        "edit_precision": ratio(changed_pii, changed),
        "tokens_changed": changed,
    }


STOP_TERMS = frozenset(
    """with without other unspecified specified type site part due than more less also each both from into upon
    initial subsequent encounter sequela episode care status history personal family elsewhere classified
    condition conditions disorder disorders disease diseases syndrome stage degree level region multiple single
    unilateral bilateral left right side sides upper lower first second third fourth fifth sixth seventh eighth ninth
    acute chronic primary secondary nontraumatic traumatic juvenile involving complicating following affecting
    associated resulting caused using after before during under over except mild moderate severe""".split()
)


@cache
def clinical_vocabulary() -> frozenset[str]:
    """Words of the ICD-10-CM 2023 long descriptions (the challenge's code table): an independent medical vocabulary."""
    path = DATA / "kaggle" / "mrhell__icd10cm-codeset-2023" / "icd10cm_codes_2023.txt"
    cached = CACHE_DIR / "clinical_vocabulary.txt"
    if cached.exists():
        return frozenset(cached.read_text(encoding="utf-8").split())
    vocab: set[str] = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            for t, _, _ in words(line[8:]):
                if len(t) >= 4 and not t.isdigit() and t not in STOP_TERMS:
                    vocab.add(t)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text("\n".join(sorted(vocab)), encoding="utf-8")
    return frozenset(vocab)


def terms(original: list[dict], anonymized: list[dict], labels: dict[str, dict]) -> dict | None:
    vocab = clinical_vocabulary()
    if not vocab:
        return None
    by_id = {r["encounter_id"]: r for r in anonymized}
    kept = total = 0
    lost: dict[str, int] = defaultdict(int)
    for r in original:
        a = by_id.get(r["encounter_id"])
        if a is None:
            continue
        g = labels.get(r["encounter_id"]) or {}
        for f in FIELDS:
            spans = _pii_mask(r[f], g.get("pii", []), f)
            co: dict[str, int] = defaultdict(int)
            for t, s, e in words(r[f]):
                if t in vocab and not any(s < e2 and s2 < e for s2, e2 in spans):
                    co[t] += 1
            ca: dict[str, int] = defaultdict(int)
            for t, _, _ in words(a.get(f) or ""):
                if t in co:
                    ca[t] += 1
            for t, n in co.items():
                k = min(n, ca[t])
                kept += k
                total += n
                if k < n:
                    lost[t] += n - k
    if not total:
        return None
    return {"kept": ratio(kept, total), "n_terms": total, "top_lost": sorted(lost.items(), key=lambda x: -x[1])[:30]}


def placeholder_rate(anonymized: list[dict]) -> float | None:
    """Placeholders per 1,000 words of output, a rough over-redaction signal readable without gold."""
    n_words = n_tags = 0
    for a in anonymized:
        for f in FIELDS:
            t = a.get(f) or ""
            n_words += len(words(t))
            n_tags += len(re.findall(r"\[[A-Z][A-Z0-9_ +\-]*\]", t))
    return round(1000 * n_tags / n_words, 2) if n_words else None

