"""Consolidate per-encounter gold annotations (gold/eval/<id>.json) into gold/eval_gold.jsonl.

Validates that every span is a verbatim substring of its field, drops/repairs the ones that are not
(case-insensitive fallback), dedupes, removes PII that the annotation itself lists as a non-PII ambiguous
term (medical eponyms, brand names), and prints stats.

The annotations were produced by LLM agents from data/, not by the organizers; run audit_gold.py to
cross-check them against data/ before relying on the reference metrics.

  python3 harness/gold/build_gold.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import EVAL_DATA, GOLD_EVAL, read_jsonl, write_jsonl  # noqa: E402

ANN_DIR = Path(__file__).resolve().parent / "eval"
PII_CATS = {"PATIENT_NAME": "direct", "CLINICIAN_NAME": "direct", "OTHER_PERSON_NAME": "direct", "DATE": "direct",
            "CONTACT": "direct", "ID": "direct", "AGE": "quasi", "LOCATION": "quasi", "ORGANIZATION": "quasi",
            "OCCUPATION": "quasi", "OTHER_QUASI": "quasi"}
NOT_PII = {"dax", "dragon", "doctor", "patient", "nurse",
           # bare weekdays carry no identifying power (the spec excludes vague relative time)
           "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}


def locate(text: str, field_text: str) -> str | None:
    if text in field_text:
        return text
    i = field_text.lower().find(text.lower())
    return field_text[i:i + len(text)] if i != -1 else None


def main():
    records = {r["encounter_id"]: r for r in read_jsonl(EVAL_DATA)}
    out, stats = [], Counter()
    missing = [i for i in records if not (ANN_DIR / f"{i}.json").exists()]
    for eid, rec in records.items():
        p = ANN_DIR / f"{eid}.json"
        if not p.exists():
            continue
        try:
            ann = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"! {eid}: invalid JSON ({e})")
            stats["invalid_json"] += 1
            continue
        g = {"encounter_id": eid, "pii": [], "medical_concepts": [], "ambiguous_terms": []}
        # terms the annotator judged ambiguous and not PII (eponyms such as "Cushing's", brand names)
        not_pii_terms = {(t.get("field"), str(t.get("text", "")).strip().lower())
                         for t in ann.get("ambiguous_terms", []) or [] if not t.get("is_pii")}
        for key in g:
            if key == "encounter_id":
                continue
            seen = set()
            for e in ann.get(key, []) or []:
                f = e.get("field")
                if f not in ("dialogue", "note") or not str(e.get("text", "")).strip():
                    stats[f"{key}_bad_entry"] += 1
                    continue
                t = locate(e["text"].strip(), rec[f])
                if t is None:
                    stats[f"{key}_not_verbatim"] += 1
                    continue
                if (f, t.lower()) in seen:
                    continue
                if key == "pii":
                    cat = str(e.get("category", "")).upper()
                    if cat not in PII_CATS or t.lower() in NOT_PII:
                        stats["pii_dropped_category"] += 1
                        continue
                    if (f, t.lower()) in not_pii_terms:
                        stats["pii_dropped_eponym"] += 1
                        continue
                    e = {**e, "category": cat, "tier": PII_CATS[cat]}
                seen.add((f, t.lower()))
                g[key].append({**e, "text": t, "field": f})
        out.append(g)

    write_jsonl(GOLD_EVAL, out)
    pii = [e for g in out for e in g["pii"]]
    med = [e for g in out for e in g["medical_concepts"]]
    print(f"wrote {GOLD_EVAL} : {len(out)} encounters ({len(missing)} missing annotation files: {missing})")
    print(f"PII strings: {len(pii)}  by category: {dict(Counter(e['category'] for e in pii).most_common())}")
    print(f"medical concepts: {len(med)}  icd_relevant: {sum(bool(e.get('icd_relevant')) for e in med)}")
    print(f"ambiguous terms: {sum(len(g['ambiguous_terms']) for g in out)}")
    print(f"validation: {dict(stats)}")


if __name__ == "__main__":
    main()
