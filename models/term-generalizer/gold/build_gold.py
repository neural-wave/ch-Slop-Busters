"""Consolidate per-encounter gold annotations (gold/eval/<id>.json) into gold/eval_gold.jsonl.

Validates that every span is a verbatim substring of its field, drops/repairs the ones that are not
(case-insensitive fallback), dedupes, removes PII that the annotation itself lists as a non-PII ambiguous
term (medical eponyms, brand names), and prints stats.

The annotations were produced by LLM agents from data/, not by the organizers; run audit_gold.py to
cross-check them against data/ before relying on the reference metrics.

  python3 ../term-generalizer/gold/build_gold.py                 # gold/eval/*.json   -> gold/eval_gold.jsonl
  python3 ../term-generalizer/gold/build_gold.py --split train   # gold/train/*.json  -> gold/train_gold.jsonl (silver)

The optional `evidence` list (code -> supporting concept spans, see SPEC.md) is validated the same way and kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_DIR, EVAL_DATA, read_jsonl, write_jsonl  # noqa: E402

GOLD_DIR = Path(__file__).resolve().parent
SPLIT_DATA = {"eval": EVAL_DATA, "train": DATA_DIR / "patient_doctor_encounters.jsonl"}
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLIT_DATA), default="eval")
    args = ap.parse_args()
    ANN_DIR, out_path = GOLD_DIR / args.split, GOLD_DIR / f"{args.split}_gold.jsonl"
    records = {r["encounter_id"]: r for r in read_jsonl(SPLIT_DATA[args.split])}
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
        if "evidence" in ann:
            codes = {c["code"] for c in rec["icd10_codes"]}
            g["evidence"] = []
            for ev in ann["evidence"] or []:
                if ev.get("code") not in codes:
                    stats["evidence_bad_code"] += 1
                    continue
                spans = []
                for s in ev.get("spans", []) or []:
                    f = s.get("field")
                    t = locate(str(s.get("text", "")).strip(), rec[f]) if f in ("dialogue", "note") else None
                    if t is None:
                        stats["evidence_not_verbatim"] += 1
                        continue
                    spans.append({"text": t, "field": f})
                g["evidence"].append({"code": ev["code"], "spans": spans})
        out.append(g)

    write_jsonl(out_path, out)
    pii = [e for g in out for e in g["pii"]]
    med = [e for g in out for e in g["medical_concepts"]]
    print(f"wrote {out_path} : {len(out)} encounters ({len(missing)} missing annotation files: {missing})")
    print(f"PII strings: {len(pii)}  by category: {dict(Counter(e['category'] for e in pii).most_common())}")
    print(f"medical concepts: {len(med)}  icd_relevant: {sum(bool(e.get('icd_relevant')) for e in med)}")
    print(f"ambiguous terms: {sum(len(g['ambiguous_terms']) for g in out)}")
    print(f"validation: {dict(stats)}")


if __name__ == "__main__":
    main()
