"""Helper for annotators (see SPEC.md): print an encounter, or validate its annotation file.

  python3 ../term-generalizer/gold/ann_tool.py show  D2N088 --split train
  python3 ../term-generalizer/gold/ann_tool.py check D2N088 --split train    # exit code 1 if there are problems
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_DIR, EVAL_DATA, read_jsonl  # noqa: E402

GOLD_DIR = Path(__file__).resolve().parent
SPLIT_DATA = {"eval": EVAL_DATA, "train": DATA_DIR / "patient_doctor_encounters.jsonl"}
PII_CATS = {"PATIENT_NAME", "CLINICIAN_NAME", "OTHER_PERSON_NAME", "DATE", "CONTACT", "ID", "AGE", "LOCATION",
            "ORGANIZATION", "OCCUPATION", "OTHER_QUASI"}
CONCEPT_TYPES = {"diagnosis", "symptom", "finding", "lab_or_test", "medication", "procedure", "anatomy"}
FIELDS = ("dialogue", "note")


def load_record(eid: str, split: str) -> dict:
    for r in read_jsonl(SPLIT_DATA[split]):
        if r["encounter_id"] == eid:
            return r
    raise SystemExit(f"{eid} not in the {split} split")


def show(rec: dict) -> None:
    print(f"# {rec['encounter_id']} ({rec['dataset']})\n")
    print("## icd10_codes")
    for c in rec["icd10_codes"]:
        print(f"- {c['code']}: {c['description']}")
    for f in FIELDS:
        print(f"\n## {f}\n{rec[f]}")


def check(rec: dict, path: Path) -> list[str]:
    if not path.exists():
        return [f"missing file {path}"]
    try:
        ann = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]
    probs = []
    if ann.get("encounter_id") != rec["encounter_id"]:
        probs.append(f"encounter_id is {ann.get('encounter_id')!r}, expected {rec['encounter_id']!r}")

    def verbatim(where: str, e: dict) -> bool:
        f, t = e.get("field"), str(e.get("text", ""))
        if f not in FIELDS:
            probs.append(f"{where}: field must be dialogue or note, got {f!r}")
            return False
        if not t.strip() or t.lower() not in rec[f].lower():
            probs.append(f"{where}: not a verbatim substring of {f}: {t!r}")
            return False
        return True

    for i, e in enumerate(ann.get("pii", [])):
        verbatim(f"pii[{i}]", e)
        if str(e.get("category", "")).upper() not in PII_CATS:
            probs.append(f"pii[{i}]: unknown category {e.get('category')!r}")
    concepts = {}
    for i, e in enumerate(ann.get("medical_concepts", [])):
        if verbatim(f"medical_concepts[{i}]", e):
            concepts[(e["field"], e["text"].lower())] = bool(e.get("icd_relevant"))
        if e.get("type") not in CONCEPT_TYPES:
            probs.append(f"medical_concepts[{i}]: unknown type {e.get('type')!r}")
        if not isinstance(e.get("icd_relevant"), bool):
            probs.append(f"medical_concepts[{i}]: icd_relevant must be true or false")
    for i, e in enumerate(ann.get("ambiguous_terms", [])):
        verbatim(f"ambiguous_terms[{i}]", e)
        if not isinstance(e.get("is_pii"), bool):
            probs.append(f"ambiguous_terms[{i}]: is_pii must be true or false")

    pii_keys = {(e.get("field"), str(e.get("text", "")).lower()) for e in ann.get("pii", [])}
    for k in pii_keys & set(concepts):
        probs.append(f"labelled both PII and medical concept: {k[1]!r} in {k[0]} (pick one, see SPEC.md)")

    codes = [c["code"] for c in rec["icd10_codes"]]
    ev_codes = [e.get("code") for e in ann.get("evidence", [])]
    for c in codes:
        if c not in ev_codes:
            probs.append(f"evidence: no entry for code {c}")
    for c in ev_codes:
        if c not in codes:
            probs.append(f"evidence: {c!r} is not one of this encounter's codes {codes}")
    for e in ann.get("evidence", []):
        for j, s in enumerate(e.get("spans", [])):
            where = f"evidence[{e.get('code')}].spans[{j}]"
            if verbatim(where, s) and concepts.get((s["field"], s["text"].lower())) is not True:
                probs.append(f"{where}: {s['text']!r} must also be a medical_concept with icd_relevant true")
    if not ann.get("pii"):
        probs.append("warning: no PII at all (every encounter so far has some; re-read for names, ages, dates)")
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "check"])
    ap.add_argument("encounter_id")
    ap.add_argument("--split", choices=list(SPLIT_DATA), default="train")
    args = ap.parse_args()
    rec = load_record(args.encounter_id, args.split)
    if args.cmd == "show":
        show(rec)
        return
    probs = check(rec, GOLD_DIR / args.split / f"{args.encounter_id}.json")
    print("\n".join(probs) if probs else "OK")
    sys.exit(1 if any(not p.startswith("warning") for p in probs) else 0)


if __name__ == "__main__":
    main()
