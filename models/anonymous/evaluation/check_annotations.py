"""Validate annotation JSON files: schema + every text must be a (case-insensitive) substring of its field."""
import json
import sys
from pathlib import Path

PHI_TYPES = {"NAME_PATIENT", "NAME_RELATIVE", "NAME_CLINICIAN", "NAME_OTHER", "AGE", "DATE", "LOCATION",
             "ORGANIZATION", "CONTACT", "ID", "OCCUPATION", "ACTIVITY", "LIFE_EVENT", "TEMPORAL", "PERSONAL_DETAIL"}
MED_TYPES = {"DIAGNOSIS", "SYMPTOM", "MEDICATION", "PROCEDURE", "TEST", "FINDING"}


def load_split(path):
    return {r["encounter_id"]: r for r in map(json.loads, open(path))}


def check(ann_path, records):
    errs = []
    try:
        ann = json.load(open(ann_path))
    except Exception as e:  # noqa: BLE001
        return [f"invalid JSON: {e}"]
    eid = ann.get("encounter_id")
    if eid not in records:
        return [f"unknown encounter_id {eid!r}"]
    rec = records[eid]
    for key, types in (("phi", PHI_TYPES), ("medical_concepts", MED_TYPES)):
        for i, it in enumerate(ann.get(key, [])):
            if not isinstance(it, dict) or not {"text", "field", "type"} <= it.keys():
                errs.append(f"{key}[{i}] missing keys: {it}")
                continue
            if it["field"] not in ("dialogue", "note"):
                errs.append(f"{key}[{i}] bad field {it['field']!r}")
                continue
            if it["type"] not in types:
                errs.append(f"{key}[{i}] bad type {it['type']!r}")
            if it["text"].lower() not in rec[it["field"]].lower():
                errs.append(f"{key}[{i}] text not found verbatim in {it['field']}: {it['text']!r}")
    return errs


if __name__ == "__main__":
    split_file, *files = sys.argv[1:]
    records = load_split(split_file)
    bad = 0
    for f in files:
        for e in check(f, records):
            bad += 1
            print(f"{Path(f).name}: {e}")
    print(f"checked {len(files)} files, {bad} problems")
