"""Check converted datasets against the format in README.md and print one summary row per dataset.

Usage: python3 -I scripts/convert/validate.py [data/converted/<name> ...]   (default: every folder in data/converted)
Exits 1 if any dataset has errors. Unknown ICD-10 codes are reported as warnings (challenge data uses a few codes
newer than ICD-10-CM 2023).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORY_RISK, ENCOUNTER_KEYS, FIELDS, ICD_SOURCES, KEEP_CATEGORIES, OUT, PII_GOLD, icd10cm  # noqa: E402

CODE = re.compile(r"^[A-Z]\d[0-9A-Z](\.[0-9A-Z]{1,4})?$")
LABEL_KEYS = ("encounter_id", "icd10_source", "pii_gold", "pii", "keep", "meta")


def check(d: Path) -> tuple[list[str], list[str], dict]:
    errs, warns = [], []
    encs = [json.loads(line) for line in (d / "encounters.jsonl").open(encoding="utf-8")]
    labs = [json.loads(line) for line in (d / "labels.jsonl").open(encoding="utf-8")]
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if len(encs) != len(labs) or manifest.get("records") != len(encs):
        errs.append(f"counts differ: {len(encs)} encounters, {len(labs)} labels, manifest {manifest.get('records')}")
    if len({e["encounter_id"] for e in encs}) != len(encs):
        errs.append("duplicate encounter_id")
    unknown: Counter = Counter()
    cats: Counter = Counter()
    for e, lab in zip(encs, labs):
        eid = e.get("encounter_id")
        if tuple(e) != ENCOUNTER_KEYS:
            errs.append(f"{eid}: encounter keys {list(e)}")
            continue
        if not all(isinstance(e[k], str) for k in ("encounter_id", "dataset", *FIELDS)) or not (e["dialogue"] or e["note"]):
            errs.append(f"{eid}: text fields must be strings, at least one non-empty")
        for c in e["icd10_codes"]:
            if set(c) != {"code", "description"} or not CODE.match(c["code"]) or not isinstance(c["description"], str):
                errs.append(f"{eid}: bad icd10 entry {c}")
            elif c["code"] not in icd10cm():
                unknown[c["code"]] += 1
        if tuple(lab) != LABEL_KEYS or lab["encounter_id"] != eid:
            errs.append(f"{eid}: labels line keys {list(lab)} / id {lab.get('encounter_id')}")
            continue
        if lab["icd10_source"] not in ICD_SOURCES or (lab["icd10_source"] == "none") != (not e["icd10_codes"]):
            errs.append(f"{eid}: icd10_source {lab['icd10_source']!r} with {len(e['icd10_codes'])} codes")
        if lab["pii_gold"] not in PII_GOLD or (lab["pii_gold"] == "none" and lab["pii"]):
            errs.append(f"{eid}: pii_gold {lab['pii_gold']!r} with {len(lab['pii'])} spans")
        if not isinstance(lab["meta"].get("language"), str):
            errs.append(f"{eid}: meta.language missing")
        for kind, allowed in (("pii", CATEGORY_RISK), ("keep", KEEP_CATEGORIES)):
            prev: dict[str, int] = {}
            for it in lab[kind]:
                f, s, t = it.get("field"), it.get("start", -1), it.get("end", -1)
                if f not in FIELDS or not 0 <= s < t <= len(e[f]) or e[f][s:t] != it.get("text"):
                    errs.append(f"{eid}: {kind} span does not match its field: {it}")
                elif it.get("category") not in allowed:
                    errs.append(f"{eid}: {kind} category {it.get('category')!r}")
                elif kind == "pii" and s < prev.get(f, 0):
                    errs.append(f"{eid}: pii spans overlap or are unsorted at {f}[{s}:{t}]")
                else:
                    prev[f] = t
                    if kind == "pii":
                        cats[it["category"]] += 1
    if unknown:
        warns.append(f"{sum(unknown.values())} codes not in ICD-10-CM 2023: {dict(unknown.most_common(8))}")
    row = {
        "dataset": d.name,
        "n": len(encs),
        "dialogue": sum(bool(e["dialogue"]) for e in encs),
        "note": sum(bool(e["note"]) for e in encs),
        "icd": ",".join(f"{k}:{v}" for k, v in Counter(lab["icd10_source"] for lab in labs).items()),
        "pii_gold": ",".join(f"{k}:{v}" for k, v in Counter(lab["pii_gold"] for lab in labs).items()),
        "pii_spans": sum(cats.values()),
        "top_categories": " ".join(f"{k}:{v}" for k, v in cats.most_common(4)),
    }
    return errs, warns, row


def main(argv: list[str] | None = None) -> int:
    dirs = [Path(x) for x in (argv if argv is not None else sys.argv[1:])] or sorted(p for p in OUT.iterdir() if (p / "encounters.jsonl").exists())
    rows, bad = [], 0
    for d in dirs:
        errs, warns, row = check(d)
        rows.append(row)
        for w in warns:
            print(f"WARN  {d.name}: {w}")
        for x in errs[:10]:
            print(f"ERROR {d.name}: {x}")
        if len(errs) > 10:
            print(f"ERROR {d.name}: ... {len(errs) - 10} more")
        bad += bool(errs)
    cols = list(rows[0]) if rows else []
    width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(width[c]) for c in cols))
    print(f"{len(rows)} datasets, {sum(r['n'] for r in rows)} encounters, {bad} with errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
