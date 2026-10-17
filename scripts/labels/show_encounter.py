"""Print one encounter (codes, numbered dialogue lines, note) for annotation.

Usage: uv run python scripts/labels/show_encounter.py D2N004
"""

from __future__ import annotations

import json
import sys

from unified_harness import paths

DATA = [paths.split_file("train"), paths.split_file("eval")]


def main(eid: str) -> None:
    for p in DATA:
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["encounter_id"] != eid:
                continue
            print(f"# {eid}  dataset={r['dataset']}  split={'eval' if 'eval_' in p.name else 'train'}")
            print("## icd10_codes")
            for c in r["icd10_codes"]:
                print(f"- {c['code']}: {c['description']}")
            print("## dialogue")
            for i, ln in enumerate(r["dialogue"].split("\n")):
                print(f"{i:3d}| {ln}")
            print("## note")
            print(r["note"])
            return
    sys.exit(f"unknown encounter {eid}")


if __name__ == "__main__":
    main(sys.argv[1])
