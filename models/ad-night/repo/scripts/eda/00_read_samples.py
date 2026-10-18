"""Dump whole encounters for reading by eye (phase 0 of every analysis).

usage: uv run python scripts/eda/00_read_samples.py D2N088 D2N029 ...
       uv run python scripts/eda/00_read_samples.py --pick   # 4 per dataset
"""
import sys
from common import *

recs = {r["encounter_id"]: r for r in load_all()}
args = sys.argv[1:]
if not args or args == ["--pick"]:
    # fixed reading list: 4 per dataset, mixing train/eval and known quirks
    args = ["D2N083", "D2N123", "D2N114", "D2N058",      # aci
            "D2N088", "D2N131", "D2N092", "D2N134",      # virtassist
            "D2N029", "D2N105", "D2N099", "D2N138"]      # virtscribe
for eid in args:
    r = recs[eid]
    print("#" * 78)
    print(f"# {eid}  dataset={r['dataset']}  split={r['split']}  codes={[c['code'] + ' ' + c['description'] for c in r['icd10_codes']]}")
    print("#" * 78)
    print("---- DIALOGUE ----")
    print(r["dialogue"])
    print("---- NOTE ----")
    print(r["note"])
    print()
