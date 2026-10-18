"""Show every occurrence of a string in an encounter: original context, anonymized context, decisions.

Usage: uv run python scripts/debug_span.py D2N083 lawrence [--preset balanced]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from neural_wave.config import make_config
from neural_wave.io import read_jsonl
from neural_wave.pipeline import Anonymizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("text")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--no-codes", action="store_true")
    a = ap.parse_args()
    recs = read_jsonl(ROOT / "resources/data/patient_doctor_encounters.jsonl") + read_jsonl(ROOT / "resources/data/eval_patient_doctor_encounters.jsonl")
    rec = next(r for r in recs if r["encounter_id"] == a.eid)
    an = Anonymizer(make_config(a.preset, explain_text=True, use_codes=not a.no_codes))
    out, dec, st = an.anonymize(rec)
    rx = re.compile(r"(?<!\w)" + re.escape(a.text.lower()) + r"(?!\w)")
    for f in ("dialogue", "note"):
        for m in rx.finditer(rec[f].lower()):
            s, e = m.start(), m.end()
            print(f"[{f} @{s}] …{rec[f][max(0, s - 70):e + 50]!r}")
            for d in dec:
                if d["field"] == f and d["start"] < e and s < d["end"]:
                    print(f"    -> {d['action']:10s} {d['category']:15s} {d['source']:28s} {d.get('policy_reason', '')[:70]}")
        for m in rx.finditer(out[f].lower()):
            print(f"  LEFT in anonymized {f}: …{out[f][max(0, m.start() - 70):m.end() + 40]!r}")
    print(st["entities"])


if __name__ == "__main__":
    main()
