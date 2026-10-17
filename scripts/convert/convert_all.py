"""Run every converter with its defaults, then validate the output.

Usage: python3 -I scripts/convert/convert_all.py [--only NAME ...] [--skip NAME ...] [--aci-gold-dir DIR ...]
A converter whose source folder is missing (not downloaded) is reported and skipped.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate  # noqa: E402

CONVERTERS = [
    "aci_bench",
    "devp_discharge",
    "imerit_scribe",
    "sadiqguru_conversations",
    "mtsamples",
    "pmc_patients",
    "nbme_notes",
    "medconv_corpus",
    "codiesp",
    "symptom2disease",
    "medconv2disease",
    "pii_external",
    "persuade_pii",
    "pii_gemini",
    "pii_gpt",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="*", default=[], choices=CONVERTERS)
    p.add_argument("--skip", nargs="*", default=[], choices=CONVERTERS)
    p.add_argument("--aci-gold-dir", nargs="*", default=[], help="harness privacy gold folders for aci_bench")
    a = p.parse_args()
    failed, done = [], []
    for name in a.only or CONVERTERS:
        if name in a.skip:
            continue
        argv = [x for d in a.aci_gold_dir for x in ("--gold-dir", d)] if name == "aci_bench" else []
        try:
            importlib.import_module(name).main(argv)
            done.append(name)
        except FileNotFoundError as exc:
            print(f"{name}: SKIPPED, source not found ({exc.filename})")
        except Exception:  # keep converting the others; report at the end
            traceback.print_exc()
            failed.append(name)
    out = validate.OUT
    print()
    rc = validate.main([str(out / n) for n in done]) if done else 0
    if failed:
        print(f"FAILED: {failed}")
    return 1 if failed or rc else 0


if __name__ == "__main__":
    sys.exit(main())
