"""Command coder for the decision-tree ICD-10 model of <container>/decision-models (eval/coders/command.py contract).

Usage: python3 -I eval/adapters/tree_coder.py --input IN.jsonl --output OUT.jsonl [--root DIR] [--scorer tree] [--view full]

decision-models/icd_rules.py codes a record from its note and dialogue with a lexicon of ICD-10 terms and a scorer
fit on the 140 ACI train encounters (`tree`: a depth-2 decision tree over candidate features, the package's
primary scorer). It never reads icd10_codes. It needs numpy only. Writes one {"encounter_id", "codes"} line per
record, codes at 4 characters (`E11.9`, `I10`); its own headline is the 3-character category F1, which the harness
reports as category level. A train encounter is scored by the cross-validation fold model that did not see it, as
in the package's own `run_icd.py predict`.

The cache key of a command coder is its command line: after the models in decision-models/data change, clear
data/eval/cache/icd/tree/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "decision-models"  # <container>/neuralwave/eval/adapters/


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--scorer", default="tree", help="primary | rule | tree | nb | evidence")
    p.add_argument("--view", default="full", help="full (note + dialogue) | dialogue")
    a = p.parse_args(argv)
    sys.path.insert(0, str(a.root))
    import icd_rules as R  # noqa: PLC0415 (lives outside the repo)

    with a.input.open(encoding="utf-8") as fh, a.output.open("w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            pred = R.predict(rec, scorer=a.scorer, fold=R.fold_of(R._view(rec)), view_name=a.view)
            out.write(json.dumps({"encounter_id": rec["encounter_id"], "codes": pred["c4"]}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
