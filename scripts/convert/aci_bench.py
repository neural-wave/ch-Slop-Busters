"""ACI-Bench (challenge data): 140 train + 67 eval encounters, already in the target schema, gold ICD-10.

Copies both splits unchanged (split in `meta.split`). With `--gold-dir` it also imports the text-based privacy
gold of the evaluation harness (`<dir>/<encounter_id>.json` with items `{field, text, category, risk}`) as spans:
every case-insensitive whole-word occurrence of each gold string, which is how the harness counts them.
Encounters with a gold file get `pii_gold: exhaustive` (the harness gold is exhaustive for high-risk items), the
others `none`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Labels, encounter, parser, sample, write_dataset  # noqa: E402

SPLITS = {"train": "patient_doctor_encounters.jsonl", "eval": "eval_patient_doctor_encounters.jsonl"}


def load_gold(dirs: list[Path]) -> dict[str, dict]:
    gold = {}
    for d in dirs:
        if not d.is_dir():
            raise ValueError(f"--gold-dir {d} is not a directory")
        for p in sorted(d.glob("*.json")):
            g = json.loads(p.read_text(encoding="utf-8"))
            gold[g["encounter_id"]] = g
    return gold


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, DATA)
    p.add_argument("--gold-dir", type=Path, action="append", default=[], help="harness gold folder (repeatable), e.g. outputs/gold/eval")
    a = p.parse_args(argv)
    gold = load_gold(a.gold_dir)
    unmatched = 0

    def items():
        nonlocal unmatched
        for split, fname in SPLITS.items():
            with (a.src / fname).open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            for r in sample(rows, a.limit, a.seed):
                enc = encounter(r["encounter_id"], r["dataset"], dialogue=r["dialogue"], note=r["note"], codes=r["icd10_codes"])
                lab = Labels(enc)
                g = gold.get(r["encounter_id"])
                for it in (g or {}).get("pii", []):
                    unmatched += not lab.match_pii(it["text"], it["category"], it["category"], fields=[it["field"]], risk=it.get("risk"))
                for it in (g or {}).get("keep", []):
                    lab.match_keep(it["text"], it["category"], it.get("why", ""), fields=[it["field"]])
                yield enc, lab.record("gold", pii_gold="exhaustive" if g is not None else "none", meta={"split": split})

    recs = list(items())  # materialized so `unmatched` is final when the manifest is built
    write_dataset("aci_bench", recs, {
        "source": "Neural Wave challenge files (ACI-Bench, Yuan, Moy et al. 2025)",
        "language": "en",
        "origin": "role-played doctor-patient encounters + physician notes",
        "license": "challenge data (see task PDF)",
        "icd10": "gold: challenge icd10_codes",
        "pii_gold": {
            "labelled": "harness categories, exhaustive for high-risk items" if gold else None,
            "notes": f"imported from {[str(d) for d in a.gold_dir]}; {unmatched} gold strings gave no span (absent from the field or overlapping another item)" if gold else "pass --gold-dir to import the harness gold",
        },
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
