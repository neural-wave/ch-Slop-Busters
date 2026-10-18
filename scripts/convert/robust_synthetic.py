"""Robustness set R3: 24 synthetic hospital documents written by the A+D team with exact gold (copied, not generated).

ER triage records, discharge letters, referral e-mails, nursing handoffs, radiology/pathology reports and phone triage
calls (`dataset` is the document type), with hand-assigned ICD-10 codes. Source: the A+D model repo,
`outputs/robustness/synthetic.jsonl` and the gold `outputs/gold/robust_synthetic/<encounter_id>.json` (text-based
items `{field, text, category, risk}` and `keep` traps). The gold strings become spans of each whole-word occurrence,
longest item first. Codes and descriptions are kept as written (a few codes are newer than ICD-10-CM 2023).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, REPO, Labels, encounter, write_dataset  # noqa: E402

AD_OUTPUTS = REPO.parent / "models" / "ad-night" / "repo" / "outputs"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--src", type=Path, default=AD_OUTPUTS, help="the A+D repo's outputs/ folder (default: <container>/models/ad-night/repo/outputs)")
    p.add_argument("--out", type=Path, default=OUT, help="output root (default: data/converted)")
    a = p.parse_args(argv)
    gold_dir = a.src / "gold" / "robust_synthetic"
    recs = [json.loads(line) for line in (a.src / "robustness" / "synthetic.jsonl").open(encoding="utf-8") if line.strip()]
    if not gold_dir.is_dir():
        raise FileNotFoundError(2, "gold folder not found", str(gold_dir))
    lost = 0

    def items():
        nonlocal lost
        for r in recs:
            enc = encounter(r["encounter_id"], r["dataset"], dialogue=r["dialogue"], note=r["note"],
                            codes=[{"code": c["code"], "description": c["description"]} for c in r["icd10_codes"]])  # fmt: skip
            g = json.loads((gold_dir / f"{r['encounter_id']}.json").read_text(encoding="utf-8"))
            lab = Labels(enc)
            for it in sorted(g["pii"], key=lambda it: -len(it["text"])):
                lost += not lab.match_pii(it["text"], it["category"], it.get("label", it["category"]), fields=[it["field"]], risk=it.get("risk"))
            for it in g.get("keep", []):
                lab.match_keep(it["text"], it["category"], it.get("why", ""), fields=[it["field"]])
            yield enc, lab.record("gold" if enc["icd10_codes"] else "none", pii_gold="exhaustive", meta={"document_type": r["dataset"]})

    out = list(items())
    write_dataset("robust_synthetic", out, {
        "source": "A+D model repo (models/ad-night/repo/outputs/robustness/synthetic.jsonl + outputs/gold/robust_synthetic/)",
        "stage": "synthetic",
        "language": "en",
        "origin": "synthetic hospital documents written at development time by the A+D team (bonus task: robustness)",
        "license": "team-made",
        "icd10": "gold: hand-assigned by the authors, descriptions as written",
        "pii_gold": {"labelled": "every personal value (A+D gold), plus keep traps", "notes": f"{lost} gold strings gave no span (only inside a longer item)"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
