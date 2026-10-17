"""pjmathematician/pii-detection-dataset-gpt: 2,000 GPT-written essays with a dict of the injected PII per essay.

Each essay becomes a `note` (not medical). The dict (NAME_STUDENT tokens, EMAIL, USERNAME, ID_NUM, PHONE_NUM,
URL_PERSONAL, STREET_ADDRESS) has no positions: name tokens are matched case-sensitively as whole words, the other
values whitespace-insensitively; multi-line addresses fall back to matching each line. No ICD-10.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, PII_LABELS, Labels, clean, encounter, parser, read_csv, sample, write_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "pjmathematician__pii-detection-dataset-gpt" / "ai_data.csv")
    a = p.parse_args(argv)
    rows = list(enumerate(read_csv(a.src)))  # header row is just "0","1": essay, PII dict
    values = missing = 0

    def items():
        nonlocal values, missing
        for i, r in sample(rows, a.limit, a.seed):
            enc = encounter(f"piigpt-{i:05d}", "pii_gpt", note=clean(r["0"]))
            lab = Labels(enc)
            for typ, vals in ast.literal_eval(r["1"]).items():
                for v in vals:
                    values += 1
                    name = typ == "NAME_STUDENT"
                    n = lab.match_pii(v, PII_LABELS[typ], typ, fields=["note"], ignore_case=not name)
                    if not n and "\n" in v:
                        n = sum(lab.match_pii(part, PII_LABELS[typ], typ, fields=["note"]) for part in v.splitlines() if part.strip())
                    missing += not n
            yield enc, lab.record("none", pii_gold="partial", meta={"row": i})

    recs = list(items())
    write_dataset("pii_gpt", recs, {
        "source": "kaggle:pjmathematician/pii-detection-dataset-gpt",
        "language": "en",
        "origin": "LLM-generated (GPT) essays with injected synthetic PII (not medical)",
        "license": "CC BY 4.0",
        "icd10": "none",
        "pii_gold": {"labelled": "PATIENT_NAME (student), ID (email, username, ID number, phone, URL), LOCATION (street address)",
                     "notes": f"{missing} of {values} listed values not found in the essay"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
