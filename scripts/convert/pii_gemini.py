"""newtonbaba12345/pii-detection-gemini-created-dataset: 3,779 Gemini-written essays with injected PII values.

Each essay becomes a `note` (not medical). The injected values are listed per type (EMAIL, USERNAME, ID_NUM,
PHONE_NUM, URL_PERSONAL, STREET_ADDRESS), without positions: every occurrence is located by string match
(whitespace-insensitive; multi-line addresses fall back to matching each line). Caveat: the lists are the values
offered to the generator, not what it wrote. About half never appear (no listed e-mail does), and essays contain
PII that is not listed (invented phones/IDs, all student names). So matched values are true PII (recall is
meaningful) but the gold is far from exhaustive (precision is not). No ICD-10.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, PII_LABELS, Labels, clean, encounter, parser, read_csv, sample, write_dataset  # noqa: E402

COLUMNS = ("EMAIL", "USERNAME", "ID_NUM", "PHONE_NUM", "URL_PERSONAL", "STREET_ADDRESS")


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "newtonbaba12345__pii-detection-gemini-created-dataset" / "pii_gemini.csv")
    a = p.parse_args(argv)
    rows = read_csv(a.src)
    values: Counter = Counter()
    missing: Counter = Counter()

    def items():
        for r in sample(rows, a.limit, a.seed):
            enc = encounter(f"piigem-{int(r['']):05d}", "pii_gemini", note=clean(r["Essay"]))
            lab = Labels(enc)
            for col in COLUMNS:
                for v in ast.literal_eval(r[col] or "[]"):
                    values[col] += 1
                    n = lab.match_pii(v, PII_LABELS[col], col, fields=["note"])
                    if not n and "\n" in v:
                        n = sum(lab.match_pii(part, PII_LABELS[col], col, fields=["note"]) for part in v.splitlines() if part.strip())
                    missing[col] += not n
            yield enc, lab.record("none", pii_gold="partial", meta={"row": r[""]})

    recs = list(items())
    write_dataset("pii_gemini", recs, {
        "source": "kaggle:newtonbaba12345/pii-detection-gemini-created-dataset",
        "language": "en",
        "origin": "LLM-generated (Gemini) essays with injected synthetic PII (not medical)",
        "license": "Apache 2.0",
        "icd10": "none",
        "pii_gold": {"labelled": "ID (username, ID number, phone, URL), LOCATION (street address)",
                     "notes": "names and unlisted PII are unlabelled; listed values not found in the essay, per type: "
                              + ", ".join(f"{k} {missing[k]}/{values[k]}" for k in COLUMNS)},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
