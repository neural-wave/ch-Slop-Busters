"""tboyle10/medicaltranscriptions (MTSamples): 4,999 dictated notes across 40 specialties, no ICD-10, no PII gold.

Each transcription becomes a `note`. The CSV export replaced line breaks with commas (`HISTORY:,  He ...`,
`cm.,2.  Normal ...`); we restore section headers, numbered lists and paragraph breaks heuristically. Empty
transcriptions (33) are skipped. Specialty, sample name, description and keywords go to `meta`.
Useful for over-redaction tests and note-style robustness; a few fake names/dates are present but unlabelled.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, encounter, parser, read_csv, sample, write_dataset  # noqa: E402

HEADER = re.compile(r"(?:^|,)\s*([A-Z0-9][A-Z0-9 /&()'#-]{1,60}?):(?:\s*,)+\s*")


def restore_layout(text: str) -> str:
    text = HEADER.sub(lambda m: f"\n\n{m.group(1).strip()}:\n" if re.search("[A-Z]", m.group(1)) else m.group(0), text)
    text = re.sub(r"(?<=[.:])\s*,(?=\s*\d{1,2}\.\s)", "\n", text)  # ".,2.  Normal" -> list item per line
    text = re.sub(r"\.\s*,(?=\s*[A-Z])", ".\n", text)  # ".,The patient" -> new paragraph
    return "\n".join(line.strip(" ,") for line in text.splitlines()).strip()


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "tboyle10__medicaltranscriptions" / "mtsamples.csv")
    a = p.parse_args(argv)
    rows = [r for r in read_csv(a.src) if r["transcription"].strip()]

    def items():
        for r in sample(rows, a.limit, a.seed):
            enc = encounter(f"mts-{int(r['']):04d}", "mtsamples", note=clean(restore_layout(r["transcription"])))
            meta = {"specialty": r["medical_specialty"].strip(), "sample_name": r["sample_name"].strip(), "description": r["description"].strip(),
                    "keywords": [k.strip() for k in r["keywords"].split(",") if k.strip()]}  # fmt: skip
            yield enc, Labels(enc).record("none", pii_gold="none", meta=meta)

    write_dataset("mtsamples", items(), {
        "source": "kaggle:tboyle10/medicaltranscriptions (mtsamples.com)",
        "language": "en",
        "origin": "transcribed sample reports",
        "license": "CC0",
        "icd10": "none",
        "pii_gold": {"labelled": None, "notes": "a few fake names and dates, unlabelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
