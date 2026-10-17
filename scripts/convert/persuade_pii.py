"""thedrcat/persuade-pii-dataset: 2,218 real student essays (PERSUADE corpus) with synthetic PII inserted, token-labelled.

Each essay becomes a `note` (not medical). `full_text` is exactly the spaCy tokens joined with their trailing
whitespace, so BIO labels map to exact character spans. Mostly NAME_STUDENT; a few addresses, phones, one e-mail.
No ICD-10.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, PII_LABELS, Labels, bio_spans, encounter, parser, sample, write_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "thedrcat__persuade-pii-dataset" / "persuade_train_v0.json")
    a = p.parse_args(argv)
    docs = json.loads(a.src.read_text(encoding="utf-8"))

    def items():
        for d in sample(docs, a.limit, a.seed):
            offsets, pos = [], 0
            for t, ws in zip(d["tokens"], d["trailing_whitespace"]):
                offsets.append((pos, pos + len(t)))
                pos += len(t) + ws
            assert pos == len(d["full_text"]), d["document"]
            enc = encounter(d["document"], "persuade_pii", note=d["full_text"])
            lab = Labels(enc)
            for s, e, typ in bio_spans(d["tokens"], d["labels"], offsets):
                lab.add_pii("note", s, e, PII_LABELS[typ], typ)
            yield enc, lab.record("none", pii_gold="exhaustive", meta={"document": d["document"]})

    write_dataset("persuade_pii", items(), {
        "source": "kaggle:thedrcat/persuade-pii-dataset",
        "language": "en",
        "origin": "real student essays with synthetic PII inserted (not medical)",
        "license": "other (see Kaggle page)",
        "icd10": "none",
        "pii_gold": {"labelled": "PATIENT_NAME (student), LOCATION (street address), ID (phone, e-mail)",
                     "notes": "names of non-students (authors, public figures) are not PII here and unlabelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
