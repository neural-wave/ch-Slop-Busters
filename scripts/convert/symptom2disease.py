"""niyarrbarman/symptom2disease: 1,200 short first-person symptom descriptions, 24 disease labels (50 each).

Each description becomes a one-turn `[patient]` dialogue. The disease label is mapped to one ICD-10-CM code with
the table below (`icd10_source: mapped`): a fixed, hand-made label -> code mapping using the ICD-10-CM index default
for the unspecified term, not an encounter-level coding, so it is coarse (`diabetes` -> E11.9, `allergy` -> T78.40XA).
Nearly PII-free (one US state, family relations), not labelled.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, dialogue_text, encounter, icd_code, parser, read_csv, sample, write_dataset  # noqa: E402

# Shared with medconv2disease.py (same 24 labels). Every code is checked against ICD-10-CM 2023 by icd_code().
LABEL_TO_ICD10 = {
    "acne": "L70.9",
    "allergy": "T78.40XA",
    "arthritis": "M19.90",
    "bronchial asthma": "J45.909",
    "cervical spondylosis": "M47.812",
    "chicken pox": "B01.9",
    "common cold": "J00",
    "dengue": "A90",
    "diabetes": "E11.9",
    "dimorphic hemorrhoids": "K64.9",
    "drug reaction": "T88.7XXA",
    "fungal infection": "B49",
    "gastroesophageal reflux disease": "K21.9",
    "hypertension": "I10",
    "impetigo": "L01.00",
    "jaundice": "R17",
    "malaria": "B54",
    "migraine": "G43.909",
    "peptic ulcer disease": "K27.9",
    "pneumonia": "J18.9",
    "psoriasis": "L40.9",
    "typhoid": "A01.00",
    "urinary tract infection": "N39.0",
    "varicose veins": "I83.90",
}


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "niyarrbarman__symptom2disease" / "Symptom2Disease.csv")
    a = p.parse_args(argv)
    rows = list(enumerate(read_csv(a.src)))  # the CSV's own index column has duplicates

    def items():
        for i, r in sample(rows, a.limit, a.seed):
            label = r["label"].strip()
            enc = encounter(f"s2d-{i:04d}", "symptom2disease", dialogue=clean(dialogue_text([("patient", r["text"])])),
                            codes=[icd_code(LABEL_TO_ICD10[label.lower()])])  # fmt: skip
            yield enc, Labels(enc).record("mapped", pii_gold="none", meta={"label": label})

    write_dataset("symptom2disease", items(), {
        "source": "kaggle:niyarrbarman/symptom2disease",
        "language": "en",
        "origin": "synthetic symptom descriptions",
        "license": "CC0",
        "icd10": "mapped: disease label -> ICD-10-CM via LABEL_TO_ICD10 (coarse, one code per label)",
        "pii_gold": {"labelled": None, "notes": "nearly PII-free (a US state, family relations); unlabelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
