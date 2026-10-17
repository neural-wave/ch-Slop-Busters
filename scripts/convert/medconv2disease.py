"""artemminiailo/medicalconversations2disease: 960 short ChatGPT patient-bot chats, same 24 labels as symptom2disease.

`User:` / `Bot:` turns (separated by `</s>`) become `[patient]` / `[doctor]` lines. The label is mapped to ICD-10-CM
with symptom2disease.LABEL_TO_ICD10 (`icd10_source: mapped`). Nearly PII-free, not labelled: travel destinations
(Southeast Asia, India, Africa) are malaria/dengue/typhoid evidence, one child's age.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, dialogue_text, encounter, icd_code, parser, read_csv, sample, write_dataset  # noqa: E402
from symptom2disease import LABEL_TO_ICD10  # noqa: E402

SPEAKER = {"user": "patient", "bot": "doctor"}


def turns(conv: str) -> list[tuple[str, str]]:
    out = []
    for chunk in conv.split("</s>"):
        m = re.match(r"\s*(User|Bot)\s*:\s*(.*)", chunk, re.S | re.I)
        if m:
            out.append((SPEAKER[m.group(1).lower()], m.group(2)))
    return out


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "artemminiailo__medicalconversations2disease" / "medical_conversations.csv")
    a = p.parse_args(argv)
    rows = list(enumerate(read_csv(a.src)))

    def items():
        for i, r in sample(rows, a.limit, a.seed):
            label = r["disease"].strip()
            enc = encounter(f"mc2d-{i:04d}", "medconv2disease", dialogue=clean(dialogue_text(turns(r["conversations"]))),
                            codes=[icd_code(LABEL_TO_ICD10[label.lower()])])  # fmt: skip
            yield enc, Labels(enc).record("mapped", pii_gold="none", meta={"label": label})

    write_dataset("medconv2disease", items(), {
        "source": "kaggle:artemminiailo/medicalconversations2disease",
        "language": "en",
        "origin": "LLM-generated (ChatGPT) chats",
        "license": "Apache 2.0",
        "icd10": "mapped: disease label -> ICD-10-CM via symptom2disease.LABEL_TO_ICD10 (coarse, one code per label)",
        "pii_gold": {"labelled": None, "notes": "nearly PII-free; travel destinations (Southeast Asia, India, Africa) are malaria/dengue/typhoid evidence; one child age"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
