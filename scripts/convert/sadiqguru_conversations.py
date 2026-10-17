"""sadiqguru/medical-record-sample-data: 4 long synthetic doctor-patient conversations (asthma, BP, cardiology, diabetes).

`Doctor:` / `Patient:` paragraphs become `[doctor]` / `[patient]` turns. The PII gold below was curated by hand
for these 4 files (names, years, ages: every mention). Occupations and lifestyle details are not labelled; the
veterinary-clinic job in the asthma visit is clinically relevant (occupational exposure). No ICD-10 codes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, dialogue_text, encounter, parser, sample, write_dataset  # noqa: E402

# file stem -> (regex, category); the span is group 1 when the regex has one
GOLD = {
    "emily_rodriguez_asthma_checkup": [(r"\bEmily\b", "PATIENT_NAME"), (r"\bRodriguez\b", "PATIENT_NAME"), (r"\bSmith\b", "PERSON_NAME")],
    "james_wilson_cardiology_referral": [(r"\bJames\b", "PATIENT_NAME"), (r"\bWilson\b", "PATIENT_NAME"), (r"\bSmith\b", "PERSON_NAME"),
                                         (r"\b2021\b", "YEAR"), (r"\bI was (56)\b", "AGE")],
    "michael_chen_diabetes_followup": [(r"\bMichael\b", "PATIENT_NAME"), (r"\bChen\b", "PATIENT_NAME")],
    "sarah_johnson_bp_consultation": [(r"\bSarah\b", "PATIENT_NAME"), (r"\bJohnson\b", "PATIENT_NAME"), (r"\bSmith\b", "PERSON_NAME"),
                                      (r"\b2019\b", "YEAR")],
}  # fmt: skip


def read_turns(path: Path) -> list[tuple[str, str]]:
    turns: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(Doctor|Patient):\s*(.*)", line)
        if m:
            turns.append([m.group(1).lower(), m.group(2)])
        elif line.strip() and turns:
            turns[-1][1] += " " + line.strip()
    return [(s, t) for s, t in turns]


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "sadiqguru__medical-record-sample-data")
    a = p.parse_args(argv)

    def items():
        for path in sample(sorted(a.src.glob("*.txt")), a.limit, a.seed):
            enc = encounter(f"sadiq-{path.stem}", "sadiqguru_conversations", dialogue=clean(dialogue_text(read_turns(path))))
            lab = Labels(enc)
            for rx, cat in GOLD.get(path.stem, []):
                for m in re.finditer(rx, enc["dialogue"]):
                    g = 1 if m.re.groups else 0
                    lab.add_pii("dialogue", m.start(g), m.end(g), cat, "manual", trim=False)
            yield enc, lab.record("none", pii_gold="partial" if path.stem in GOLD else "none", meta={"source_file": path.name, "audio": f"{path.stem}.mp3"})

    write_dataset("sadiqguru_conversations", items(), {
        "source": "kaggle:sadiqguru/medical-record-sample-data",
        "language": "en",
        "origin": "synthetic conversations (with MP3 audio, not used)",
        "license": "MIT",
        "icd10": "none (topics: asthma, hypertension, coronary artery disease, type 2 diabetes)",
        "pii_gold": {"labelled": "PATIENT_NAME, PERSON_NAME, YEAR, AGE (hand-curated)",
                     "notes": "occupations/lifestyle details not labelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
