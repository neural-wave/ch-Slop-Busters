"""priyamchoksi/pmc-patients-dataset-for-clinical-decision-support: 167,034 patient summaries from PubMed Central.

Each summary becomes a `note`; by default a seeded sample of 2,000 (`--limit 0` for all, ~520 MB). The structured
age (e.g. `[[60.0, 'year']]`) is used to label the age expressions in the text (`60-year-old`, `aged 60`,
`60 years`, `sixty-year-old`) as AGE: the only PII gold here. Dates, places and relatives' ages are unlabelled. No ICD-10.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, encounter, parser, read_csv, sample, write_dataset  # noqa: E402

UNITS = {"year": r"years?|yrs?|y/o|yo|y\.o\.", "month": r"months?|mos?", "week": r"weeks?|wks?", "day": r"days?", "hour": r"hours?|hrs?"}
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number(n: int) -> str:
    """Regex for `n` as digits or English words (`54`, `fifty-four`, `fifty four`), up to 119."""
    if n < 20:
        words = ONES[n]
    elif n < 100:
        words = TENS[n // 10] + (rf"[\s-]{ONES[n % 10]}" if n % 10 else "")
    elif n < 120:
        words = rf"(?:one\s)?hundred(?:\s(?:and\s)?{number(n - 100)})?" if n > 100 else r"(?:one\s)?hundred"
    else:
        return str(n)
    return rf"(?:{n}|{words})"


def age_patterns(age: str) -> list[re.Pattern]:
    pats = []
    for value, unit in ast.literal_eval(age or "[]"):
        num = number(int(value)) if float(value).is_integer() else re.escape(str(value))
        units = UNITS.get(unit, re.escape(unit))
        pats.append(re.compile(rf"(?<![\w.]){num}[\s-]*(?:{units})(?:[\s-]+old)?(?!\w)", re.I))
        if unit == "year":
            pats.append(re.compile(rf"\baged?\s+{num}(?![\w.])", re.I))
    return pats


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "priyamchoksi__pmc-patients-dataset-for-clinical-decision-support" / "PMC-Patients.csv", limit=2000)
    a = p.parse_args(argv)
    rows = sample(read_csv(a.src), a.limit, a.seed)

    def items():
        for r in rows:
            enc = encounter(f"pmc-{r['patient_uid']}", "pmc_patients", note=clean(r["patient"]))
            lab = Labels(enc)
            for rx in age_patterns(r["age"]):
                for m in rx.finditer(enc["note"]):
                    lab.add_pii("note", m.start(), m.end(), "AGE", "age (structured)", trim=False)
            meta = {"pmid": r["PMID"], "patient_uid": r["patient_uid"], "title": r["title"], "age": ast.literal_eval(r["age"] or "[]"), "gender": r["gender"]}
            yield enc, lab.record("none", pii_gold="partial", meta=meta)

    write_dataset("pmc_patients", items(), {
        "source": "kaggle:priyamchoksi/pmc-patients-dataset-for-clinical-decision-support",
        "language": "en",
        "origin": "real published case reports (PubMed Central)",
        "license": "CC BY-NC-SA 4.0 (non-commercial)",
        "icd10": "none",
        "pii_gold": {"labelled": "AGE of the patient, located from the structured age field",
                     "notes": f"sample of {len(rows)} (seed {a.seed}); dates, places, relatives' ages unlabelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
