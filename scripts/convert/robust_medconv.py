"""Robustness set R2: 80 web medical conversations (cased, punctuated) as `[patient]`/`[doctor]` dialogue with injected PII.

A port of the A+D robustness builder (models/ad-night/repo/src/neural_wave/eval/robustness.py, `build_medconv`):
the test split of thedevastator/medical-conversation-corpus-100k is shuffled (seed 7), and each conversation with at
least two turns gets a synthetic patient (name, age, DOB, city and state, phone, e-mail, employer, job, hospital, visit
date) introduced in its first turn and a doctor's signature in the second; at most six turns are kept. The gold is the
injected values found in the dialogue. A name that is also an ordinary word (`Will`, `Price`) is labelled only as the
full name or after `my name is`/`I'm`/`this is`/`Dr.`, so the gold string stays unambiguous.

The posts are real and carry their own names, ages and places, which are not labelled: `pii_gold` is `partial` (leak
recall is meaningful, over-removal is not). No ICD-10 codes. Injected spans carry `label: "injected:<kind>"`.
"""

from __future__ import annotations

import random
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, encounter, parser, read_csv, write_dataset  # noqa: E402
from robust_discharge import IN_FIRST, IN_LAST, MONTHS, dob_formats, labels  # noqa: E402

INJECT_FIRST = "Grace Will Hunter Faith Rose Mark Taylor Jordan Madison Brooke Dale Hope Chase Max Daisy Jack Lily Ryan Sophia Olivia Liam Noah Emma Ava".split()
INJECT_LAST = "Hill Price Walker Young White Cook Long Bell Parker Brooks Rivera Nguyen Patel Kowalski Okafor Schneider Fischer Moreau Rossi".split()
CITIES = [("Denver", "Colorado"), ("Austin", "Texas"), ("Portland", "Oregon"), ("Columbus", "Ohio"), ("Tampa", "Florida"), ("Madison", "Wisconsin"),
          ("Phoenix", "Arizona"), ("Raleigh", "North Carolina")]  # fmt: skip
EMPLOYERS = ["Walmart", "Amazon", "FedEx", "Home Depot", "Kroger", "Boeing", "Starbucks", "Costco"]
JOBS = ["teacher", "truck driver", "nurse", "electrician", "accountant", "cashier", "software engineer", "mechanic"]
HOSPITALS = ["St. Mary's Hospital", "Mercy General Hospital", "Riverside Medical Center", "Lakeside Clinic"]
# The injectable names that A+D's lexicon calls ordinary words (`is_common_word`: frequent lowercase use), fixed here
# so the build needs no lexicon and stays identical to theirs.
COMMON_NAMES = frozenset("bell chase cook faith grace hill hope hunter long mark max price rose taylor walker white will young".split())
TURN = re.compile(r"\[\|(Human|AI)\|\]\s*(.*?)(?=\[\|(?:Human|AI)\|\]|$)", re.S)
KINDS = {"PATIENT_NAME": "patient_name", "AGE": "age", "DOB": "dob", "ID": "contact", "ORGANIZATION": "employer", "OCCUPATION": "job",
         "DATE": "visit_date", "PERSON_NAME": "physician"}  # fmt: skip


def build(rows: list[dict], n: int = 80, seed: int = 7) -> list[tuple[dict, list[tuple[str, str, str, str]], dict]]:
    """A+D's R2 records: (encounter, gold PII (text, category, risk, label), meta)."""
    order = list(enumerate(rows))
    rnd = random.Random(seed)
    rnd.shuffle(order)
    out = []
    for i, (row_no, row) in enumerate(order[: n * 2]):
        turns = [(sp, re.sub(r"\s+", " ", t).strip()) for sp, t in TURN.findall(row["Conversation"]) if t.strip()]
        if len(turns) < 2 or len(out) >= n:
            continue
        f, last = rnd.choice(INJECT_FIRST), rnd.choice(INJECT_LAST)
        age = rnd.randrange(19, 95)
        city, state = rnd.choice(CITIES)
        dob = date(2025 - age, rnd.randrange(1, 13), rnd.randrange(1, 28))
        dob_s = dob_formats(dob)[i % 4]
        phone = f"({rnd.randrange(200, 999)}) {rnd.randrange(200, 999)}-{rnd.randrange(1000, 9999)}"
        email = f"{f.lower()}.{last.lower()}{rnd.randrange(10, 99)}@gmail.com"
        emp, job, hosp = rnd.choice(EMPLOYERS), rnd.choice(JOBS), rnd.choice(HOSPITALS)
        visit = date(2024, rnd.randrange(1, 13), rnd.randrange(1, 28))
        visit_s = f"{MONTHS[visit.month - 1]} {visit.day}, {visit.year}"
        df, dl = rnd.choice(IN_FIRST + INJECT_FIRST), rnd.choice(INJECT_LAST + IN_LAST)
        intro = rnd.choice([
            f"Hi doctor, my name is {f} {last} and I am {age} years old, living in {city}, {state}.",
            f"Hello, I'm {f} ({age} y/o, DOB {dob_s}) from {city}.",
            f"Hi, this is {f} {last}. I'm a {age}-year-old {job} working at {emp} in {city}, {state}.",
        ])  # fmt: skip
        extra = rnd.choice([
            f" I was seen at {hosp} on {visit_s}. You can reach me at {phone} or {email}.",
            f" I work as a {job} at {emp}. My phone number is {phone}.",
            f" My date of birth is {dob_s} and I was treated at {hosp} on {visit_s}.",
        ])  # fmt: skip
        lines = []
        for k, (sp, t) in enumerate(turns[:6]):
            if sp == "Human":
                lines.append(f"[patient] {intro + ' ' + t + extra if k == 0 else t}")
            else:
                lines.append(f"[doctor] {t + (f' Regards, Dr. {df} {dl}.' if k == 1 else '')}")
        dialogue = "\n".join(lines)
        low = dialogue.lower()
        cand = [(f, "PATIENT_NAME"), (last, "PATIENT_NAME"), (f"{age} years old", "AGE"), (f"{age} y/o", "AGE"), (f"{age}-year-old", "AGE"),
                (dob_s, "DOB"), (city, "LOCATION"), (state, "LOCATION"), (phone, "ID"), (email, "ID"), (emp, "ORGANIZATION"),
                (job, "OCCUPATION"), (hosp, "LOCATION"), (visit_s, "DATE"), (df, "PERSON_NAME"), (dl, "PERSON_NAME")]  # fmt: skip
        items, seen = [], set()
        for text, cat in cand:
            if not re.search(r"(?<!\w)" + re.escape(text.lower()) + r"(?!\w)", low):
                continue
            if cat in ("PATIENT_NAME", "PERSON_NAME") and text.lower() in COMMON_NAMES:
                full = f"{f} {last}" if text in (f, last) else f"{df} {dl}"
                m = re.search(re.escape(full.lower()), low) or re.search(rf"(?:name is|i'm|this is|dr\.) {re.escape(text.lower())}", low)
                if not m:
                    continue
                text = dialogue[m.start() : m.end()]
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            kind = "hospital" if text == hosp else "place" if cat == "LOCATION" else "email" if text == email else "phone" if text == phone else KINDS[cat]
            items.append((text, cat, "medium" if cat in ("LOCATION", "ORGANIZATION", "OCCUPATION") else "high", f"injected:{kind}"))
        out.append((encounter(f"MC{i:04d}", "medconv_injected", dialogue=dialogue), items, {"split": "test", "row": row_no}))
    return out


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "thedevastator__medical-conversation-corpus-100k")
    p.add_argument("--n", type=int, default=80, help="conversations (default: 80, as A+D)")
    a = p.parse_args(argv)
    built = build(read_csv(a.src / "test.csv"), a.n)
    lost = 0

    def items():
        nonlocal lost
        for enc, pii, meta in built:
            lab, k = labels(enc, pii, [], "dialogue", "partial", "none", meta)
            lost += k
            yield enc, lab

    recs = list(items())
    write_dataset("robust_medconv", recs, {
        "source": "kaggle:thedevastator/medical-conversation-corpus-100k (test split)",
        "stage": "inject",
        "parent": "medconv_corpus",
        "language": "en",
        "origin": "real user questions + doctor answers (HealthCareMagic via ChatDoctor), with synthetic patient PII injected by the A+D "
                  "robustness builder (R2)",
        "license": "CC0 (doubtful: scraped)",
        "icd10": "none",
        "pii_gold": {"labelled": "the injected values", "notes": f"real names/ages/places of the posts are unlabelled; {lost} gold strings gave no span"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
