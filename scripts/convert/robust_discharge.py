"""Robustness set R1: the devp discharge summaries in the layout `devp_discharge` does not render, with synthetic PII.

A port of the A+D robustness builder (models/ad-night/repo/src/neural_wave/eval/robustness.py, `build_discharge`). Each
of the 500 records of `Data/discharge_summaries_ground_truth.csv` is rendered in layout `i % 3` of three: a structured
form, a narrative letter and a lowercase dictation. Every record gets a synthetic admission and discharge date, the
age at admission, an attending physician and a phone number (seeded by the row index), so it carries dates, ages,
clinician names and phone numbers that the source JSON lacks. The gold is the A+D gold: every rendered personal value,
as spans of each whole-word occurrence (longest item first; an occurrence inside a longer item is not a span of its own).

`devp_discharge` already renders the form and the letter from the same records, so by default only the dictation
layout is kept (166 records, `DS200003`, `DS200006`, ...): lowercase, no punctuation cues, the MRN spoken digit by
digit. `--layouts form letter dictation` rebuilds A+D's full set (500 records, identical to their `discharge.jsonl`).
The synthetic values are injected, so their spans carry `label: "injected:<kind>"`.
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, dotted, encounter, icd_code, parser, read_csv, sample, write_dataset  # noqa: E402

LAYOUTS = ("form", "letter", "dictation")
IN_FIRST = "Anand Priya Rahul Kavya Suresh Lakshmi Arjun Meera Vikram Divya Karthik Ananya Rohan Sneha Arvind Pooja Manoj Deepa Sanjay Nisha".split()
IN_LAST = "Krishnan Sharma Iyer Nair Reddy Menon Gupta Rao Pillai Bose Kapoor Mehta Joshi Verma Chatterjee".split()
MONTHS = "January February March April May June July August September October November December".split()


def dob_formats(d: date) -> list[str]:
    return [d.isoformat(), d.strftime("%d/%m/%Y"), f"{d.day} {MONTHS[d.month - 1]} {d.year}", f"{MONTHS[d.month - 1]} {d.day}, {d.year}"]


def render(i: int, row: dict) -> tuple[dict, list[tuple[str, str, str, str]], list[tuple[str, str]], dict]:
    """Record `i` of the CSV as A+D render it: (encounter, gold PII (text, category, risk, label), keep (text, why), meta)."""
    j = json.loads(row["json_data"])
    rnd = random.Random(i)
    hosp, pat = j["hospital"], j["patient"]
    name = pat["name"].strip()
    first, last = name.split()[0], name.split()[-1]
    dob = date.fromisoformat(pat["dob"])
    adm = date(2024, 1, 1) + timedelta(days=rnd.randrange(0, 600))
    dis = adm + timedelta(days=rnd.randrange(1, 9))
    age = adm.year - dob.year - ((adm.month, adm.day) < (dob.month, dob.day))
    doc_f, doc_l = rnd.choice(IN_FIRST), rnd.choice(IN_LAST)
    phone = f"+91 9{rnd.randrange(10**8, 10**9)}"
    dob_s = dob_formats(dob)[i % 4]
    adm_s, dis_s = adm.strftime("%d/%m/%Y"), dis.strftime("%d/%m/%Y")
    addr = hosp["address"]
    procs = "\n".join(f"- {p['desc']} (CPT {p['cpt']})" for p in j.get("procedures", []))
    meds = "\n".join(f"- {m}" for m in j.get("medications", []))
    layout = LAYOUTS[i % 3]
    spoken_mrn = " ".join(re.sub(r"\D", "", pat["mrn"]))
    if layout == "form":
        note = (f"DISCHARGE SUMMARY\n{hosp['name']}\n{addr}\nNPI: {hosp['npi']}\n\nPatient Name: {name}\nDOB: {dob_s}    MRN: {pat['mrn']}\n"
                f"Age: {age}\nDate of Admission: {adm_s}    Date of Discharge: {dis_s}\nAttending Physician: Dr. {doc_f} {doc_l}\n"
                f"Insurance: {j.get('insurance', '')}\n\nDIAGNOSIS\n{j['diagnosis']}\n\nHOSPITAL COURSE\n{j['hospital_course']}\n\n"
                f"PROCEDURES\n{procs}\n\nDISCHARGE MEDICATIONS\n{meds}\n\nFOLLOW-UP\nFollow up with Dr. {doc_l} in 2 weeks. Contact: {phone}.\n")  # fmt: skip
    elif layout == "letter":
        note = (f"{hosp['name']}, {addr}\n\nRe: {name}, MRN {pat['mrn']}, born {dob_s}\n\nDear Colleague,\n\n{first} {last}, a {age}-year-old patient, "
                f"was admitted to our hospital on {adm_s} with {j['diagnosis'].lower()}. {j['hospital_course']} {first} was discharged home on {dis_s} "
                f"in a stable condition on the following medications:\n{meds}\n\nProcedures performed:\n{procs}\n\nThe patient is covered by "
                f"{j.get('insurance', 'self-pay')}. Please contact me on {phone} with any questions.\n\nYours sincerely,\nDr. {doc_f} {doc_l}\n"
                f"Department of Medicine, {hosp['name']}\n")  # fmt: skip
    else:
        note = (f"discharge summary dictated by doctor {doc_f.lower()} {doc_l.lower()} . patient is {first.lower()} {last.lower()} , date of birth {dob_s.lower()} , "
                f"medical record number {spoken_mrn} . {first.lower()} is a {age} year old admitted on {adm_s} . diagnosis {j['diagnosis'].lower()} . "
                f"{j['hospital_course'].lower()} medications {' , '.join(m.lower() for m in j.get('medications', []))} . discharged on {dis_s} . "
                f"call {phone} for follow up at {hosp['name'].lower()} .\n")  # fmt: skip
    code = j.get("icd_10", "").strip()
    enc = encounter(f"DS{200001 + i}", "discharge_summary", note=note, codes=[icd_code(code)])
    low = note.lower()
    pii = [(first, "PATIENT_NAME", "high", "patient.name"), (last, "PATIENT_NAME", "high", "patient.name"), (dob_s, "DOB", "high", "patient.dob"),
           (adm_s, "DATE", "high", "injected:admission_date"), (dis_s, "DATE", "high", "injected:discharge_date"),
           (doc_f, "PERSON_NAME", "high", "injected:physician"), (doc_l, "PERSON_NAME", "high", "injected:physician"),
           (phone, "ID", "high", "injected:phone"), (hosp["name"], "LOCATION", "medium", "hospital.name")]  # fmt: skip
    if layout == "dictation":
        pii += [(spoken_mrn, "ID", "high", "patient.mrn"), (f"{age} year old", "AGE", "high", "injected:age")]
    else:
        pii += [(pat["mrn"], "ID", "high", "patient.mrn"), (f"{age}-year-old" if layout == "letter" else f"Age: {age}", "AGE", "high", "injected:age")]
        if j.get("insurance"):
            pii.append((j["insurance"], "ORGANIZATION", "medium", "insurance"))
    if layout == "form":
        pii.append((hosp["npi"], "ID", "high", "hospital.npi"))
    if layout != "dictation":
        for part in [p.strip() for p in addr.split(",")]:
            m = re.match(r"(.*?)\s*(\d{6})$", part)
            for piece in [m.group(1), m.group(2)] if m else [part]:
                if piece and piece.lower() in low:
                    pii.append((piece, "LOCATION", "medium", "hospital.address"))
    seen, items = set(), []
    for it in pii:
        if it[0] and it[0].lower() in low and it[0].lower() not in seen:
            seen.add(it[0].lower())
            items.append(it)
    keep = [(j["diagnosis"], dotted(code))] if j["diagnosis"].lower() in low else []
    meta = {"source_file": row["filename"], "source_row": i, "layout": layout, "patient": pat["mrn"]}
    return enc, items, keep, meta


def labels(enc: dict, pii: list[tuple[str, str, str, str]], keep: list[tuple[str, str]], field: str, pii_gold: str, icd10_source: str,
           meta: dict) -> tuple[dict, int]:  # fmt: skip
    """Text gold -> spans: every whole-word occurrence of each item, longest item first (an occurrence inside a longer
    item stays part of it). Returns the labels line and the number of items that got no span."""
    lab, lost = Labels(enc), 0
    for text, cat, risk, label in sorted(pii, key=lambda it: -len(it[0])):
        lost += not lab.match_pii(text, cat, label, fields=[field], risk=risk)
    for text, why in keep:
        lab.match_keep(text, "CODE_EVIDENCE", why, fields=[field])
    return lab.record(icd10_source, pii_gold=pii_gold, meta=meta), lost


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "devp1866__noisy-medical-document-images-ocr" / "Data")
    p.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=["dictation"], help="layouts to keep (default: dictation, the one devp_discharge lacks)")
    a = p.parse_args(argv)
    rows = read_csv(a.src / "discharge_summaries_ground_truth.csv")
    kept = [i for i in range(len(rows)) if LAYOUTS[i % 3] in a.layouts]
    lost = 0

    def items():
        nonlocal lost
        for i in sample(kept, a.limit, a.seed):
            enc, pii, keep, meta = render(i, rows[i])
            lab, n = labels(enc, pii, keep, "note", "exhaustive", "gold", meta)
            lost += n
            yield enc, lab

    recs = list(items())
    write_dataset("robust_discharge", recs, {
        "source": "kaggle:devp1866/noisy-medical-document-images-ocr",
        "stage": "inject",
        "parent": "devp_discharge",
        "language": "en",
        "origin": "synthetic (templated JSON ground truth) rendered by the A+D robustness builder (R1), with injected admission and discharge "
                  "dates, age, physician and phone",
        "license": "CC BY-SA 4.0",
        "icd10": "gold: record icd_10 field, official ICD-10-CM 2023 description",
        "pii_gold": {"labelled": "every rendered personal value (A+D gold)", "layouts": sorted(a.layouts, key=LAYOUTS.index),
                     "notes": f"{lost} gold strings gave no span (only inside a longer item)"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
