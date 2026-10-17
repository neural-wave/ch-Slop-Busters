"""Robustness to unseen formats (bonus task e): build three sets with exact gold, then evaluate.

R1 `discharge`  Kaggle devp1866/noisy-medical-document-images-ocr ground truth (500 synthetic Indian
                hospital discharge summaries, CC BY-SA): rendered as text in three layouts (structured
                form, narrative letter, lowercase dictation). Gold = the labelled personal fields plus the
                synthetic admission/discharge dates, physician and phone added during rendering.
R2 `medconv`    Kaggle thedevastator/medical-conversation-corpus-100k (web medical Q&A, cased and
                punctuated): 80 conversations re-formatted as `[patient]`/`[doctor]` dialogue with injected
                PII (names, ages, DOB, cities, phones, e-mails, employers, hospitals, dates). No codes.
R3 `synthetic`  24 hospital documents (ER triage, discharge letter, referral e-mail, nursing handoff,
                radiology/pathology report, phone triage) written at development time with gold labels
                (`data/robustness/synthetic.jsonl`, `labels/upstream/robust_synthetic/`).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import date, timedelta

from .. import lexicons as L
from .. import paths

OUT = paths.robustness_dir()
GOLD = paths.labels_dir() / "upstream"
EXTERNAL = paths.data_dir() / "external"  # raw Kaggle CSVs, needed only for --build

IN_FIRST = "Anand Priya Rahul Kavya Suresh Lakshmi Arjun Meera Vikram Divya Karthik Ananya Rohan Sneha Arvind Pooja Manoj Deepa Sanjay Nisha".split()
IN_LAST = "Krishnan Sharma Iyer Nair Reddy Menon Gupta Rao Pillai Bose Kapoor Mehta Joshi Verma Chatterjee".split()
MONTHS = "January February March April May June July August September October November December".split()


def _dob_formats(d: date) -> list[str]:
    return [
        d.isoformat(),
        d.strftime("%d/%m/%Y"),
        f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    ]


def _item(field, text, cat, risk="high"):
    return {"field": field, "text": text, "category": cat, "risk": risk}


def build_discharge(n: int | None = None) -> tuple[list[dict], dict[str, dict]]:
    src = EXTERNAL / "kaggle_discharge/discharge_summaries_ground_truth.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))[: n or None]
    icd = L.icd10cm()
    recs, golds = [], {}
    for i, row in enumerate(rows):
        j = json.loads(row["json_data"])
        rnd = random.Random(i)
        eid = f"DS{200001 + i}"
        hosp, pat = j["hospital"], j["patient"]
        name = pat["name"].strip()
        first, last = name.split()[0], name.split()[-1]
        dob = date.fromisoformat(pat["dob"])
        adm = date(2024, 1, 1) + timedelta(days=rnd.randrange(0, 600))
        dis = adm + timedelta(days=rnd.randrange(1, 9))
        age = adm.year - dob.year - ((adm.month, adm.day) < (dob.month, dob.day))
        doc_f, doc_l = rnd.choice(IN_FIRST), rnd.choice(IN_LAST)
        phone = f"+91 9{rnd.randrange(10**8, 10**9)}"
        dob_s = _dob_formats(dob)[i % 4]
        adm_s, dis_s = adm.strftime("%d/%m/%Y"), dis.strftime("%d/%m/%Y")
        addr = hosp["address"]
        procs = "\n".join(f"- {p['desc']} (CPT {p['cpt']})" for p in j.get("procedures", []))
        meds = "\n".join(f"- {m}" for m in j.get("medications", []))
        layout = i % 3
        if layout == 0:
            note = (f"DISCHARGE SUMMARY\n{hosp['name']}\n{addr}\nNPI: {hosp['npi']}\n\nPatient Name: {name}\nDOB: {dob_s}    MRN: {pat['mrn']}\n"
                    f"Age: {age}\nDate of Admission: {adm_s}    Date of Discharge: {dis_s}\nAttending Physician: Dr. {doc_f} {doc_l}\n"
                    f"Insurance: {j.get('insurance', '')}\n\nDIAGNOSIS\n{j['diagnosis']}\n\nHOSPITAL COURSE\n{j['hospital_course']}\n\n"
                    f"PROCEDURES\n{procs}\n\nDISCHARGE MEDICATIONS\n{meds}\n\nFOLLOW-UP\nFollow up with Dr. {doc_l} in 2 weeks. Contact: {phone}.\n")  # fmt: skip
        elif layout == 1:
            note = (f"{hosp['name']}, {addr}\n\nRe: {name}, MRN {pat['mrn']}, born {dob_s}\n\nDear Colleague,\n\n{first} {last}, a {age}-year-old patient, "
                    f"was admitted to our hospital on {adm_s} with {j['diagnosis'].lower()}. {j['hospital_course']} {first} was discharged home on {dis_s} "
                    f"in a stable condition on the following medications:\n{meds}\n\nProcedures performed:\n{procs}\n\nThe patient is covered by "
                    f"{j.get('insurance', 'self-pay')}. Please contact me on {phone} with any questions.\n\nYours sincerely,\nDr. {doc_f} {doc_l}\n"
                    f"Department of Medicine, {hosp['name']}\n")  # fmt: skip
        else:
            spoken_mrn = " ".join(re.sub(r"\D", "", pat["mrn"]))
            note = (f"discharge summary dictated by doctor {doc_f.lower()} {doc_l.lower()} . patient is {first.lower()} {last.lower()} , date of birth {dob_s.lower()} , "
                    f"medical record number {spoken_mrn} . {first.lower()} is a {age} year old admitted on {adm_s} . diagnosis {j['diagnosis'].lower()} . "
                    f"{j['hospital_course'].lower()} medications {' , '.join(m.lower() for m in j.get('medications', []))} . discharged on {dis_s} . "
                    f"call {phone} for follow up at {hosp['name'].lower()} .\n")  # fmt: skip
        code = j.get("icd_10", "").strip()
        desc = icd.get(code) or icd.get(code.rstrip("0").rstrip(".")) or j["diagnosis"]
        rec = {
            "encounter_id": eid,
            "dataset": "discharge_summary",
            "dialogue": "",
            "note": note,
            "icd10_codes": [{"code": code, "description": desc}],
        }
        low = note.lower()
        pii = [_item("note", first, "PATIENT_NAME"), _item("note", last, "PATIENT_NAME"), _item("note", dob_s, "DOB"),
               _item("note", adm_s, "DATE"), _item("note", dis_s, "DATE"), _item("note", doc_f, "PERSON_NAME"), _item("note", doc_l, "PERSON_NAME"),
               _item("note", phone, "ID"), _item("note", hosp["name"], "LOCATION", "medium")]  # fmt: skip
        if layout == 2:
            pii.append(_item("note", spoken_mrn, "ID"))
            pii.append(_item("note", f"{age} year old", "AGE"))
        else:
            pii.append(_item("note", pat["mrn"], "ID"))
            pii.append(_item("note", f"{age}-year-old" if layout == 1 else f"Age: {age}", "AGE"))
            if j.get("insurance"):
                pii.append(_item("note", j["insurance"], "ORGANIZATION", "medium"))
        if layout == 0:
            pii.append(_item("note", hosp["npi"], "ID"))
        if layout != 2:
            for part in [p.strip() for p in addr.split(",")]:
                m = re.match(r"(.*?)\s*(\d{6})$", part)
                for piece in [m.group(1), m.group(2)] if m else [part]:
                    if piece and piece.lower() in low:
                        pii.append(_item("note", piece, "LOCATION", "medium"))
        seen, items = set(), []
        for it in pii:
            if it["text"] and it["text"].lower() in low and it["text"].lower() not in seen:
                seen.add(it["text"].lower())
                items.append(it)
        golds[eid] = {
            "encounter_id": eid,
            "pii": items,
            "keep": [{"field": "note", "text": j["diagnosis"], "category": "CODE_EVIDENCE", "why": code}]
            if j["diagnosis"].lower() in low
            else [],
        }
        recs.append(rec)
    return recs, golds


INJECT_FIRST = "Grace Will Hunter Faith Rose Mark Taylor Jordan Madison Brooke Dale Hope Chase Max Daisy Jack Lily Ryan Sophia Olivia Liam Noah Emma Ava".split()
INJECT_LAST = "Hill Price Walker Young White Cook Long Bell Parker Brooks Rivera Nguyen Patel Kowalski Okafor Schneider Fischer Moreau Rossi".split()
CITIES = [
    ("Denver", "Colorado"),
    ("Austin", "Texas"),
    ("Portland", "Oregon"),
    ("Columbus", "Ohio"),
    ("Tampa", "Florida"),
    ("Madison", "Wisconsin"),
    ("Phoenix", "Arizona"),
    ("Raleigh", "North Carolina"),
]
EMPLOYERS = ["Walmart", "Amazon", "FedEx", "Home Depot", "Kroger", "Boeing", "Starbucks", "Costco"]
JOBS = ["teacher", "truck driver", "nurse", "electrician", "accountant", "cashier", "software engineer", "mechanic"]
HOSPITALS = ["St. Mary's Hospital", "Mercy General Hospital", "Riverside Medical Center", "Lakeside Clinic"]


def build_medconv(n: int = 80, seed: int = 7) -> tuple[list[dict], dict[str, dict]]:
    csv.field_size_limit(10**9)
    rows = list(csv.DictReader((EXTERNAL / "kaggle_medconv/test.csv").open(encoding="utf-8")))
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    recs, golds = [], {}
    for i, row in enumerate(rows[: n * 2]):
        conv = row["Conversation"]
        turns = re.findall(r"\[\|(Human|AI)\|\]\s*(.*?)(?=\[\|(?:Human|AI)\|\]|$)", conv, re.S)
        turns = [(sp, re.sub(r"\s+", " ", t).strip()) for sp, t in turns if t.strip()]
        if len(turns) < 2 or len(recs) >= n:
            continue
        f, last = rnd.choice(INJECT_FIRST), rnd.choice(INJECT_LAST)
        age = rnd.randrange(19, 95)
        city, state = rnd.choice(CITIES)
        dob = date(2025 - age, rnd.randrange(1, 13), rnd.randrange(1, 28))
        dob_s = _dob_formats(dob)[i % 4]
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
                t = (intro + " " + t + (extra if k == 0 else "")) if k == 0 else t
                lines.append(f"[patient] {t}")
            else:
                t = t + (f" Regards, Dr. {df} {dl}." if k == 1 else "")
                lines.append(f"[doctor] {t}")
        dialogue = "\n".join(lines)
        eid = f"MC{i:04d}"
        low = dialogue.lower()
        cand = [(f, "PATIENT_NAME"), (last, "PATIENT_NAME"), (f"{age} years old", "AGE"), (f"{age} y/o", "AGE"), (f"{age}-year-old", "AGE"),
                (dob_s, "DOB"), (city, "LOCATION"), (state, "LOCATION"), (phone, "ID"), (email, "ID"), (emp, "ORGANIZATION"),
                (job, "OCCUPATION"), (hosp, "LOCATION"), (visit_s, "DATE"), (df, "PERSON_NAME"), (dl, "PERSON_NAME")]  # fmt: skip
        items, seen = [], set()
        for text, cat in cand:
            pat = re.compile(r"(?<!\w)" + re.escape(text.lower()) + r"(?!\w)")
            if pat.search(low):
                if cat in ("PATIENT_NAME", "PERSON_NAME") and L.is_common_word(text.lower()):
                    # collision words (`will`, `price`): annotate the full name so the item stays unambiguous
                    full = f"{f} {last}" if text in (f, last) else f"{df} {dl}"
                    m = re.search(re.escape(full.lower()), low) or re.search(
                        rf"(?:name is|i'm|this is|dr\.) {re.escape(text.lower())}", low
                    )
                    if not m:
                        continue
                    text = dialogue[m.start() : m.end()]
                if text.lower() in seen:
                    continue
                seen.add(text.lower())
                risk = "medium" if cat in ("LOCATION", "ORGANIZATION", "OCCUPATION") else "high"
                items.append(_item("dialogue", text, cat, risk))
        recs.append(
            {"encounter_id": eid, "dataset": "medconv_injected", "dialogue": dialogue, "note": "", "icd10_codes": []}
        )
        golds[eid] = {"encounter_id": eid, "pii": items, "keep": []}
    return recs, golds


def _write(name: str, recs: list[dict], golds: dict[str, dict]) -> None:
    from ..io import write_jsonl

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / f"{name}.jsonl", recs)
    gd = GOLD / f"robust_{name}"
    gd.mkdir(parents=True, exist_ok=True)
    for old in gd.glob("*.json"):
        old.unlink()
    for eid, g in golds.items():
        (gd / f"{eid}.json").write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        f"built {name}: {len(recs)} records, {sum(len(g['pii']) for g in golds.values())} gold PII items -> {OUT / (name + '.jsonl')}"
    )


def build() -> None:
    _write("discharge", *build_discharge())
    _write("medconv", *build_medconv())


def main(a: argparse.Namespace) -> int:
    from ..cli import run_one
    from .report import table

    if a.build or not (OUT / "discharge.jsonl").exists():
        build()
    rows = []
    for name in ("discharge", "medconv", "synthetic"):
        path = OUT / f"{name}.jsonl"
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        for no_codes in (False, True):
            if name == "medconv" and not no_codes:
                continue  # no codes in this set
            tag = f"robust_{name}_{a.preset}{'_nocodes' if no_codes else ''}{'_' + a.detectors.replace(',', '+') if a.detectors else ''}"
            rows.append(
                run_one(
                    str(path),
                    tag,
                    a.preset,
                    no_codes,
                    a.detectors,
                    a.icd if name != "medconv" else "none",
                    gold=str(GOLD / f"robust_{name}"),
                )
            )
    md = table(rows)
    out = paths.eval_dir() / f"robustness_{a.preset}{'_' + a.detectors.replace(',', '+') if a.detectors else ''}.md"
    out.write_text("# Robustness to unseen formats\n\n" + md + "\n", encoding="utf-8")
    print(md)
    return 0
