"""devp1866/noisy-medical-document-images-ocr: 500 synthetic discharge summaries (Indian hospitals), gold ICD-10.

The dataset is images plus their JSON ground truth; we use only the JSON (`Data/*_ground_truth.csv`) and render
each record as a text note, in two layouts (structured form, referral letter) alternated per patient. Only fields
present in the record are rendered, so the PII gold is exact: patient name (and first name alone in the letter), DOB,
MRN, hospital name/address/NPI, insurer. The diagnosis text is labelled `keep` / CODE_EVIDENCE. One code per record,
only 5 distinct codes (J20.9, E78.5, E11.9, M17.9, I10). `--with-bills` adds the 500 bill records (same schema).

The 500 records combine at random 10 patients, 5 hospitals, 5 insurers and 5 diagnosis + hospital-course pairs, so
they repeat the same few PII values. By default we keep one record per (document type, patient, hospital): 50
discharge summaries, every patient at every hospital once. `--all` keeps all of them. `meta.patient` (the MRN) is
the key to split on.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, TextBuilder, dotted, encounter, icd_code, parser, read_csv, sample, write_dataset  # noqa: E402


def dedup(rows: list[dict]) -> list[dict]:
    """First record of each (document type, patient, hospital), in source order."""
    seen, out = set(), []
    for row in rows:
        j = json.loads(row["json_data"])
        key = (row["document_type"], j["patient"]["mrn"], j["hospital"]["name"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def render_form(j: dict) -> TextBuilder:
    h, pt = j["hospital"], j["patient"]
    b = TextBuilder().add("DISCHARGE SUMMARY\n")
    b.add(h["name"], "LOCATION", "hospital.name").add("\n").add(h["address"], "LOCATION", "hospital.address")
    b.add("\nNPI: ").add(h["npi"], "ID", "hospital.npi").add("\n\nPatient Name: ").add(pt["name"], "PATIENT_NAME", "patient.name")
    b.add("\nDOB: ").add(pt["dob"], "DOB", "patient.dob").add("\nMRN: ").add(pt["mrn"], "ID", "patient.mrn")
    if j.get("insurance"):
        b.add("\nInsurance: ").add(j["insurance"], "ORGANIZATION", "insurance")
    b.add("\n\nDIAGNOSIS\n").add(j["diagnosis"]).add("\n\nHOSPITAL COURSE\n").add(j["hospital_course"])
    if j.get("procedures"):
        b.add("\n\nPROCEDURES\n" + "\n".join(f"- {p['desc']} (CPT {p['cpt']})" for p in j["procedures"]))
    if j.get("medications"):
        b.add("\n\nDISCHARGE MEDICATIONS\n" + "\n".join(f"- {m}" for m in j["medications"]))
    return b


def render_letter(j: dict) -> TextBuilder:
    h, pt = j["hospital"], j["patient"]
    first = pt["name"].split()[0]
    b = TextBuilder().add(h["name"], "LOCATION", "hospital.name").add("\n").add(h["address"], "LOCATION", "hospital.address")
    b.add("\n\nRe: ").add(pt["name"], "PATIENT_NAME", "patient.name").add(" (MRN ").add(pt["mrn"], "ID", "patient.mrn")
    b.add(", DOB ").add(pt["dob"], "DOB", "patient.dob").add(")\n\nDear Colleague,\n\n")
    b.add(pt["name"], "PATIENT_NAME", "patient.name").add(" was admitted to ").add(h["name"], "LOCATION", "hospital.name")
    b.add(" with ").add(j["diagnosis"].lower()).add(". ").add(j["hospital_course"])
    if j.get("procedures"):
        b.add("\n\nProcedures performed: " + "; ".join(f"{p['desc']} (CPT {p['cpt']})" for p in j["procedures"]) + ".")
    if j.get("medications"):
        b.add("\nDischarge medications: " + "; ".join(j["medications"]) + ".")
    if j.get("insurance"):
        b.add("\n\n").add(first, "PATIENT_NAME", "patient.name").add(" is covered by ").add(j["insurance"], "ORGANIZATION", "insurance").add(".")
    b.add("\n\nYours sincerely,\nDepartment of Medicine, ").add(h["name"], "LOCATION", "hospital.name")
    return b


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "devp1866__noisy-medical-document-images-ocr" / "Data")
    p.add_argument("--with-bills", action="store_true", help="also convert medical_bills_ground_truth.csv")
    p.add_argument("--all", action="store_true", help="keep every record (default: one per document type, patient and hospital)")
    a = p.parse_args(argv)
    files = ["discharge_summaries_ground_truth.csv"] + (["medical_bills_ground_truth.csv"] if a.with_bills else [])
    source = [r for f in files for r in read_csv(a.src / f)]
    rows = source if a.all else dedup(source)

    def items():
        per_patient: Counter = Counter()
        for row in sample(rows, a.limit, a.seed):
            j = json.loads(row["json_data"])
            mrn = j["patient"]["mrn"]
            layout = ("form", "letter")[per_patient[mrn] % 2]
            per_patient[mrn] += 1
            b = render_form(j) if layout == "form" else render_letter(j)
            num = row["filename"].split("_")[-2]  # med_doc_discharge_summary_200001_noisy.jpg
            enc = encounter(f"devp-{num}", "devp_discharge", note=b.text(), codes=[icd_code(j["icd_10"])])
            lab = Labels(enc)
            for s, e, cat, label in b.spans:
                lab.add_pii("note", s, e, cat, label, trim=False)
            lab.match_keep(j["diagnosis"], "CODE_EVIDENCE", dotted(j["icd_10"]), fields=["note"])
            meta = {"source_file": row["filename"], "document_type": row["document_type"], "layout": layout, "patient": mrn}
            yield enc, lab.record("gold", pii_gold="exhaustive", meta=meta)

    js = [json.loads(r["json_data"]) for r in source]
    pool = {k: len({f(j) for j in js}) for k, f in (("patients", lambda j: j["patient"]["mrn"]), ("hospitals", lambda j: j["hospital"]["name"]),
                                                     ("insurers", lambda j: j.get("insurance")), ("clinical stories", lambda j: j["hospital_course"]))}  # fmt: skip
    write_dataset("devp_discharge", items(), {
        "source": "kaggle:devp1866/noisy-medical-document-images-ocr",
        "language": "en",
        "origin": "synthetic (templated JSON ground truth), rendered to text by this script",
        "license": "CC BY-SA 4.0",
        "icd10": "gold: record icd_10 field, official ICD-10-CM 2023 description",
        "pii_gold": {"labelled": "every personal field rendered into the note",
                     "notes": "no dates, ages or physician names exist in the source records"},
        "diversity": f"{len(source)} source records combine only {', '.join(f'{v} {k}' for k, v in pool.items())}; "
                     + ("all kept (--all)" if a.all else f"kept {len(rows)}, one per (document type, patient, hospital)"),
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
