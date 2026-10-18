"""Synthetic external-cause encounters (TRAINING_PLAN.md §3): templates written by a larger model, PII filled in here.

Train has no V00-Y99 codes but eval has 24, and their only evidence (activity, place, mechanism) looks like a
quasi-identifier. Each template in rl/synth/templates/<id>.json is an injury encounter in the style of data/, with
codes set by construction, labelled medical concepts and evidence, and slots where identifiers go. The G*.json
templates (rl/synth_gen.py, Claude Haiku) use the same format for any specialty (dataset "synthetic_<visit>"); only
"synthetic_injury" templates must carry an external-cause code:

  {"id": "S001", "dataset": "synthetic_injury", "sex": "female",
   "icd10_codes": [{"code": "S82.101A", "description": "<CMS title>"}, {"code": "W00.0XXA", ...}, ...],
   "dialogue": "[doctor] hi , {PATIENT_FIRST} . ...",          lowercase, spaced punctuation, [doctor]/[patient]
   "note": "CHIEF COMPLAINT\\n\\n...{PATIENT_FIRST} {PATIENT_LAST} is a {AGE}-year-old female...",
   "medical_concepts": [{"text", "field", "type", "icd_relevant"}],   verbatim in the slot-free template text
   "ambiguous_terms": [{"text", "field", "is_pii", "comment"}],
   "evidence": [{"code", "spans": [{"text", "field"}]}],
   "pii_fixed": [{"text", "field", "category", "comment"}]}          PII written literally (occupation, hobby, ...)

Slots: {PATIENT_FIRST} {PATIENT_LAST} {DOCTOR_FIRST} {DOCTOR_LAST} {OTHER_FIRST} {AGE} {DATE_SPOKEN} {DATE_NUM}
{YEAR} {CITY} {PLACE} {ORG} {PHONE} {MRN}. fill() draws one fake identity per (template, seed); in the dialogue the
values are lowercased. Gold PII is generated from the slots, including the multi-word forms that occur
("Mr. {PATIENT_LAST}", "{PATIENT_FIRST} {PATIENT_LAST}").

  python3 rl/synth.py check            validate every template (codes in the ICD-10-CM table, spans verbatim)
  python3 rl/synth.py build [--seeds 1] write data/synth_external_cause.jsonl and ../term-generalizer/gold/synth_gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
from common import ROOT, write_jsonl  # noqa: E402
from spec import TIER, is_valid_code  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent / "synth" / "templates"
OUT_DATA = ROOT / "data" / "synth_external_cause.jsonl"
OUT_GOLD = ROOT.parent / "term-generalizer" / "gold" / "synth_gold.jsonl"
FIELDS = ("dialogue", "note")
CONCEPT_TYPES = {"diagnosis", "symptom", "finding", "lab_or_test", "medication", "procedure", "anatomy"}

# Fill-in pools: diverse, unambiguous values (no eponyms, no common English words), so that the seeds of one template
# differ in every slot. A value that also occurs in the template's own text is redrawn (fill()).
FIRST = {"female": ["olivia", "grace", "hannah", "maria", "denise", "priya", "keisha", "lauren", "megan", "rosa",
                    "tamara", "yvonne", "chloe", "naomi", "brenda", "ingrid", "aisha", "beatriz", "carmen", "dana",
                    "eleni", "farah", "gwen", "hiroko", "imani", "jolene", "kateryna", "latoya", "mei", "nadine",
                    "oksana", "paulina", "rania", "svetlana", "tanisha", "ulrike", "valeria", "wendy", "ximena",
                    "yasmin", "zainab", "adaeze", "bridget", "colleen", "dolores", "esperanza", "fiona", "gabriela",
                    "harriet", "isabel", "jacqueline", "kimberly", "lucinda", "marisol", "nkechi", "ophelia",
                    "pilar", "renata", "siobhan", "thuy"],
         "male": ["marcus", "derek", "tomas", "ethan", "raymond", "kevin", "luis", "harold", "jamal", "victor",
                  "owen", "gordon", "felix", "darnell", "simon", "wesley", "abdul", "bogdan", "cedric", "dmitri",
                  "emeka", "fernando", "gustavo", "hamid", "ignacio", "jerome", "kwame", "lars", "malik", "nikhil",
                  "oleksandr", "pedro", "quincy", "rashad", "santiago", "terrence", "umar", "vikram", "wojciech",
                  "xavier", "yusuf", "zoltan", "anders", "bartholomew", "cormac", "desmond", "elias", "florian",
                  "giovanni", "hector", "isaiah", "jorge", "kenji", "leroy", "mateo", "nathaniel", "orlando",
                  "phillip", "reginald", "stanislav"]}
LAST = ["brennan", "okafor", "lindqvist", "castillo", "whitaker", "nakamura", "delgado", "hargrove", "petrov",
        "fitzgerald", "abernathy", "moreau", "sandoval", "kowalski", "ashby", "thibodeaux", "vance", "okonkwo",
        "abernethy", "bergstrom", "chaudhry", "dubois", "esposito", "fairweather", "gallagher", "haddad", "iwasaki",
        "jankowski", "kaminski", "lefebvre", "mbeki", "novak", "oyelaran", "pemberton", "quintana", "rasmussen",
        "szymanski", "takahashi", "underwood", "villanueva", "wojcik", "yamamoto", "zielinski", "achebe",
        "bianchi", "contreras", "dasgupta", "eriksen", "ferreira", "goldberg", "hutchinson", "ibrahim", "jovanovic",
        "kowalczyk", "lombardi", "mcallister", "nwosu", "obrien", "papadopoulos", "ramirez", "schaefer",
        "trujillo", "ueda", "vasquez", "weatherby", "xiong", "yilmaz", "zamora", "alvarado", "blackwood",
        "cavanaugh", "donnelly", "everett", "fontaine", "gutierrez", "holloway", "iverson", "jablonski",
        "kensington", "lockhart", "montgomery", "northcott", "ortega", "prescott", "radcliffe", "salazar",
        "thornbury", "valdivia", "whitmore", "yarborough", "zimmerman", "agarwal", "bjornsson", "cardenas",
        "dimitrov", "estrada", "fujimoto", "grabowski", "hernandez", "isaacson", "kovacs", "mendoza", "nguyen",
        "olawale", "pacheco", "rinaldi", "sorensen", "tran", "vukovic", "wakefield", "yoshida", "zhang"]
CITY = ["boise", "spokane", "duluth", "flagstaff", "asheville", "missoula", "provo", "bangor", "eugene", "tulsa",
        "albuquerque", "bozeman", "cheyenne", "des moines", "el paso", "fresno", "gainesville", "harrisburg",
        "knoxville", "lubbock", "macon", "naperville", "omaha", "pensacola", "reno", "savannah", "tacoma",
        "wichita", "akron", "billings", "chattanooga", "dayton", "erie", "fargo", "green bay", "huntsville",
        "laramie", "modesto", "norfolk", "olympia", "peoria", "rapid city", "sioux falls", "toledo", "yakima",
        "burlington", "columbia", "durango", "fort collins", "grand rapids", "kalamazoo", "lafayette",
        "midland", "ocala", "pueblo", "roanoke", "scranton", "topeka", "waco", "youngstown"]
PLACE = ["mount hood", "lake tahoe", "pine ridge trail", "crystal mountain", "silver lake", "cedar creek park",
         "eagle rock", "riverside park", "timberline lodge", "bear lake", "copper canyon", "dunmore pier",
         "elk meadow", "falcon ridge golf course", "glacier point", "harbor view marina", "iron horse trail",
         "juniper hills", "kettle falls", "lantern square", "maple hollow campground", "northgate mall",
         "oak bluff", "pelican bay", "quarry lake", "red butte garden", "sandy point beach", "thunder basin",
         "union station", "vista ridge", "willow creek reservoir", "yellow pine trailhead", "aspen grove",
         "blue spring state park", "cliffside rec center", "driftwood cove", "emerald bay", "fox run stables",
         "granite peak", "hidden valley ski area", "indian river lagoon", "jasper lake", "kingsley field",
         "lighthouse point", "mill creek greenway", "osprey landing", "prairie wind park", "rocky fork lake",
         "stonebridge plaza", "tamarack lodge"]
ORG = ["northside logistics", "kessler construction", "blue harbor foods", "summit roofing", "valley packaging",
       "redwood elementary", "harlow steel", "greenfield farms", "metro transit", "lakeview hospital",
       "brightline freight", "carver dental group", "delta pipefitters local 12", "evergreen assisted living",
       "foxglove bakery", "granite state insurance", "hollis auto body", "ironwood brewing", "jefferson high school",
       "keystone credit union", "lakeside montessori", "midstate electric", "northwind airlines", "orchard ridge church",
       "pinnacle fitness", "quarry stone works", "riverbend pharmacy", "silverline software", "tri-county ems",
       "united grain co-op", "vantage call center", "westfield distribution", "yellowjacket hockey club",
       "allied waste services", "bayside shipyard", "cornerstone realty", "dunmore public library",
       "eastgate veterinary clinic", "frontier cable", "goodwin law offices", "harvest moon diner",
       "interstate movers", "jubilee food bank", "kingston middle school", "lone pine sawmill", "meridian bank",
       "nova robotics", "oakmont country club", "patriot security", "riverside rowing club", "st. brigid parish",
       "tidewater seafood", "upland dairy", "victory nursing home", "wildcat youth soccer", "zenith printing",
       "anchor marine supply", "birchwood daycare", "coastal wind energy", "fairview urgent care"]
MONTHS = ["january", "february", "march", "april", "june", "july", "august", "september", "october", "november"]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 7: "seventh", 9: "ninth", 12: "twelfth",
       14: "fourteenth", 18: "eighteenth", 21: "twenty first", 23: "twenty third", 27: "twenty seventh"}

SLOT_CATEGORY = {"PATIENT_FIRST": "PATIENT_NAME", "PATIENT_LAST": "PATIENT_NAME", "DOCTOR_FIRST": "CLINICIAN_NAME",
                 "DOCTOR_LAST": "CLINICIAN_NAME", "OTHER_FIRST": "OTHER_PERSON_NAME", "AGE": "AGE",
                 "DATE_SPOKEN": "DATE", "DATE_NUM": "DATE", "YEAR": "DATE", "CITY": "LOCATION", "PLACE": "LOCATION",
                 "ORG": "ORGANIZATION", "PHONE": "CONTACT", "MRN": "ID"}
SLOT = re.compile(r"\{(" + "|".join(SLOT_CATEGORY) + r")\}")
# multi-word forms gold must list as their own strings (the evaluator counts every listed string)
COMPOUND = re.compile(r"(?:(?:mr|mrs|ms|dr|doctor)\.?[ \t]+)?\{(?:PATIENT|DOCTOR)_(?:FIRST|LAST)\}"
                      r"(?:[ \t]+\{(?:PATIENT|DOCTOR)_LAST\})?"
                      r"|\{AGE\}(?:[ \t]*-?[ \t]*years?[ \t]*-?[ \t]*old|[ \t]+years?[ \t]+of[ \t]+age)?",
                      re.IGNORECASE)


def identity(sex: str, rng: random.Random, age_range=(19, 84)) -> dict:
    day = rng.choice(list(ORD))
    mon = rng.randrange(len(MONTHS))
    year = rng.randint(2019, 2025)
    ident = {"PATIENT_FIRST": rng.choice(FIRST[sex]), "PATIENT_LAST": rng.choice(LAST),
             "DOCTOR_FIRST": rng.choice(FIRST[rng.choice(["female", "male"])]), "DOCTOR_LAST": rng.choice(LAST),
             "OTHER_FIRST": rng.choice(FIRST[rng.choice(["female", "male"])]), "AGE": str(rng.randint(*age_range)),
             "DATE_SPOKEN": f"{MONTHS[mon]} {ORD[day]}", "DATE_NUM": f"{mon + 1:02d}/{day:02d}/{year}",
             "YEAR": str(year), "CITY": rng.choice(CITY), "PLACE": rng.choice(PLACE), "ORG": rng.choice(ORG),
             "PHONE": f"{rng.randint(200, 989)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
             "MRN": f"mrn{rng.randint(100000, 999999)}"}
    # distinct people get distinct names: fill() labels each string once, so a shared name would lose a category
    # (redrawn after all other draws, so identities without a collision are unchanged)
    while ident["DOCTOR_FIRST"] == ident["PATIENT_FIRST"]:
        ident["DOCTOR_FIRST"] = rng.choice(FIRST[rng.choice(["female", "male"])])
    while ident["DOCTOR_LAST"] == ident["PATIENT_LAST"]:
        ident["DOCTOR_LAST"] = rng.choice(LAST)
    while ident["OTHER_FIRST"] in (ident["PATIENT_FIRST"], ident["DOCTOR_FIRST"]):
        ident["OTHER_FIRST"] = rng.choice(FIRST[rng.choice(["female", "male"])])
    return ident


def _cased(value: str, field: str) -> str:
    return value.lower() if field == "dialogue" else value.title() if not re.search(r"\d", value) else value


def fill(t: dict, seed: int = 0) -> tuple[dict, dict]:
    """(record, gold) for one template and seed."""
    rng = random.Random(f"{t['id']}:{seed}")
    ident = identity(t.get("sex", "female"), rng, t.get("age_range", (19, 84)))
    plain = " ".join(SLOT.sub(" ", t[f]) for f in FIELDS).lower()
    pools = {"PATIENT_FIRST": FIRST[t.get("sex", "female")], "PATIENT_LAST": LAST, "DOCTOR_LAST": LAST,
             "DOCTOR_FIRST": FIRST["female"] + FIRST["male"], "OTHER_FIRST": FIRST["female"] + FIRST["male"],
             "CITY": CITY, "PLACE": PLACE, "ORG": ORG}
    for slot, pool in pools.items():   # a value that is also ordinary text of the template would be over-counted
        while re.search(rf"\b{re.escape(ident[slot])}\b", plain):
            ident[slot] = rng.choice(pool)
    eid = f"{t['id']}" + (f"-{seed}" if seed else "")
    rec = {"encounter_id": eid, "dataset": t.get("dataset", "synthetic_injury"), "dialogue": "", "note": "",
           "icd10_codes": t["icd10_codes"]}
    pii = []
    for f in FIELDS:
        tmpl = t[f]
        rec[f] = SLOT.sub(lambda m: _cased(ident[m.group(1)], f), tmpl)
        strings = {}
        for m in COMPOUND.finditer(tmpl):
            slots = SLOT.findall(m.group(0))
            filled = SLOT.sub(lambda mm: _cased(ident[mm.group(1)], f), m.group(0))
            strings.setdefault(filled, SLOT_CATEGORY[slots[0]])
        for m in SLOT.finditer(tmpl):
            strings.setdefault(_cased(ident[m.group(1)], f), SLOT_CATEGORY[m.group(1)])
        for s, cat in strings.items():
            pii.append({"text": s, "field": f, "category": cat, "tier": TIER[cat], "comment": "synthetic slot"})
    for e in t.get("pii_fixed", []):
        pii.append({**e, "tier": TIER[e["category"]]})
    gold = {"encounter_id": eid, "pii": pii, "medical_concepts": t["medical_concepts"],
            "ambiguous_terms": t.get("ambiguous_terms", []), "evidence": t["evidence"]}
    return rec, gold


def check(t: dict) -> list[str]:
    errs = []
    codes = {c["code"] for c in t.get("icd10_codes", [])}
    if t.get("dataset", "synthetic_injury") == "synthetic_injury" and not any(c[0] in "VWXY" for c in codes):
        errs.append("no external-cause (V00-Y99) code")
    errs += [f"code not in ICD-10-CM table: {c}" for c in codes if not is_valid_code(c)]
    plain = {f: SLOT.sub("X", t.get(f, "")) for f in FIELDS}
    for f in FIELDS:
        if not t.get(f, "").strip():
            errs.append(f"empty {f}")
        if f == "dialogue" and SLOT.sub("", t[f]) != SLOT.sub("", t[f]).lower():
            errs.append("dialogue must be lowercase outside slots")
        bad = set(re.findall(r"\{([A-Z_]+)\}", t.get(f, ""))) - set(SLOT_CATEGORY)
        errs += [f"unknown slot {{{s}}} in {f}" for s in bad]
    for key in ("medical_concepts", "ambiguous_terms", "pii_fixed"):
        for e in t.get(key, []):
            if e.get("field") not in FIELDS or e.get("text", "") not in plain.get(e.get("field"), ""):
                errs.append(f"{key}: not verbatim in {e.get('field')}: {e.get('text')!r}")
    for e in t.get("medical_concepts", []):
        if e.get("type") not in CONCEPT_TYPES:
            errs.append(f"medical_concepts: bad type {e.get('type')!r}")
    for e in t.get("pii_fixed", []):
        if e.get("category") not in TIER:
            errs.append(f"pii_fixed: bad category {e.get('category')!r}")
    rel = {(c["field"], c["text"]) for c in t.get("medical_concepts", []) if c.get("icd_relevant")}
    ev_codes = set()
    for ev in t.get("evidence", []):
        ev_codes.add(ev.get("code"))
        for s in ev.get("spans", []):
            if (s.get("field"), s.get("text")) not in rel:
                errs.append(f"evidence span not an icd_relevant concept: {s}")
    errs += [f"no evidence entry for {c}" for c in codes - ev_codes]
    return errs


def load_templates() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(TEMPLATES.glob("*.json"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "build"])
    ap.add_argument("--seeds", type=int, default=1, help="identities per template")
    ap.add_argument("--only", default=None, help="check one template id")
    args = ap.parse_args()
    ts = load_templates()
    if args.only:
        ts = [t for t in ts if t["id"] == args.only]
    bad = 0
    for t in ts:
        errs = check(t)
        if errs:
            bad += 1
            print(f"! {t['id']}:\n  - " + "\n  - ".join(errs))
    if args.cmd == "check":
        print(f"{len(ts) - bad}/{len(ts)} templates OK")
        sys.exit(1 if bad else 0)
    rows = [fill(t, s) for t in ts if not check(t) for s in range(args.seeds)]
    write_jsonl(OUT_DATA, [r for r, _ in rows])
    write_jsonl(OUT_GOLD, [g for _, g in rows])
    print(f"wrote {len(rows)} records -> {OUT_DATA} and {OUT_GOLD}")


if __name__ == "__main__":
    main()
