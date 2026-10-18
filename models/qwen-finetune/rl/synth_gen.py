"""Generate varied synthetic templates with Claude Haiku (rl/synth/templates/G*.json, format of rl/synth.py).

The 40 hand-made S*.json templates are all injury encounters. These add new scenarios: every template gets a
scenario drawn from a seeded matrix (specialty, visit type, speakers, length, 2-3 PII categories to feature, 1-2
hard negatives), so the set covers the categories the train data is thin on (OCCUPATION, OTHER_QUASI, LOCATION,
ORGANIZATION, OTHER_PERSON_NAME, CONTACT, ID). Labels come by construction: identifiers are slots, PII written
literally (occupations, hobbies, life events) is listed in pii_fixed.

Per template:
  1. Haiku writes the template as JSON (structured output); ICD descriptions are replaced by the CMS titles.
  2. rl/synth.py check + the checks below; on errors Haiku gets them back and fixes the template (max --attempts).
  3. A second Haiku call reviews the PII labels on the slot-free text: literal PII the template missed is added to
     pii_fixed (verbatim only), pii_fixed entries that are not PII are dropped. Then the checks run again.
Failures go to rl/synth/gen_failures.jsonl; existing G*.json files are skipped, so a rerun resumes.

  .venv-train/bin/python rl/synth_gen.py --dry-run --n 3          print scenarios and the prompt (no API call)
  .venv-train/bin/python rl/synth_gen.py --n 5                    G0001-G0005 (look at them before the full run)
  .venv-train/bin/python rl/synth_gen.py --n 834                  166 train encounters + 834 = 1000
  .venv-train/bin/python rl/synth_gen.py --check --start 1 --n 5  normalize + check G files written by hand/agents
  python3 rl/synth.py build                                       then rebuild data/synth_external_cause.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import synth as S  # noqa: E402
from spec import fmt_code, icd_table, norm_code  # noqa: E402

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50   # $/MTok, prompts under 100K tokens
FAILURES = S.TEMPLATES.parent / "gen_failures.jsonl"

# ---------------------------------------------------------------- scenario matrix

SPECIALTIES = [
    "family medicine", "internal medicine", "cardiology", "endocrinology", "gastroenterology", "pulmonology",
    "nephrology", "rheumatology", "neurology", "psychiatry", "dermatology", "orthopedics", "sports medicine",
    "obstetrics", "gynecology", "urology", "ENT", "ophthalmology", "allergy and immunology", "infectious disease",
    "hematology", "oncology", "geriatrics", "pediatrics (adolescent patient with a parent)", "pain management",
    "physical medicine and rehabilitation", "occupational medicine", "urgent care", "emergency department",
    "general surgery follow-up", "sleep medicine", "podiatry", "travel medicine", "addiction medicine",
]
VISITS = [  # (dataset suffix, description, weight)
    ("followup", "follow-up visit for a chronic condition", 4),
    ("new", "new patient consultation on referral", 3),
    ("acute", "acute visit for a new problem", 3),
    ("annual", "annual physical / wellness visit with one or two active problems", 1),
    ("postop", "post-operative or post-procedure check", 1),
    ("telehealth", "telehealth video visit (mentions of the camera, connection, pharmacy for pick-up)", 1),
    ("injury", "injury visit: the external cause (mechanism, place, activity) must be coded with V00-Y99 codes", 2),
]
FOCUS = [  # (category, what to write, weight)
    ("OCCUPATION", "the patient's (or a relative's) specific job, how it relates to the complaint, phrased "
                   "naturally and more than once (e.g. 'i drive a forklift on the night shift', 'i'm a pastry chef')", 4),
    ("OTHER_QUASI", "rare personal details as full clauses: an unusual hobby with specifics, family size, veteran "
                    "status, a life event, a distinctive routine, a pet with a name-free description "
                    "(e.g. 'i restore vintage motorcycles in my garage', 'we just adopted twins', 'i swim at 5:45 every morning')", 5),
    ("LOCATION", "named places: hometown, a recent trip, a venue, a street or neighborhood ({CITY}/{PLACE} slots "
                 "and/or literal place names listed in pii_fixed)", 3),
    ("ORGANIZATION", "named employer, school, team, church, pharmacy or other clinic ({ORG} slot and/or literal "
                     "names listed in pii_fixed)", 3),
    ("OTHER_PERSON_NAME", "a named relative, friend, coworker or another clinician who is referred to by name "
                          "({OTHER_FIRST} slot)", 2),
    ("DATE", "calendar dates: a surgery date, a birthday, a date of last visit, a year of diagnosis "
             "({DATE_SPOKEN}/{DATE_NUM}/{YEAR} slots)", 2),
    ("CONTACT", "a callback phone number read out or written in the note ({PHONE} slot)", 1),
    ("ID", "a medical record number mentioned ({MRN} slot)", 1),
]
HARD_NEGATIVES = [
    "an eponymous disease or sign (parkinson's, crohn's, graves' disease, hashimoto's, bell's palsy, addison's, "
    "cushing's, raynaud's, achilles tendon, baker's cyst, barrett's esophagus) - label it in ambiguous_terms, is_pii false",
    "numbers that are not PII: doses, vitals, lab values, pain scores ('7/10'), durations ('for three weeks')",
    "guideline ages that are not PII ('screening starts at 45', 'over the age of 65')",
    "relative time that is not PII ('last tuesday', 'two years ago', 'a 3-month follow-up')",
    "a brand or software name that looks like a proper noun (tylenol, eliquis, fitbit, dexcom, dragon) - "
    "label it in ambiguous_terms, is_pii false",
    "a generic place or activity that supports an external-cause or Z code and is NOT PII ('at work in the "
    "warehouse', 'in the kitchen', 'playing pickleball') - label it in medical_concepts as a finding",
    "a word that is both a surname and a common word or place used non-identifyingly ('the washington guidelines' "
    "is not a person; 'she went to the pharmacy' names no organization)",
]
SPEAKERS = ["doctor and patient", "doctor and patient", "doctor, patient and a family member who speaks too "
            "(their lines start with [patient] as in the source data)", "doctor and patient; a nurse is mentioned"]
LENGTHS = [("short", 2500, 3500, 1200), ("medium", 4000, 5500, 1800), ("long", 6000, 8000, 2600)]
AGE_BANDS = [(19, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
# specialties that fix the patient's sex or age band
SEX_OF = {"obstetrics": "female", "gynecology": "female"}
AGES_OF = {"obstetrics": [(19, 30), (30, 45)], "geriatrics": [(75, 90)],
           "pediatrics (adolescent patient with a parent)": [(13, 17)]}


def scenario(i: int) -> dict:
    rng = random.Random(f"G:{i}")
    visit = rng.choices(VISITS, weights=[w for *_, w in VISITS])[0]
    k = rng.choice([2, 2, 3])
    focus = []
    while len(focus) < k:
        c = rng.choices(FOCUS, weights=[w for *_, w in FOCUS])[0]
        if c not in focus:
            focus.append(c)
    length = rng.choices(LENGTHS, weights=[2, 5, 3])[0]
    spec = rng.choice(SPECIALTIES)
    sex = SEX_OF.get(spec) or rng.choice(["female", "male"])
    return {"id": f"G{i:04d}", "specialty": spec, "visit": visit[0], "visit_desc": visit[1],
            "sex": sex, "age_range": rng.choice(AGES_OF.get(spec, AGE_BANDS)), "speakers": rng.choice(SPEAKERS),
            "length": length[0], "dialogue_chars": length[1:3], "note_chars": length[3],
            "focus": [(c, d) for c, d, _ in focus], "hard_negatives": rng.sample(HARD_NEGATIVES, rng.choice([1, 2])),
            "n_codes": rng.choice([2, 3, 3, 4, 5])}


# ---------------------------------------------------------------- prompts

def _example() -> str:
    t = json.loads((S.TEMPLATES / "S001.json").read_text())
    dlg = "\n".join(t["dialogue"].split("\n")[:12])
    note = t["note"][:900]
    ex = {"dialogue (first lines)": dlg, "note (start)": note, "pii_fixed": t["pii_fixed"][:3],
          "ambiguous_terms": t["ambiguous_terms"][:2], "medical_concepts (some)": t["medical_concepts"][:5],
          "evidence (one)": {**t["evidence"][1], "spans": t["evidence"][1]["spans"][:3]}}
    return json.dumps(ex, indent=1, ensure_ascii=False)


SYSTEM = f"""You write synthetic doctor-patient encounters for training a de-identification model. Each encounter is a
TEMPLATE: a recorded visit transcript ("dialogue") and the physician note written from it ("note"), plus labels.

## Identifier slots
Every identifier of these kinds is written as a slot, never as a literal value:
{{PATIENT_FIRST}} {{PATIENT_LAST}} {{DOCTOR_FIRST}} {{DOCTOR_LAST}} {{OTHER_FIRST}} (a named relative/friend/other person)
{{AGE}} (digits only: "{{AGE}}-year-old", "i'm {{AGE}}") {{DATE_SPOKEN}} ("march third") {{DATE_NUM}} ("03/03/2024")
{{YEAR}} {{CITY}} {{PLACE}} (a named venue/park/area) {{ORG}} (a named employer/school/pharmacy/team) {{PHONE}} {{MRN}}
Use the same slot for the same entity throughout. A second named place or organization that needs a different value is
written literally and listed in pii_fixed. Do NOT write any person name literally.

## Format
- dialogue: one turn per line, each line starting with "[doctor] " or "[patient] " (family members also use
  [patient]). Everything outside slots is lowercase, with spaced punctuation and split contractions as in automatic
  transcripts: "hi , {{PATIENT_FIRST}} . how are you ?", "i do n't know", "it 's been rough". Natural speech: fillers,
  small talk, interruptions, numbers mostly spelled out ("one forty over ninety").
- note: a normal cased clinical note with section headers (e.g. CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS,
  SOCIAL HISTORY, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN) that agrees with the dialogue.
- icd10_codes: valid billable ICD-10-CM codes (dotted, e.g. "E11.65") supported by the text.

## Labels
All "text" values are exact, verbatim substrings of that field (same case and spacing) and never contain a slot:
slots are labelled automatically.
- pii_fixed: every identifying detail written literally (not as a slot), once per distinct string per field, with
  category one of OCCUPATION, OTHER_QUASI, LOCATION, ORGANIZATION, DATE, AGE and a short comment. Keep spans minimal
  but complete: "pastry chef", "i restore vintage motorcycles in my garage". The same detail in dialogue and note is
  two entries (one per field).
  OCCUPATION: the person's job or role. OTHER_QUASI: rare personal details that narrow down who it is (unusual hobby
  with specifics, family size, veteran status, life events, distinctive routines). NOT PII: generic roles ("my wife",
  "the nurse"), weekdays, relative time and durations, doses, vitals, lab values, pain scores, guideline ages,
  diagnoses and other medical content.
- medical_concepts: EVERY clinically meaningful mention in both fields (diagnoses, symptoms, findings, labs/tests,
  medications with dose if written together, procedures, anatomy) - typically 60-120 entries. type is one of
  diagnosis, symptom, finding, lab_or_test, medication, procedure, anatomy. icd_relevant true when the concept supports,
  refines or rules out one of the icd10_codes. External-cause and Z-code context (activity, mechanism, place of
  occurrence, occupational exposure, lifestyle) is a medical concept of type finding, never PII; if it also names a
  specific place or employer, put the generic part in medical_concepts ("warehouse") and the name in pii_fixed.
- ambiguous_terms: strings that could be read as PII or as medical/other (eponyms, brand names, software, place-like
  words), with is_pii and a short comment.
- evidence: one entry per code with 2-8 supporting spans from either field; every span must also be a
  medical_concepts entry with icd_relevant true (same text and field).

## Example (excerpt of an existing template, injury visit)
{_example()}
"""


def user_prompt(sc: dict) -> str:
    focus = "\n".join(f"  - {c}: {d}" for c, d in sc["focus"])
    neg = "\n".join(f"  - {h}" for h in sc["hard_negatives"])
    ext = ("Include the external-cause codes (V00-Y99: mechanism, place of occurrence Y92, activity Y93) that the "
           "story supports, in addition to the injury code(s)." if sc["visit"] == "injury" else
           "Do not make this an injury visit.")
    return f"""Write one new template.

- Specialty / setting: {sc['specialty']}
- Visit: {sc['visit_desc']}
- Patient: {sc['sex']}, age {sc['age_range'][0]}-{sc['age_range'][1]} (written as the {{AGE}} slot; keep the story
  consistent with any age in that range)
- Speakers: {sc['speakers']}
- Length: dialogue about {sc['dialogue_chars'][0]}-{sc['dialogue_chars'][1]} characters, note about {sc['note_chars']}
- ICD-10-CM codes: {sc['n_codes']}. {ext}
- Feature these identifier categories prominently (several mentions, natural, specific, not generic):
{focus}
- Also include these hard negatives (things that must NOT be labelled PII):
{neg}
- Use at least {{PATIENT_FIRST}} or {{PATIENT_LAST}} and {{DOCTOR_LAST}} in the dialogue and the patient's name and
  {{AGE}} in the note. Make the story specific and different from a textbook case: concrete circumstances, the
  patient's own words, a realistic plan."""


REVIEW_SYSTEM = """You audit PII labels of a synthetic clinical encounter. Identifier slots in braces ({PATIENT_FIRST},
{CITY}, ...) are already labelled. Report only literal text, as exact verbatim substrings of the given field.

PII categories: PATIENT_NAME, CLINICIAN_NAME, OTHER_PERSON_NAME (any literal person name), DATE (calendar dates,
years, birthdays/holidays used as anchors), CONTACT, ID, AGE (exact ages), LOCATION (named places), ORGANIZATION
(named employers, schools, pharmacies, teams), OCCUPATION (a person's job or role), OTHER_QUASI (rare personal details:
unusual hobbies with specifics, family size, veteran status, life events, distinctive routines).
NOT PII: generic roles ("my wife", "the nurse"), weekdays, relative time, durations, doses, vitals, lab values, pain
scores, guideline ages, diagnoses, eponymous diseases, drug and device brands, generic places or activities that
describe how an injury happened ("in the kitchen", "playing soccer")."""


def review_prompt(t: dict) -> str:
    return (f"Listed literal PII (pii_fixed):\n{json.dumps(t['pii_fixed'], ensure_ascii=False)}\n\n"
            f"dialogue:\n{t['dialogue']}\n\nnote:\n{t['note']}\n\n"
            "Return `missing`: literal PII not in the list (one entry per distinct string per field, minimal span), "
            "and `not_pii`: listed entries that are not PII by the rules (copy their text and field).")


# ---------------------------------------------------------------- schemas

def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


FIELD = {"type": "string", "enum": list(S.FIELDS)}
STR = {"type": "string"}
FIXED_CATS = ["OCCUPATION", "OTHER_QUASI", "LOCATION", "ORGANIZATION", "DATE", "AGE"]
TEMPLATE_SCHEMA = _obj({
    "icd10_codes": {"type": "array", "items": _obj({"code": STR, "description": STR})},
    "dialogue": STR,
    "note": STR,
    "medical_concepts": {"type": "array", "items": _obj({
        "text": STR, "field": FIELD, "type": {"type": "string", "enum": sorted(S.CONCEPT_TYPES)},
        "icd_relevant": {"type": "boolean"}})},
    "ambiguous_terms": {"type": "array", "items": _obj({"text": STR, "field": FIELD, "is_pii": {"type": "boolean"},
                                                        "comment": STR})},
    "evidence": {"type": "array", "items": _obj({"code": STR, "spans": {"type": "array",
                                                                        "items": _obj({"text": STR, "field": FIELD})}})},
    "pii_fixed": {"type": "array", "items": _obj({"text": STR, "field": FIELD,
                                                  "category": {"type": "string", "enum": FIXED_CATS},
                                                  "comment": STR})},
})
REVIEW_SCHEMA = _obj({
    "missing": {"type": "array", "items": _obj({"text": STR, "field": FIELD, "category": {
        "type": "string", "enum": sorted(S.TIER)}, "comment": STR})},
    "not_pii": {"type": "array", "items": _obj({"text": STR, "field": FIELD})},
})


# ---------------------------------------------------------------- checks

def normalize(sc: dict, raw: dict) -> dict:
    """Template dict from the model output: id/dataset, CMS titles, codes in dotted form, deduplicated labels."""
    table = icd_table()
    codes, seen = [], set()
    for c in raw["icd10_codes"]:
        code = fmt_code(c["code"])
        if norm_code(code) not in seen:
            seen.add(norm_code(code))
            codes.append({"code": code, "description": table.get(norm_code(code), c["description"])})
    remap = {norm_code(c["code"]): fmt_code(c["code"]) for c in raw["icd10_codes"]}
    evidence = [{"code": remap.get(norm_code(e["code"]), e["code"]), "spans": e["spans"]} for e in raw["evidence"]]

    def dedup(xs, key):
        out, keys = [], set()
        for x in xs:
            if key(x) not in keys:
                keys.add(key(x))
                out.append(x)
        return out

    return {"id": sc["id"], "dataset": f"synthetic_{sc['visit']}", "sex": sc["sex"], "age_range": sc["age_range"],
            "icd10_codes": codes,
            "dialogue": raw["dialogue"].strip(), "note": raw["note"].strip(),
            "medical_concepts": dedup(raw["medical_concepts"], lambda e: (e["field"], e["text"])),
            "ambiguous_terms": dedup(raw["ambiguous_terms"], lambda e: (e["field"], e["text"])),
            "evidence": evidence,
            "pii_fixed": dedup(raw["pii_fixed"], lambda e: (e["field"], e["text"])),
            "scenario": {k: sc[k] for k in ("specialty", "visit", "length")} | {"focus": [c for c, _ in sc["focus"]]}}


def problems(t: dict, sc: dict) -> list[str]:
    errs = S.check(t)
    lines = [l for l in t["dialogue"].split("\n") if l.strip()]
    untagged = [l for l in lines if not re.match(r"^\[(doctor|patient)\] ", l)]
    if untagged:
        errs.append(f"{len(untagged)} dialogue lines do not start with '[doctor] ' or '[patient] ', e.g. {untagged[0][:60]!r}")
    if len(t["dialogue"]) < 0.6 * sc["dialogue_chars"][0]:
        errs.append(f"dialogue too short: {len(t['dialogue'])} characters, aim for {sc['dialogue_chars'][0]}+")
    if len(t["note"]) < 0.5 * sc["note_chars"]:
        errs.append(f"note too short: {len(t['note'])} characters, aim for about {sc['note_chars']}")
    if len(t["medical_concepts"]) < 25:
        errs.append(f"only {len(t['medical_concepts'])} medical_concepts: label every clinical mention in both fields")
    if not re.search(r"\{PATIENT_(FIRST|LAST)\}", t["note"]):
        errs.append("the note never names the patient ({PATIENT_FIRST}/{PATIENT_LAST})")
    rec, gold = S.fill(t)
    cats = {e["category"] for e in gold["pii"]}
    errs += [f"featured category {c} has no labelled mention (slot or pii_fixed)" for c, _ in sc["focus"] if c not in cats]
    return errs


# ---------------------------------------------------------------- API calls

class Usage:
    def __init__(self):
        self.inp = self.cache_w = self.cache_r = self.out = 0

    def add(self, u):
        self.inp += u.input_tokens
        self.cache_w += u.cache_creation_input_tokens or 0
        self.cache_r += u.cache_read_input_tokens or 0
        self.out += u.output_tokens

    def dollars(self) -> float:
        return (self.inp + 1.25 * self.cache_w + 0.1 * self.cache_r) * PRICE_IN / 1e6 + self.out * PRICE_OUT / 1e6


class Refused(Exception):
    pass


async def call(client, usage: Usage, system: str, messages: list, schema: dict, max_tokens: int):
    """One structured-output request; returns (parsed JSON, assistant content to append for a follow-up turn)."""
    async with client.messages.stream(
        model=MODEL, max_tokens=max_tokens, system=system, messages=messages,
        cache_control={"type": "ephemeral"}, output_config={"effort": "medium", "format": {
            "type": "json_schema", "schema": schema}},
    ) as stream:
        msg = await stream.get_final_message()
    usage.add(msg.usage)
    if msg.stop_reason == "refusal":
        raise Refused(getattr(msg.stop_details, "category", None))
    if msg.stop_reason == "max_tokens":
        raise ValueError("output hit max_tokens")
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text), msg.content


def apply_review(t: dict, rv: dict) -> tuple[dict, list[str]]:
    notes = []
    drop = {(e["field"], e["text"]) for e in rv["not_pii"]}
    keep = [e for e in t["pii_fixed"] if (e["field"], e["text"]) not in drop]
    notes += [f"dropped {f}:{x!r}" for f, x in drop if any((e["field"], e["text"]) == (f, x) for e in t["pii_fixed"])]
    have = {(e["field"], e["text"]) for e in keep}
    concepts = {(e["field"], e["text"].lower()) for e in t["medical_concepts"]}
    plain = {f: S.SLOT.sub("X", t[f]) for f in S.FIELDS}
    for e in rv["missing"]:
        k = (e["field"], e["text"])
        if k in have or not e["text"].strip() or e["text"] not in plain[e["field"]]:
            continue
        if (e["field"], e["text"].lower()) in concepts:
            continue   # spec: external-cause / medical context stays a concept
        keep.append({"text": e["text"], "field": e["field"], "category": e["category"],
                     "comment": e.get("comment") or "added by review"})
        have.add(k)
        notes.append(f"added {e['category']} {e['field']}:{e['text']!r}")
    return {**t, "pii_fixed": keep}, notes


async def generate(client, sc: dict, usage: Usage, attempts: int) -> dict:
    messages = [{"role": "user", "content": user_prompt(sc)}]
    errs = []
    for a in range(attempts):
        raw, content = await call(client, usage, SYSTEM, messages, TEMPLATE_SCHEMA, 32000)
        t = normalize(sc, raw)
        errs = problems(t, sc)
        if not errs:
            break
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": "The template has these problems:\n- " + "\n- ".join(errs[:40]) +
                      "\nReturn the complete corrected template."}]
    else:
        raise ValueError(f"still invalid after {attempts} attempts: {errs[:10]}")
    rv, _ = await call(client, usage, REVIEW_SYSTEM, [{"role": "user", "content": review_prompt(t)}], REVIEW_SCHEMA, 16000)
    t, notes = apply_review(t, rv)
    errs = problems(t, sc)
    if errs:
        raise ValueError(f"invalid after review: {errs[:10]}")
    t["scenario"]["attempts"] = a + 1
    t["scenario"]["review"] = notes
    return t


async def run(args) -> None:
    import anthropic
    client = anthropic.AsyncAnthropic(max_retries=6)
    todo = [scenario(i) for i in range(args.start, args.start + args.n)
            if not (S.TEMPLATES / f"G{i:04d}.json").exists()]
    print(f"[gen] {len(todo)} templates to write ({args.n - len(todo)} exist), {MODEL}, concurrency {args.concurrency}")
    usage, sem, done, failed, t0 = Usage(), asyncio.Semaphore(args.concurrency), 0, 0, time.time()

    async def one(sc):
        nonlocal done, failed
        async with sem:
            try:
                t = await generate(client, sc, usage, args.attempts)
                (S.TEMPLATES / f"{sc['id']}.json").write_text(json.dumps(t, indent=1, ensure_ascii=False) + "\n")
                done += 1
            except Exception as e:   # one bad template must not stop the run
                failed += 1
                with open(FAILURES, "a") as f:
                    f.write(json.dumps({"id": sc["id"], "error": f"{type(e).__name__}: {e}"[:2000]}) + "\n")
                if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
                    raise
            n = done + failed
            if n % 10 == 0 or n == len(todo):
                print(f"[gen] {n}/{len(todo)} ok {done} failed {failed} ${usage.dollars():.2f} "
                      f"({time.time() - t0:.0f}s, cache read {usage.cache_r / max(1, usage.inp + usage.cache_r + usage.cache_w):.0%})",
                      flush=True)

    await asyncio.gather(*(one(sc) for sc in todo))
    print(f"[gen] done: {done} written, {failed} failed (see {FAILURES}), "
          f"tokens in {usage.inp + usage.cache_w + usage.cache_r} out {usage.out}, ${usage.dollars():.2f}")


def check_files(start: int, n: int) -> int:
    bad = 0
    for i in range(start, start + n):
        p, sc = S.TEMPLATES / f"G{i:04d}.json", scenario(i)
        if not p.exists():
            print(f"- {sc['id']}: missing")
            bad += 1
            continue
        try:
            raw = json.loads(p.read_text())
            t = normalize(sc, raw)
            t["scenario"] |= {k: v for k, v in raw.get("scenario", {}).items() if k not in t["scenario"]}
            errs = problems(t, sc)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errs = [f"unreadable template: {type(e).__name__}: {e}"]
        else:
            p.write_text(json.dumps(t, indent=1, ensure_ascii=False) + "\n")
        if errs:
            bad += 1
            print(f"! {sc['id']}:\n  - " + "\n  - ".join(errs))
        else:
            print(f"ok {sc['id']} ({len(t['dialogue'])}/{len(t['note'])} chars, {len(t['medical_concepts'])} concepts, "
                  f"{len(t['pii_fixed'])} pii_fixed)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=1, help="first template number (G0001)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=3, help="generation turns per template (fixes included)")
    ap.add_argument("--dry-run", action="store_true", help="print the scenarios and the prompts, no API call")
    ap.add_argument("--no-system", action="store_true", help="with --dry-run: only the per-template prompts")
    ap.add_argument("--check", action="store_true",
                    help="normalize and check existing G files of the range (CMS titles, id/dataset/sex/age_range), "
                         "for templates written without the API; exit 1 if any has problems")
    args = ap.parse_args()
    if args.check:
        sys.exit(check_files(args.start, args.n))
    if args.dry_run:
        if not args.no_system:
            print(SYSTEM)
        for i in range(args.start, args.start + args.n):
            print("=" * 100)
            print(user_prompt(scenario(i)))
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
