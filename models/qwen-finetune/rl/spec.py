"""Label spec of the policy: entity types, the output format, the system prompt and the closed list of category words.

The policy reads one encounter and writes one JSON object (TRAINING_PLAN.md §2):

  {"dx": [{"code": "E11.9", "desc": "type 2 diabetes mellitus without complications"}],
   "evidence": [{"code": "E11.9", "spans": ["type 2 diabetes", "hemoglobin a1c of 8.3"]}],
   "entities": [{"text": "martha", "type": "PATIENT_NAME", "reason": "patient first name"},
                {"text": "parkinson", "ctx": "mr. parkinson", "type": "OTHER_PERSON_NAME", "reason": "relative"},
                {"text": "mount baker", "type": "LOCATION", "cat": "a mountain", "reason": "place of the fall"}],
   "keep": [{"text": "parkinson's disease", "reason": "diagnosis eponym"}]}

`dx` lists the diagnoses, `evidence` names their support in the text (one item per dx code), `entities` are the spans
to remove, `keep` vetoes regex spans and explains retained ambiguous terms. When the record has icd10_codes, the
prompt builder prefills `{"dx": [...], "evidence": [` as the assistant prefix (codes never enter the user turn), so
the model writes the evidence and everything after it. The plan's example nests evidence inside each dx item; a
separate `evidence` list is used instead because an assistant prefix can only fix a prefix of the output: with
nesting, the second dx item would come after model-written text and could not be prefilled.
"""
from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / "resources"
ICD_TABLE = RESOURCES / "icd10cm_2026.tsv.gz"   # CMS FY2026 ICD-10-CM order file (code, billable, title)

TIER = {"PATIENT_NAME": "direct", "CLINICIAN_NAME": "direct", "OTHER_PERSON_NAME": "direct", "DATE": "direct",
        "CONTACT": "direct", "ID": "direct", "AGE": "quasi", "LOCATION": "quasi", "ORGANIZATION": "quasi",
        "OCCUPATION": "quasi", "OTHER_QUASI": "quasi"}
TYPES = list(TIER)
NAME_TYPES = {"PATIENT_NAME", "CLINICIAN_NAME", "OTHER_PERSON_NAME"}
QUASI_TYPES = {"LOCATION", "ORGANIZATION", "OCCUPATION", "OTHER_QUASI"}
# Types of the zero-shot harness (anonymize.PII_TYPES) that a model may still emit
LEGACY_TYPES = {"PERSON": "OTHER_PERSON_NAME", "NAME": "OTHER_PERSON_NAME", "PATIENT": "PATIENT_NAME",
                "DOCTOR": "CLINICIAN_NAME"}

MAX_SPAN_WORDS = 6
MAX_REASON_WORDS = 8


def norm_type(t) -> str | None:
    t = str(t or "").strip().upper()
    t = LEGACY_TYPES.get(t, t)
    return t if t in TIER else None


def norm_code(c) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(c).upper())


def fmt_code(c: str) -> str:
    """'E119' -> 'E11.9' (the dataset's dotted form)."""
    c = norm_code(c)
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


def short_reason(text, n: int = MAX_REASON_WORDS) -> str:
    return " ".join(str(text or "").split()[:n])


# ---------------------------------------------------------------- ICD-10-CM table

@lru_cache(maxsize=1)
def icd_table() -> dict[str, str]:
    """{normalized code: title} for every ICD-10-CM code and header in the FY2026 table."""
    with gzip.open(ICD_TABLE, "rt", encoding="utf-8") as f:
        next(f)
        return {code: title for code, _, title in (line.rstrip("\n").split("\t", 2) for line in f)}


def is_valid_code(code: str) -> bool:
    return norm_code(code) in icd_table()


# ---------------------------------------------------------------- category words (external-cause places)

_Y92_SUFFIX = re.compile(r"\s+as the place of occurrence of the external cause.*$", re.IGNORECASE)


@lru_cache(maxsize=1)
def category_words() -> frozenset:
    """Closed list of words that replace a place which is ICD evidence ("mount baker" -> "a mountain").

    Built from the billable Y92 titles: "Supermarket, store or market as the place of occurrence ..." gives
    "a supermarket", "a store", "a market". Activity words (Y93) and occupational exposure (Z57) stay verbatim in
    the text, so only places need a category word.
    """
    words = {"a place", "a mountain", "a park", "a trail", "a lake", "a river", "a beach", "a ski resort",
             "a golf course", "a gym", "a workplace", "a warehouse", "a farm", "a construction site", "a store",
             "a restaurant", "a school", "a road", "a highway", "a sports field", "an ice rink", "a pool"}
    for code, title in icd_table().items():
        if not code.startswith("Y92"):
            continue
        head = _Y92_SUFFIX.sub("", title)
        head = re.sub(r"\([^)]*\)", " ", head).lower()
        for alt in re.split(r",| or |/", head):
            alt = " ".join(alt.split())
            if re.search(r" (?:in|of|on|at)\b", alt):   # "kitchen in nursing home": too specific to generalize
                continue
            if 2 < len(alt) <= 30 and not alt.startswith(("other", "unspecified", "non-", "private", "public",
                                                          "local", "the ")):
                words.add(("an " if alt[0] in "aeiou" else "a ") + alt)
    return frozenset(words)


def render_category(cat) -> str | None:
    c = " ".join(str(cat or "").lower().split())
    if not c:
        return None
    if c in category_words():
        return c
    with_art = ("an " if c[0] in "aeiou" else "a ") + c
    return with_art if with_art in category_words() else None


# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You de-identify clinical encounters (a doctor-patient dialogue and the physician note) while keeping all medical information.
Read the whole encounter, then return one JSON object with these keys, in this order:
"dx": the diagnoses addressed, [{"code": "<ICD-10-CM code>", "desc": "<description>"}].
"evidence": for each dx code, the shortest verbatim phrases that support it, [{"code": "...", "spans": ["...", ...]}].
"entities": every piece of private information to remove, [{"text": "<verbatim>", "type": "<TYPE>", "reason": "<max 8 words>"}].
"keep": terms that look private but must stay (eponyms, clinical dates and ages, product names), [{"text": "...", "reason": "..."}].
Types: PATIENT_NAME, CLINICIAN_NAME, OTHER_PERSON_NAME, DATE (calendar dates, years, birthdays), CONTACT (phone, email, address, URL), ID (record or insurance numbers), AGE (exact ages), LOCATION (named places), ORGANIZATION (named employers, clinics, pharmacies, schools), OCCUPATION (a person's job), OTHER_QUASI (rare personal details that narrow down who it is).
Rules:
- "text" is copied exactly from the encounter, at most 6 words, one line. Every occurrence in both the dialogue and the note is removed, so list each string once.
- If the same word is also medical ("parkinson" as a surname and in "parkinson's disease"), add "ctx": a longer verbatim phrase around the private use ("mr. parkinson"); only occurrences inside ctx are removed.
- A named place that is evidence for an external-cause code gets "cat": a category word ("a mountain", "a supermarket"); the activity and the mechanism are medical evidence, never entities.
- Not private: relative time and durations, doses, vitals, lab values, pain scores, guideline ages, gestational ages, weekdays, generic roles (my wife, the nurse), software names (dax, dragon), diseases, symptoms, medications, procedures, anatomy.
- Do not list text in both entities and keep."""

KNOWN_HEADER = "Identifiers already found in earlier parts of this record (remove them here too):"


def user_message(sections: list[tuple[str, str]], known: list[dict] | None = None) -> str:
    """User turn: the encounter's fields as titled sections, plus the carry-over list of a window."""
    parts = []
    if known:
        parts.append(KNOWN_HEADER + "\n" + "\n".join(f"- {k['text']} ({k['type']})" for k in known))
    for title, text in sections:
        parts.append(f"{title}:\n{text}")
    return "\n\n".join(parts)


def dx_prefix(icd10_codes: list[dict]) -> str:
    """Assistant prefix for records with codes: dx is fixed, evidence is left open for the model."""
    dx = [{"code": c["code"], "desc": str(c.get("description", "")).lower()} for c in icd10_codes]
    return '{"dx": ' + json.dumps(dx, ensure_ascii=False) + ', "evidence": ['


def dump_output(out: dict) -> str:
    """Canonical serialization of a target, the same one dx_prefix starts (so a prefix is a string prefix)."""
    def items(xs):
        return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in xs) + "]"
    return ("{" + ", ".join(f'"{k}": {items(out.get(k) or [])}' for k in ("dx", "evidence", "entities", "keep"))
            + "}")
