"""Label harmonization of the three ACI-eval golds into the harness's categories, and the gold levels.

The harness gold (`harness`, data/converted/aci_bench), the VM teacher gold (`vm`, vm-backup/neuralwave/annotations)
and ketai's reference gold (`ketai`, ketai-eval/gold) use different label sets. Every source label maps to one of
the harness's 11 categories plus `CONTACT` (common.CATEGORY_RISK); the source label stays in the item's `label`.
`direct` follows the paper's 4-type PHI schema (PERSON, DATE, CONTACT, ADDRESS without the quasi part) and the task's
example ("names, date of birth"): names, DOB, dates, IDs and contacts.

Levels: `union` (every merged item), `consensus` (direct, or at least two golds agree), `direct`. An item without
vote metadata (every suite except aci_eval) belongs to every level.
"""

from __future__ import annotations

import re
from collections import Counter

from eval.common import CATEGORY_RISK, occurrences

SOURCES = ("harness", "vm", "ketai")  # output order
PRIORITY = ("harness", "ketai", "vm")  # tie-break of category votes
LEVELS = ("union", "consensus", "direct")
DIRECT = frozenset({"PATIENT_NAME", "PERSON_NAME", "DOB", "DATE", "ID", "CONTACT"})

VM_LABELS = {
    "NAME_PATIENT": "PATIENT_NAME", "NAME_RELATIVE": "PERSON_NAME", "NAME_CLINICIAN": "PERSON_NAME", "NAME_OTHER": "PERSON_NAME",
    "AGE": "AGE", "DATE": "DATE", "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION", "CONTACT": "CONTACT", "ID": "ID",
    "OCCUPATION": "OCCUPATION", "ACTIVITY": "OTHER_QI", "LIFE_EVENT": "OTHER_QI", "TEMPORAL": "OTHER_QI", "PERSONAL_DETAIL": "OTHER_QI",
}  # fmt: skip
KETAI_LABELS = {
    "PATIENT_NAME": "PATIENT_NAME", "CLINICIAN_NAME": "PERSON_NAME", "OTHER_PERSON_NAME": "PERSON_NAME", "DATE": "DATE",
    "CONTACT": "CONTACT", "ID": "ID", "AGE": "AGE", "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION",
    "OCCUPATION": "OCCUPATION", "OTHER_QUASI": "OTHER_QI",
}  # fmt: skip
CONCEPT_TYPES = {"LAB_OR_TEST": "TEST"}  # VM: DIAGNOSIS SYMPTOM MEDICATION PROCEDURE TEST FINDING; ketai adds ANATOMY
# Words ketai's gold drops as never PII (ketai-eval/gold/build_gold.py NOT_PII): ketai never votes for them.
KETAI_NOT_PII = frozenset("dax dragon doctor patient nurse monday tuesday wednesday thursday friday saturday sunday".split())
DOB_CUE = re.compile(r"\b(?:date\s+of\s+birth|dob|born)\b", re.IGNORECASE)
DOB_WINDOW = 25


def dob_context(field_text: str, text: str) -> bool:
    """A DOB cue (`date of birth`, `dob`, `born`) in the 25 characters before an occurrence of `text`."""
    return any(DOB_CUE.search(field_text[max(0, s - DOB_WINDOW) : s]) for s, _ in occurrences(field_text, text))


def category(source: str, label: str, text: str, field_text: str = "", dob_elsewhere: bool = False) -> str:
    """The common category of a source label. Harness categories are kept as they are; a VM or ketai DATE becomes DOB
    when another gold says DOB or a DOB cue precedes it, YEAR when it is a bare 4-digit year. Unknown labels: OTHER_QI."""
    if source == "harness":
        return label if label in CATEGORY_RISK else "OTHER_QI"
    cat = {"vm": VM_LABELS, "ketai": KETAI_LABELS}.get(source, {}).get(str(label).upper(), "OTHER_QI")
    if cat == "DATE":
        if dob_elsewhere or (field_text and dob_context(field_text, text)):
            return "DOB"
        if re.fullmatch(r"\d{4}", text.strip()):
            return "YEAR"
    return cat


def concept_type(label: str) -> str:
    t = str(label).upper()
    return CONCEPT_TYPES.get(t, t)


def majority(votes: dict[str, str]) -> str:
    """Most frequent value among the sources' votes {source: value}; a tie goes to the source first in PRIORITY."""
    n = Counter(votes.values())
    rank = {s: i for i, s in enumerate(PRIORITY)}
    return min(votes.items(), key=lambda sv: (-n[sv[1]], rank.get(sv[0], len(rank)), sv[0]))[1]


def is_direct(cat: str) -> bool:
    return cat in DIRECT


def in_level(item: dict, level: str) -> bool:
    """Whether a gold item counts at a level. Items without `votes` (no merge metadata) count at every level."""
    if level == "union" or "votes" not in item:
        return True
    if level == "direct":
        return bool(item.get("direct"))
    return bool(item.get("direct")) or item["votes"] >= 2
