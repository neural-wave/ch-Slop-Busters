"""Shared fixtures: a hand-made 3-record suite with known answers, and word lists that never touch data/ or models/."""

from __future__ import annotations

import copy

import pytest

from eval import metrics as M
from eval.common import CATEGORY_RISK

COMMON_WORDS = frozenset({"supermarket", "classes"})
VOCABULARY = frozenset({"cough", "chest", "pain", "campbell"})


@pytest.fixture(autouse=True)
def word_lists(monkeypatch):
    """The privacy token rule and the terms metric read word lists from models/ and data/; tests use fixed ones."""
    monkeypatch.setattr(M, "common_words", lambda: (COMMON_WORDS, None))
    monkeypatch.setattr(M, "clinical_vocabulary", lambda: VOCABULARY)


def rec(eid: str, dialogue: str, note: str, codes: list[dict] | None = None) -> dict:
    return {"encounter_id": eid, "dataset": "toy", "dialogue": dialogue, "note": note, "icd10_codes": codes or []}


def item(field: str, text: str, category: str, **kw) -> dict:
    return {"field": field, "text": text, "category": category, "risk": CATEGORY_RISK.get(category, "high"), **kw}


COUGH, HYPERTENSION = [{"code": "R05.9", "description": "Cough, unspecified"}], [{"code": "I10", "description": "Essential hypertension"}]

ORIGINAL = [
    rec("e1", "Doctor: Good morning, Andrew Campbell. Patient: Morning.", "Andrew Campbell, 59 years old. BP 120/59. Cough and chest pain.", COUGH),
    rec("e2", "Patient: My name is Wei Li. Doctor: Hello Wei Li.", "Wei Li works in Boston.", HYPERTENSION),
    rec("e3", "Doctor: Martha, how is Boston? Patient: Boston is cold.", "Martha reports chest pain. History of Parkinson's disease.", []),
]  # fmt: skip

LABELS = {
    "e1": {"encounter_id": "e1", "pii_gold": "exhaustive", "language": "en",
           "pii": [item("dialogue", "Andrew Campbell", "PATIENT_NAME"), item("note", "Andrew Campbell", "PATIENT_NAME"), item("note", "59", "AGE")],
           "keep": [item("note", "120/59", "CLINICAL_NUMBER")],
           "concepts": [{"field": "note", "text": "BP", "type": "TEST", "icd_relevant": False}]},
    "e2": {"encounter_id": "e2", "pii_gold": "exhaustive", "language": "en",
           "pii": [item("dialogue", "Wei Li", "PATIENT_NAME"), item("note", "Wei Li", "PATIENT_NAME"), item("note", "Boston", "LOCATION")], "keep": []},
    "e3": {"encounter_id": "e3", "pii_gold": "exhaustive", "language": "en",
           "pii": [item("dialogue", "Martha", "PATIENT_NAME"), item("dialogue", "Boston", "LOCATION"), item("note", "Martha", "PATIENT_NAME")],
           "keep": [item("note", "Parkinson's", "EPONYM")],
           "concepts": [{"field": "note", "text": "chest pain", "type": "SYMPTOM", "icd_relevant": True},
                        {"field": "note", "text": "Parkinson's disease", "type": "DIAGNOSIS", "icd_relevant": True}]},
}  # fmt: skip

# Every gold PII occurrence replaced: 11 occurrences of 9 items (Wei Li and Boston appear twice in e2/e3 dialogue).
CLEAN = [
    rec("e1", "Doctor: Good morning, [PATIENT]. Patient: Morning.", "[PATIENT], [AGE] years old. BP 120/59. Cough and chest pain.", COUGH),
    rec("e2", "Patient: My name is [PATIENT]. Doctor: Hello [PATIENT].", "[PATIENT] works in [LOCATION].", HYPERTENSION),
    rec("e3", "Doctor: [PATIENT], how is [LOCATION]? Patient: [LOCATION] is cold.", "[PATIENT] reports chest pain. History of Parkinson's disease.", []),
]  # fmt: skip


@pytest.fixture
def original() -> list[dict]:
    return copy.deepcopy(ORIGINAL)


@pytest.fixture
def labels() -> dict[str, dict]:
    return copy.deepcopy(LABELS)


@pytest.fixture
def clean() -> list[dict]:
    return copy.deepcopy(CLEAN)
