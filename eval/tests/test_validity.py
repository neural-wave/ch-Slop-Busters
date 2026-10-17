"""metrics.validity: same records, ids, order and keys, icd10_codes unchanged; `explanations` (a list) is the one extra key allowed."""

from __future__ import annotations

from eval import metrics as M


def test_valid_output(original, clean):
    v = M.validity(original, clean)
    assert v == {"valid_rate": 1.0, "records_changed": 1.0, "errors": []}


def test_identity_is_valid_but_unchanged(original):
    v = M.validity(original, [dict(r) for r in original])
    assert v["valid_rate"] == 1.0 and v["records_changed"] == 0.0 and v["errors"] == []


def test_missing_record(original, clean):
    v = M.validity(original, clean[:2])
    assert v["valid_rate"] == 0.6667
    assert "record count 2 != 3" in v["errors"]
    assert "encounter ids or their order differ from the input" in v["errors"]
    assert "e3: missing" in v["errors"]


def test_reordered_ids(original, clean):
    v = M.validity(original, [clean[1], clean[0], clean[2]])
    assert v["errors"] == ["encounter ids or their order differ from the input"]
    assert v["valid_rate"] == 1.0  # every record is still valid on its own; the order error is reported, not counted per record


def test_changed_icd10_codes(original, clean):
    clean[0]["icd10_codes"] = [{"code": "R05", "description": "Cough, unspecified"}]
    v = M.validity(original, clean)
    assert v["valid_rate"] == 0.6667 and v["errors"] == ["e1: icd10_codes changed"]


def test_reordered_icd10_codes_are_a_change(original, clean):
    codes = [{"code": "I10", "description": "Essential hypertension"}, {"code": "E11.9", "description": "Type 2 diabetes"}]
    original[1]["icd10_codes"], clean[1]["icd10_codes"] = codes, codes[::-1]
    assert M.validity(original, clean)["errors"] == ["e2: icd10_codes changed"]


def test_extra_key(original, clean):
    clean[2]["confidence"] = 0.9
    v = M.validity(original, clean)
    assert v["valid_rate"] == 0.6667 and v["errors"] == ["e3: keys ['confidence']"]


def test_missing_key(original, clean):
    del clean[2]["dataset"]
    assert M.validity(original, clean)["errors"] == ["e3: keys ['dataset'], dataset changed"]


def test_explanations_list_allowed(original, clean):
    clean[0]["explanations"] = [{"field": "note", "start": 0, "end": 9, "text": "[PATIENT]", "reason": "patient name"}]
    clean[1]["explanations"] = []
    assert M.validity(original, clean) == {"valid_rate": 1.0, "records_changed": 1.0, "errors": []}


def test_explanations_not_a_list_rejected(original, clean):
    clean[0]["explanations"] = {"reason": "patient name"}
    clean[1]["explanations"] = "removed the name"
    v = M.validity(original, clean)
    assert v["valid_rate"] == 0.3333
    assert v["errors"] == ["e1: explanations not a list", "e2: explanations not a list"]


def test_explanations_do_not_excuse_other_keys(original, clean):
    clean[0]["explanations"], clean[0]["spans"] = [], []
    assert M.validity(original, clean)["errors"] == ["e1: keys ['spans']"]


def test_text_fields_must_be_strings(original, clean):
    clean[1]["note"] = None
    assert M.validity(original, clean)["errors"] == ["e2: dialogue/note not strings"]
