"""coders.codes: canonical ICD-10-CM spelling (ported from daniel-eval's normalization tests), and gold codes."""

from __future__ import annotations

import pytest

from eval import icd
from eval.coders.codes import category, count_invalid, normalize, normalize_set


@pytest.mark.parametrize(("raw", "code"), [
    (" i110 ", "I11.0"), ("I11.0", "I11.0"), ("I10", "I10"), ("e119", "E11.9"), ("G43909", "G43.909"), ("G43.909", "G43.909"),
    ("S93.401A", "S93.401A"), ("F32.A", "F32.A"), ("(E11.9),", "E11.9"), ("E 11.9", "E11.9"), ({"code": "I11.9"}, "I11.9"),
    ({"code": "i10", "description": "Essential hypertension"}, "I10"),
])  # fmt: skip
def test_valid_codes_get_one_spelling(raw, code):
    assert normalize(raw) == code


@pytest.mark.parametrize("raw", ["I10 hypertension", "I.10", "made up", "11.9", "E11.12345", "E1", "", ".", None, 10, {"code": 5}, {"description": "x"}, ["I10"]])
def test_invalid_codes_are_none(raw):
    assert normalize(raw) is None


def test_normalization_is_exact_not_hierarchical():
    assert normalize_set([" i110 ", "I11.0", {"code": "I11.9"}]) == ["I11.0", "I11.9"]
    assert normalize("I11.0") != normalize("I11.9")


def test_set_keeps_first_seen_order_and_drops_invalid():
    assert normalize_set(["Z00.00", "junk", "I10", "z0000", "E11.9"]) == ["Z00.00", "I10", "E11.9"]
    assert count_invalid(["Z00.00", "junk", None, "I10"]) == 2


def test_a_non_list_is_no_codes():
    assert normalize_set("I10") == [] and count_invalid("I10") == 1
    assert normalize_set(None) == [] and count_invalid(None) == 1
    assert normalize_set(()) == [] and count_invalid([]) == 0


def test_category():
    assert category("E11.9") == "E11" and category("I10") == "I10"


def test_gold_codes_undotted_dict_and_invalid():
    rec = {"icd10_codes": [{"code": "I10"}, {"code": "e119"}, {"code": "E11.9"}, {"code": "Z09"}, {"code": "bad code"}, "J45.909"]}
    assert icd.gold_codes(rec) == ["I10", "E11.9", "Z09", "BAD CODE", "J45.909"]  # an invalid gold code still counts as a miss
    assert icd.gold_codes({"icd10_codes": []}) == [] and icd.gold_codes({}) == []
