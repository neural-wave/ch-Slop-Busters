"""Privacy, keep/concept survival, edits, terms and placeholder rate on the hand-made 3-record suite (conftest.py)."""

from __future__ import annotations

import pytest

from eval import metrics as M
from eval.common import count


def privacy(original, anonymized, labels):
    return M.privacy(original, anonymized, labels)


# privacy -----------------------------------------------------------------------------------------------------------


def test_full_removal(original, clean, labels):
    p = privacy(original, clean, labels)
    assert p["removed"] == 1.0 and p["removed_high_risk"] == 1.0 and p["leaks"] == []
    assert p["counts"]["ALL"] == [11, 11] and p["counts"]["items_fully_removed"] == [9, 9]
    assert p["counts"]["risk:high"] == [8, 8] and p["counts"]["cat:LOCATION"] == [3, 3]
    assert p["n_encounters"] == 3 and p["token_wordlist"] is None


def test_identity_removes_nothing(original, labels):
    p = privacy(original, original, labels)
    assert p["removed"] == 0.0 and p["counts"]["ALL"] == [0, 11] and len(p["leaks"]) == 9


def test_partial_leak(original, clean, labels):
    clean[2]["dialogue"] = "Doctor: [PATIENT], how is [LOCATION]? Patient: Boston is cold."  # one of two `Boston` left
    p = privacy(original, clean, labels)
    assert p["removed"] == 0.9091  # 10 of 11 occurrences
    assert p["counts"]["items_fully_removed"] == [8, 9]
    assert p["breakdown"]["cat:LOCATION"] == 0.6667 and p["removed_high_risk"] == 1.0
    assert p["leaks"] == [{"encounter_id": "e3", "field": "dialogue", "text": "Boston", "category": "LOCATION", "risk": "medium", "left": 1, "of": 2}]


def test_distinctive_token_leaks_the_name(original, clean, labels):
    """`[PATIENT] Campbell` leaks `Andrew Campbell`: the surname alone re-identifies."""
    clean[0]["dialogue"] = "Doctor: Good morning, [PATIENT] Campbell. Patient: Morning."
    p = privacy(original, clean, labels)
    assert p["removed"] == 0.9091
    assert [(x["text"], x["field"], x["left"]) for x in p["leaks"]] == [("Andrew Campbell", "dialogue", 1)]
    assert M.leaked("Andrew Campbell", "PATIENT_NAME", original[0]["dialogue"], clean[0]["dialogue"], 1) == 1


def test_short_tokens_are_not_distinctive(original, clean, labels):
    """Documented blind spot: tokens under 4 characters are never distinctive, so `[PATIENT] Li` does NOT leak `Wei Li`."""
    clean[1]["dialogue"] = "Patient: My name is [PATIENT] Li. Doctor: Hello [PATIENT] Li."
    assert M._distinctive("Wei Li", "PATIENT_NAME") == []
    p = privacy(original, clean, labels)
    assert p["removed"] == 1.0 and p["leaks"] == []


def test_number_boundary():
    """`59` is not matched inside `120/59` (nor `5.9`, `1,590`, `59.5`), so a kept blood pressure is no age leak."""
    assert count("Andrew Campbell, 59 years old. BP 120/59.", "59") == 1
    assert [count(t, "59") for t in ("BP 120/59", "5.9 mg", "1,590", "59.5 kg", "59-year-old", "age 59.")] == [0, 0, 0, 0, 1, 1]


def test_number_boundary_in_privacy(original, clean, labels):
    p = privacy(original, clean, labels)  # the kept `120/59` does not count as the age
    assert p["breakdown"]["cat:AGE"] == 1.0 and p["counts"]["cat:AGE"] == [1, 1]
    clean[0]["note"] = "[PATIENT], 59 years old. BP 120/59. Cough and chest pain."
    p = privacy(original, clean, labels)
    assert p["counts"]["cat:AGE"] == [0, 1] and p["leaks"][0]["left"] == 1


def test_common_words_of_other_items_are_not_distinctive():
    orig = "She works as a cashier in a supermarket."
    assert M.leaked("cashier in a supermarket", "OCCUPATION", orig, "She works as [OCCUPATION] in a supermarket.", 1) == 0
    assert M.leaked("cashier in a supermarket", "OCCUPATION", orig, "She works as a cashier in a [PLACE].", 1) == 1


def test_token_also_outside_the_item_is_not_a_leak():
    """`Campbell` that the original also has on its own (another relative) is not attributed to `Andrew Campbell`."""
    orig, anon = "Andrew Campbell came with Mrs Campbell.", "[PATIENT] came with Mrs Campbell."
    assert M.leaked("Andrew Campbell", "PATIENT_NAME", orig, anon, 1) == 0


def test_no_gold_for_key(original, clean, labels):
    assert M.privacy(original, clean, labels, "pii_alt") is None


# keep and concept survival -----------------------------------------------------------------------------------------

LOSSY_E3_NOTE = "[PATIENT] reports [SYMPTOM]. History of [PERSON] disease."


def test_keep_survives(original, clean, labels):
    k = M.survival(original, clean, labels, "keep", "category")
    assert k["kept"] == 1.0 and k["counts"]["ALL"] == [2, 2] and k["top_lost"] == []


def test_keep_lost(original, clean, labels):
    clean[2]["note"] = LOSSY_E3_NOTE
    k = M.survival(original, clean, labels, "keep", "category")
    assert k["kept"] == 0.5
    assert k["breakdown"] == {"ALL": 0.5, "category:CLINICAL_NUMBER": 1.0, "category:EPONYM": 0.0}
    assert k["top_lost"] == [("EPONYM: parkinson's", 1)]


def test_concepts(original, clean, labels):
    assert M.survival(original, clean, labels, "concepts", "type")["kept"] == 1.0
    clean[2]["note"] = LOSSY_E3_NOTE
    c = M.survival(original, clean, labels, "concepts", "type")
    assert c["kept"] == 0.3333 and c["breakdown"]["icd_relevant"] == 0.0 and c["breakdown"]["type:TEST"] == 1.0
    assert c["counts"]["icd_relevant"] == [0, 2]


def test_survival_without_gold(original, clean, labels):
    for g in labels.values():
        g.pop("concepts", None)
    assert M.survival(original, clean, labels, "concepts", "type") is None


# edits ---------------------------------------------------------------------------------------------------------------

EDIT_ORIG = [{"encounter_id": "x", "dataset": "toy", "dialogue": "Hello Andrew Campbell how are you", "note": "Andrew Campbell has a cough", "icd10_codes": []}]
EDIT_GOLD = {"x": {"encounter_id": "x", "pii_gold": "exhaustive",
                   "pii": [{"field": f, "text": "Andrew Campbell", "category": "PATIENT_NAME"} for f in ("dialogue", "note")]}}  # fmt: skip


def edited(dialogue: str, note: str) -> list[dict]:
    return [{**EDIT_ORIG[0], "dialogue": dialogue, "note": note}]


def test_edits_exact():
    e = M.edits(EDIT_ORIG, edited("Hello [PATIENT] how are you", "[PATIENT] has a cough"), EDIT_GOLD)
    assert e == {"non_pii_tokens_kept": 1.0, "pii_tokens_changed": 1.0, "edit_precision": 1.0, "tokens_changed": 4}


def test_edits_over_redaction():
    e = M.edits(EDIT_ORIG, edited("Hello [PATIENT] how are you", "[PATIENT] has a [SYMPTOM]"), EDIT_GOLD)
    assert e == {"non_pii_tokens_kept": 0.8571, "pii_tokens_changed": 1.0, "edit_precision": 0.8, "tokens_changed": 5}


def test_edits_under_redaction():
    e = M.edits(EDIT_ORIG, edited("Hello [PATIENT] Campbell how are you", "[PATIENT] has a cough"), EDIT_GOLD)
    assert e["pii_tokens_changed"] == 0.75 and e["edit_precision"] == 1.0


def test_edits_need_exhaustive_gold():
    gold = {"x": {**EDIT_GOLD["x"], "pii_gold": "partial"}}
    assert M.edits(EDIT_ORIG, edited("Hello [PATIENT] how are you", "[PATIENT] has a cough"), gold) is None


# terms ---------------------------------------------------------------------------------------------------------------


def test_terms_kept(original, clean, labels):
    """Vocabulary {cough, chest, pain, campbell}: `campbell` sits inside gold PII, so removing it is not lost medicine."""
    t = M.terms(original, clean, labels)
    assert t == {"kept": 1.0, "n_terms": 5, "top_lost": []}


def test_terms_lost(original, clean, labels):
    clean[2]["note"] = LOSSY_E3_NOTE
    t = M.terms(original, clean, labels)
    assert t["kept"] == 0.6 and t["n_terms"] == 5 and sorted(t["top_lost"]) == [("chest", 1), ("pain", 1)]


def test_terms_without_vocabulary(monkeypatch, original, clean, labels):
    monkeypatch.setattr(M, "clinical_vocabulary", lambda: frozenset())
    assert M.terms(original, clean, labels) is None


# placeholders --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "rate"), [("[PATIENT] said hi", 333.33), ("no tags here", 0.0), ("", None)])
def test_placeholder_rate(text, rate):
    assert M.placeholder_rate([{"dialogue": text, "note": ""}]) == rate
