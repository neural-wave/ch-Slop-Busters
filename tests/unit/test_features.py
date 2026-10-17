"""M5 feature flags F0 (code-evidence consistency) and F2 (targeted LLM detector), without models."""

from __future__ import annotations

import pytest

from unified_harness.cli import _features
from unified_harness.detect import encoders
from unified_harness.shield import medical
from unified_harness.text.normalize import Doc

DIALOGUE = (
    "[patient] i hurt my wrist when i was skiing last weekend .\n"
    "[doctor] do you ski a lot ?\n"
    "[patient] yes , skiing is my thing . i also love pottery and my back hurts .\n"
    "[doctor] okay . your blood pressure is fine ."
)


def _activity(shields, text):
    return sorted(text[s.start : s.end] for s in shields if s.kind == "activity_context")


def test_f0_needs_a_mechanism_verb_in_the_sentence():
    doc = Doc("dialogue", DIALOGUE)
    window = medical.activity_context(doc)
    sentence = medical.activity_context(doc, sentence_level=True)
    assert "pottery" in _activity(window, DIALOGUE)  # an injury word nearby is enough for the window rule
    assert "pottery" not in _activity(sentence, DIALOGUE)  # "hurts" (chronic pain) is not a mechanism
    assert _activity(sentence, DIALOGUE) == ["skiing"]


def test_f0_keeps_every_mention_of_an_evidence_activity():
    docs = {"dialogue": Doc("dialogue", DIALOGUE), "note": Doc("note", "Patient enjoys skiing and pottery.")}
    base = medical.activity_context(docs["dialogue"], sentence_level=True)
    extra = medical.evidence_consistency(docs, base)
    assert sorted((s.field, docs[s.field].raw[s.start : s.end]) for s in extra) == [
        ("dialogue", "skiing"),
        ("note", "skiing"),
    ]


def test_f2_segments_keep_personal_context_only():
    doc = Doc("note", "Pain is 6/10. She works as a florist in town. Blood pressure 140/90. Continue metformin.")
    picked = [doc.raw[s:e].strip() for s, e in encoders.personal_segments(doc)]
    assert picked == ["She works as a florist in town."]
    turns = Doc("dialogue", DIALOGUE)
    first = encoders.personal_segments(turns)[0]
    assert turns.raw[first[0] : first[1]].startswith("[patient] i hurt my wrist")  # opening turns


def _bare_llm(tmp_path, reply):
    det = object.__new__(encoders.TargetedLLM)
    det.model, det.digest, det.cache, det.calls, det.hits = "m", "d", {}, 0, 0
    det.cache_path = tmp_path / "replay.jsonl"
    det._call = lambda prompt, num_predict=600: reply
    return det


def test_f2_replay_cache_and_fail_closed(tmp_path):
    det = _bare_llm(tmp_path, '{"jobs": ["florist"], "names": [], "places": [], "orgs": []}')
    doc = Doc("note", "She works as a florist in town.")
    spans = det.detect(doc)
    assert [(doc.raw[s.start : s.end], s.cat) for s in spans] == [("florist", "OCCUPATION")]
    det._call = lambda prompt, num_predict=600: pytest.fail("cached segment must not call the model")
    det.detect(doc)
    assert (det.calls, det.hits) == (1, 1)
    assert "florist" in det.cache_path.read_text()
    broken = _bare_llm(tmp_path / "b", '{"jobs": ["flor')
    with pytest.raises(RuntimeError, match="invalid JSON"):
        broken.detect(Doc("note", "He works as a baker."))


def test_f2_drops_staff_roles(tmp_path):
    det = _bare_llm(tmp_path, "")
    det._answer = lambda chunk: {"jobs": ["medical assistant", "nurse"]}
    assert det.detect(Doc("note", "My daughter is a nurse and the medical assistant saw me.")) == []


def test_unknown_feature_is_an_error():
    assert _features("f0, F2") == {"F0", "F2"}
    with pytest.raises(SystemExit):
        _features("F9")
