"""bonus/phi_privacy.py: chunking, the detector's dict answers, the label map, and before/after counts with a fake detector."""

from __future__ import annotations

import json

import pytest

from eval.bonus import phi_privacy as P
from eval.tests.conftest import CLEAN, ORIGINAL


def test_chunks_cut_at_line_breaks_and_long_lines_at_spaces():
    assert P.chunks("a\nb\nc", limit=3) == ["a\nb", "c"]
    assert P.chunks("one two three", limit=8) == ["one two ", "three"]
    assert P.chunks("\n\n") == []
    text = "\n".join(f"line {i}" for i in range(500))
    assert "\n".join(P.chunks(text, limit=100)) == text and all(len(c) <= 100 for c in P.chunks(text, limit=100))


@pytest.mark.parametrize(("answer", "parsed"), [
    ("{'firstname': ['Abner'], 'dob': ['23/03/1926']} trailing", {"firstname": ["Abner"], "dob": ["23/03/1926"]}),
    ('{"City": "Boston", "age": [59]}', {"city": ["Boston"], "age": ["59"]}),
    ("{}", {}), ("", {}),
])  # fmt: skip
def test_parse(answer, parsed):
    assert P.parse(answer) == parsed


def test_parse_rejects_garbage():
    for bad in ("no braces here", "{'a': [1", "{1, 2}"):
        with pytest.raises((ValueError, SyntaxError)):
            P.parse(bad)


def test_mentions_map_labels_drop_non_identifying_and_placeholders():
    got = P.mentions({"firstname": ["Wei", "[PATIENT]", "Martha"], "lastname": ["Li", "martha"], "sex": ["female"], "city": ["Boston"], "unknownlabel": ["x"]})
    assert got == {"PERSON_NAME": 3, "LOCATION": 1}  # Martha as first and last name counts once


def fake_detector(model: str, prompt: str) -> str:
    text = prompt.split("### Input:\n    ", 1)[1]
    found = {"firstname": [w for w in ("Andrew", "Wei", "Martha") if w in text], "city": ["Boston"] * text.count("Boston")}
    return json.dumps(found)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    for mod_attr, path in (("RUNS_DIR", tmp_path / "runs"), ("SUITES_DIR", tmp_path / "suites"), ("PCACHE", tmp_path / "cache"), ("PHI_DIR", tmp_path / "phi")):
        monkeypatch.setattr(P, mod_attr, path)
    (tmp_path / "suites" / "toy").mkdir(parents=True)
    (tmp_path / "suites" / "toy" / "encounters.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ORIGINAL))
    for model, recs in (("identity", ORIGINAL), ("clean", CLEAN)):
        d = tmp_path / "runs" / model / "toy"
        d.mkdir(parents=True)
        (d / "output.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (d / "run.json").write_text(json.dumps({"exit_code": 0}))
    return tmp_path


def test_before_after_counts(harness):
    scanner = P.Scanner("phi", "d1", fake_detector)
    ident = P.evaluate(scanner, "identity", "toy")
    assert ident["total"]["before"] == ident["total"]["after"] and ident["total"]["removed"] == 0.0
    # names: Andrew (dialogue, note), Wei (dialogue, note), Martha (dialogue, note); Boston: e2 note, e3 dialogue (twice, one mention)
    assert ident["by_category"] == {"LOCATION": {"before": 2, "after": 2, "removed": 0.0}, "PERSON_NAME": {"before": 6, "after": 6, "removed": 0.0}}
    calls = scanner.calls
    clean = P.evaluate(scanner, "clean", "toy")
    assert clean["total"] == {"before": 8, "after": 0, "removed": 1.0} and clean["new_calls"] == scanner.calls - calls
    assert P.evaluate(P.Scanner("phi", "d1", lambda m, p: "garbage"), "clean", "toy")["new_calls"] == 0  # the originals come from the cache
    broken = P.evaluate(P.Scanner("phi", "d2", lambda m, p: "garbage"), "clean", "toy")
    assert broken["errors"] == 12 and broken["total"]["before"] == 0 and broken["total"]["removed"] is None  # 6 fields before, 6 after
    assert P.evaluate(scanner, "missing", "toy") == "no output"


def test_truncated_answers_are_salvaged_and_failed_chunks_not_asked_twice(harness):
    loop = "\n{'firstname': ['Andrew'], 'city': ['Boston'], 'firstname': ['Andrew', 'Wei'], 'jobtype': ['doc"  # cut by the output limit
    assert P.salvage(loop) == {"firstname": ["Andrew", "Wei"], "city": ["Boston"]}
    with pytest.raises(ValueError):
        P.salvage("{'jobtype': ['doc")
    cut = P.evaluate(P.Scanner("phi", "d3", lambda m, p: fake_detector(m, p)[:-1]), "identity", "toy")  # every answer loses its closing brace
    assert cut["salvaged"] == 6 and cut["errors"] == 0 and cut["total"]["before"] == cut["total"]["after"] == 8
    asked: list[str] = []

    def flaky(model: str, prompt: str) -> str:  # unreadable the first time a chunk is asked, fine the next time
        asked.append(prompt)
        return "garbage" if asked.count(prompt) == 1 else fake_detector(model, prompt)

    ident = P.evaluate(P.Scanner("phi", "d4", flaky), "identity", "toy")
    assert len(asked) == 6 and ident["errors"] == 12 and ident["total"]["before"] == ident["total"]["after"] == 0  # same count before and after
