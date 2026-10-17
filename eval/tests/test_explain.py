"""Explainability: token coverage of the changed text by decisions with a reason, on a hand-made record."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval import metrics as M
from eval import run as R
from eval.adapters import anonymous_vm
from eval.bonus import explain as X

DIALOGUE, NOTE = "Hello Andrew Campbell how are you", "Andrew Campbell is 59 and has a cough"
ORIG = [{"encounter_id": "x", "dataset": "toy", "dialogue": DIALOGUE, "note": NOTE, "icd10_codes": []}]
ANON = [{**ORIG[0], "dialogue": "Hello [PATIENT] how are you", "note": "[PATIENT] is [AGE] and has a cough"}]
# changed tokens: andrew, campbell (dialogue); andrew, campbell, 59 (note)
NAME_D = {"encounter_id": "x", "field": "dialogue", "start": 6, "end": 21, "text": "Andrew Campbell", "action": "remove", "category": "PATIENT_NAME",
          "reason": "patient name", "source": "names"}  # fmt: skip
NAME_N = {**NAME_D, "field": "note", "start": 0, "end": 15}
AGE_N = {"encounter_id": "x", "field": "note", "start": 19, "end": 21, "text": "59", "action": "generalize", "category": "AGE", "reason": "exact age",
         "source": "regex"}  # fmt: skip


def sidecar(tmp_path, rows) -> object:
    with (tmp_path / X.SIDECAR).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    return tmp_path


def test_align():
    toks = M.align("Hello Andrew Campbell how are you", "Hello [PATIENT] how are you")
    assert [(t, k) for t, _, _, k in toks] == [("hello", True), ("andrew", False), ("campbell", False), ("how", True), ("are", True), ("you", True)]
    assert toks[1][1:3] == (6, 12) and M.align("", "anything") == []


def test_full_coverage(tmp_path):
    x = X.score(sidecar(tmp_path, [NAME_D, NAME_N, AGE_N]), ORIG, ANON)
    assert x == {"explained": 1.0, "changed_tokens": 5, "covered_tokens": 5, "decisions": 3, "with_reason": 3, "unaligned": 0, "source": "sidecar"}


def test_partial_coverage(tmp_path):
    x = X.score(sidecar(tmp_path, [NAME_D, NAME_N]), ORIG, ANON)  # the age decision is missing
    assert x["explained"] == 0.8 and x["covered_tokens"] == 4 and x["changed_tokens"] == 5


def test_empty_reason_not_counted(tmp_path):
    x = X.score(sidecar(tmp_path, [NAME_D, NAME_N, {**AGE_N, "reason": "  "}]), ORIG, ANON)
    assert x["explained"] == 0.8 and x["decisions"] == 3 and x["with_reason"] == 2


def test_keep_decision_explains_no_change(tmp_path):
    x = X.score(sidecar(tmp_path, [NAME_D, NAME_N, {**AGE_N, "action": "keep"}]), ORIG, ANON)
    assert x["explained"] == 0.8 and x["with_reason"] == 3


def test_no_sidecar(tmp_path):
    assert X.score(tmp_path, ORIG, ANON) == {"explained": 0.0, "changed_tokens": 5, "covered_tokens": 0, "decisions": 0, "with_reason": 0,
                                             "unaligned": 0, "source": "none"}  # fmt: skip
    assert X.score(tmp_path, ORIG, ORIG)["explained"] is None  # identity: nothing changed, nothing to explain


def test_inline_explanations(tmp_path):
    rows = [{k: v for k, v in d.items() if k != "encounter_id"} for d in (NAME_D, NAME_N)]
    x = X.score(tmp_path, ORIG, [{**ANON[0], "explanations": rows}])
    assert x["source"] == "inline" and x["explained"] == 0.8 and x["decisions"] == 2


def test_sidecar_wins_over_inline(tmp_path):
    x = X.score(sidecar(tmp_path, [NAME_D]), ORIG, [{**ANON[0], "explanations": [NAME_N, AGE_N]}])
    assert x["source"] == "sidecar" and x["covered_tokens"] == 2


AD_ROW = {"encounter_id": "x", "field": "note", "start": 19, "end": 21, "category": "AGE", "action": "GENERALIZE", "replacement": "[50s]",
          "source": "rule:age_year_old", "confidence": 1.0, "detector_reason": "age expression", "policy_reason": "exact age is a quasi-identifier",
          "entity": None, "explanation": "GENERALIZE AGE -> [50s]: age expression [rule:age_year_old, conf 1.00]; exact age is a quasi-identifier"}  # fmt: skip


def test_ad_schema_normalization():
    d = X.normalize(AD_ROW)
    assert d == {"encounter_id": "x", "field": "note", "start": 19, "end": 21, "text": None, "action": "generalize", "category": "AGE",
                 "reason": AD_ROW["explanation"], "source": "rule:age_year_old"}  # fmt: skip
    assert X.place(d, ORIG[0]) == [("note", 19, 21)]  # no text: the offsets index the original
    assert X.normalize({**AD_ROW, "action": "REPLACE", "replacement": "[PATIENT]"})["action"] == "remove"
    assert X.normalize({**AD_ROW, "action": "REPLACE", "replacement": "Jane Doe"})["action"] == "perturb"  # a surrogate
    assert X.normalize({**AD_ROW, "action": "KEEP", "replacement": None})["action"] == "keep"
    assert X.normalize({**AD_ROW, "explanation": ""})["reason"] == "age expression; exact age is a quasi-identifier"


def test_ad_sidecar_scores(tmp_path):
    ad = [{**AD_ROW, "field": f, "start": s, "end": s + 15, "category": "PATIENT_NAME", "action": "REPLACE", "replacement": "[PATIENT]"}
          for f, s in (("dialogue", 6), ("note", 0))] + [AD_ROW]  # fmt: skip
    assert X.score(sidecar(tmp_path, ad), ORIG, ANON)["explained"] == 1.0


def test_rows_without_offsets_cover_every_occurrence():
    """Ketai's generalize log names the text and type, not the offsets."""
    d = X.normalize({"encounter_id": "x", "text": "andrew campbell", "type": "PERSON", "action": "removed", "reason": "a person name"})
    assert d["action"] == "remove" and d["category"] == "PERSON" and X.place(d, ORIG[0]) == [("dialogue", 6, 21), ("note", 0, 15)]


@pytest.mark.parametrize(("row", "spans"), [
    ({**AGE_N, "start": 18, "end": 20}, [("note", 19, 21)]),  # off by one: moved to the nearest occurrence of its text
    ({**AGE_N, "text": "60"}, []),                               # the text is not in the record
    ({**AGE_N, "start": 30, "end": 99, "text": None}, []),       # out of range, nothing to search for
])  # fmt: skip
def test_place_checks_the_original(row, spans):
    assert X.place(X.normalize(row), ORIG[0]) == spans


def test_unaligned_rows(tmp_path):
    rows = [NAME_D, NAME_N, AGE_N, "not json", {**AGE_N, "encounter_id": "nope"}, {**AGE_N, "text": "60"}]
    x = X.score(sidecar(tmp_path, rows), ORIG, ANON)
    assert x["explained"] == 1.0 and x["decisions"] == 6 and x["unaligned"] == 3


def test_sidecar_bad_encoding(tmp_path):
    good = [json.dumps(r).encode() for r in (NAME_D, NAME_N)]
    (tmp_path / X.SIDECAR).write_bytes(b"\xef\xbb\xbf" + b"\n".join([good[0], b"\xff\xfe", b'{"reason": "\xe9"}', good[1]]) + b"\n")
    x = X.score(tmp_path, ORIG, ANON)  # a BOM is ignored; a line that is not UTF-8 is one unaligned row, not a crash
    assert x["covered_tokens"] == 4 and x["decisions"] == 4 and x["unaligned"] == 2 and x["source"] == "sidecar"


# producers of decisions: the anonymous adapter and precomputed runs ------------------------------------------------------


def span(start, end, typ, tier, conf, source, reason=""):
    return SimpleNamespace(field="note", start=start, end=end, type=typ, tier=tier, confidence=conf, source=source, reason=reason)


def test_anonymous_decisions():
    members = [span(0, 6, "NAME_PATIENT", 1, 0.95, "names", "name 'andrew'"), span(0, 15, "NAME_PATIENT", 1, 0.99, "ner"),
               span(19, 21, "AGE", 1, 0.97, "regex", "regex: age")]  # fmt: skip
    merged = anonymous_vm.merge(members)
    assert [m[:4] for m in merged] == [(0, 15, "NAME_PATIENT", 1), (19, 21, "AGE", 1)] and merged[0][4] == [members[1], members[0]]
    d = anonymous_vm.decision("x", "note", NOTE, merged[0], True)
    assert d == {"encounter_id": "x", "field": "note", "start": 0, "end": 15, "text": "Andrew Campbell", "action": "remove", "category": "NAME_PATIENT",
                 "reason": "tier 1 NAME_PATIENT -> [NAME_PATIENT]: ner found NAME_PATIENT [ner, conf 0.99]; name 'andrew' [names, conf 0.95]",
                 "source": "names+ner"}  # fmt: skip
    assert anonymous_vm.decision("x", "note", NOTE, merged[1], False)["action"] == "keep"


def test_precomputed_sidecar_is_copied(tmp_path, monkeypatch):
    (tmp_path / "suites" / "toy").mkdir(parents=True)
    (tmp_path / "suites" / "toy" / "encounters.jsonl").write_text('{"encounter_id": "a"}\n')
    monkeypatch.setattr(R, "SUITES_DIR", tmp_path / "suites")
    monkeypatch.setattr(R, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "out.jsonl").write_text('{"encounter_id": "a"}\n')
    side, copied = tmp_path / "out.decisions.jsonl", tmp_path / "runs" / "m" / "toy" / X.SIDECAR
    side.write_text('{"encounter_id": "a", "reason": "one"}\n')
    spec = {"precomputed": str(tmp_path / "out.jsonl")}
    first = R.run_one("m", spec, "toy")
    assert first["exit_code"] == 0 and copied.read_text() == side.read_text() and R.run_one("m", spec, "toy") == first
    side.write_text('{"encounter_id": "a", "reason": "two"}\n')  # a new sidecar invalidates the cached run
    assert R.run_one("m", spec, "toy") != first and "two" in copied.read_text()
    side.unlink()
    assert R.run_one("m", spec, "toy")["exit_code"] == 0 and not copied.exists()  # no stale sidecar left behind
