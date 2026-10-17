"""eval/icd.py: the score shape, ranked cut at k, unranked sets, error counting, coder-level failure, the cache (and its
compatibility with the plain lists of scripts/eval), and the coders.toml registry."""

from __future__ import annotations

import argparse
import json

import pytest

from eval import icd
from eval import score as SC
from eval.coders import ollama
from eval.coders.base import Coder, CoderError, Prediction

LEVEL = {"precision", "recall", "f1", "mean_f1", "ci95"}
KEYS = ["coder", "config", "ranked", "inputs", "n_records", "n_codes", "original", "anonymized", "agreement", "f1_delta", "f1_retention", "errors",
        "invalid_codes", "valid_for_comparison", "changed_predictions"]  # fmt: skip


class Fake(Coder):
    """Answers from a table keyed by the note text; records every record it is asked to predict."""

    def __init__(self, answers: dict, ranked: bool = True, name: str = "fake", inputs: str = "note", fail_on: str | None = None):
        self.answers, self.ranked, self.name, self.inputs, self.fail_on = answers, ranked, name, inputs, fail_on
        self.config, self.asked = f"fake:{ranked}", []

    def predict(self, records, on_result=None):
        out = {}
        for r in records:
            if r["note"] == self.fail_on:
                raise CoderError("server went away")
            self.asked.append(r["note"])
            a = self.answers[r["note"]]
            p = a if isinstance(a, Prediction) else Prediction(list(a))
            out[r["encounter_id"]] = p
            if on_result:
                on_result(r, p)
        return out


def rec(eid: str, note: str, codes: list[str]) -> dict:
    return {"encounter_id": eid, "dataset": "toy", "dialogue": "", "note": note, "icd10_codes": [{"code": c, "description": c} for c in codes]}


ORIGINAL = [rec("a", "hypertension", ["I10"]), rec("b", "diabetes and asthma", ["E11.9", "J45.909"]), rec("c", "no codes", [])]
ANONYMIZED = [rec("a", "hypertension", ["I10"]), rec("b", "[NAME] diabetes", ["E11.9", "J45.909"]), rec("c", "no codes", [])]


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(icd, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def test_shape_and_ranked_cut_at_k():
    coder = Fake({"hypertension": ["I10", "E11.9", "R05.9"], "diabetes and asthma": ["E11.9", "J45.909", "I10"], "[NAME] diabetes": ["E11.9", "I10", "J45.909"]})
    res = icd.score(coder, ORIGINAL, ANONYMIZED, resamples=50)
    assert list(res) == KEYS
    assert res["n_records"] == 2 and res["n_codes"] == 3  # only records with gold codes
    for arm in ("original", "anonymized", "agreement"):
        assert set(res[arm]) == {"code", "category"} and set(res[arm]["code"]) == LEVEL
    assert res["original"]["code"]["f1"] == 1.0  # cut at k: I10 | E11.9, J45.909
    assert res["anonymized"]["code"] == {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667, "mean_f1": 0.75, "ci95": res["anonymized"]["code"]["ci95"]}
    assert res["agreement"]["code"]["f1"] == 0.6667  # anonymized vs original predictions as gold
    assert res["anonymized"]["category"]["f1"] == 0.6667
    assert res["f1_delta"]["value"] == -0.3333 and res["f1_retention"] == 0.6667
    assert res["changed_predictions"] == [{"encounter_id": "b", "gold": ["E11.9", "J45.909"], "original": ["E11.9", "J45.909"], "anonymized": ["E11.9", "I10"]}]
    assert res["errors"] == {"original": 0, "anonymized": 0} and res["valid_for_comparison"] and res["invalid_codes"] == {"original": 0, "anonymized": 0}


def test_unranked_set_is_scored_whole_and_category_level():
    coder = Fake({"hypertension": ["I10", "I11.9"], "diabetes and asthma": ["E11.65", "J45.909"], "[NAME] diabetes": ["E11.65", "J45.909"]}, ranked=False)
    res = icd.score(coder, ORIGINAL, ANONYMIZED, resamples=50)
    assert res["original"]["code"]["precision"] == 0.5 and res["original"]["code"]["recall"] == 0.6667
    assert res["original"]["category"]["f1"] == 0.8571  # I10, I11 | E11, J45 vs I10 | E11, J45: tp 3, fp 1
    assert not res["ranked"] and res["inputs"] == "note"


def test_identical_arms_delta_zero():
    coder = Fake({"hypertension": ["I10"], "diabetes and asthma": ["E11.9"]})
    res = icd.score(coder, ORIGINAL, ORIGINAL, resamples=50)
    assert res["f1_delta"] == {"value": 0.0, "ci95": [0.0, 0.0]} and res["f1_retention"] == 1.0
    assert res["agreement"]["code"]["f1"] == 1.0 and res["changed_predictions"] == []


def test_failed_predictions_count_as_empty_and_are_not_cached(cache):
    answers = {"hypertension": ["I10"], "diabetes and asthma": ["E11.9", "J45.909"], "[NAME] diabetes": Prediction([], "timeout", 2)}
    coder = Fake(answers, ranked=False)
    res = icd.score(coder, ORIGINAL, ANONYMIZED, resamples=50)
    assert res["errors"] == {"original": 0, "anonymized": 1} and not res["valid_for_comparison"]
    assert res["anonymized"]["code"]["recall"] == 0.3333
    assert len(list((cache / "icd" / "fake").glob("*.json"))) == 2  # the failure is not cached
    coder.answers["[NAME] diabetes"] = ["E11.9"]
    coder.asked.clear()
    res = icd.score(coder, ORIGINAL, ANONYMIZED, resamples=50)
    assert coder.asked == ["[NAME] diabetes"] and res["valid_for_comparison"]  # only the failed one is asked again


def test_invalid_codes_are_counted():
    coder = Fake({"hypertension": ["I10", "not a code"], "diabetes and asthma": ["E11.9"], "[NAME] diabetes": ["E11.9"]}, ranked=False)
    assert icd.score(coder, ORIGINAL, ANONYMIZED, resamples=10)["invalid_codes"] == {"original": 1, "anonymized": 1}


def test_coder_level_failure_gives_an_error_entry():
    coder = Fake({"hypertension": ["I10"]}, fail_on="diabetes and asthma")
    assert icd.score(coder, ORIGINAL, ANONYMIZED) == {"error": "server went away"}


def test_no_record_with_gold_codes():
    assert icd.score(Fake({}), [ORIGINAL[2]], [ANONYMIZED[2]]) is None


def test_cache_reads_old_plain_lists_and_writes_the_new_format(cache):
    coder = Fake({"hypertension": ["E11.9"], "diabetes and asthma": ["E11.9", "J45.909"]}, inputs="both")
    old = icd._cache_path(coder, ORIGINAL[0])
    assert old.name == f"{icd.sha1_text(coder.config, 'a', '', 'hypertension')}.json"  # scripts/eval key: config, id, dialogue, note
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(["I10", "R05.9"]))
    out = icd.cached_predict(coder, ORIGINAL[:2])
    assert out["a"].codes == ["I10", "R05.9"] and coder.asked == ["diabetes and asthma"]
    new = json.loads(icd._cache_path(coder, ORIGINAL[1]).read_text())
    assert new == {"codes": ["E11.9", "J45.909"], "attempts": 1, "seconds": 0.0, "usage": {}, "raw_invalid": 0}
    note_only = Fake({}, inputs="note")
    assert icd._cache_path(note_only, ORIGINAL[0]).name == f"{icd.sha1_text(note_only.config, 'a', 'hypertension')}.json"


def test_a_corrupt_cache_file_is_recomputed(cache):
    coder = Fake({"hypertension": ["I10"]})
    path = icd._cache_path(coder, ORIGINAL[0])
    path.parent.mkdir(parents=True)
    path.write_text("{trunc")
    assert icd.cached_predict(coder, ORIGINAL[:1])["a"].codes == ["I10"] and coder.asked == ["hypertension"]


def test_registry_defaults_and_suites():
    reg = icd.load_registry()
    assert icd.default_names(reg) == ["llm_note", "knn", "lexical"]  # llm_both is opt-in
    assert icd.primary(reg) == "llm_note" and not reg["paper"].get("enabled", True)
    assert "aci_eval" in reg["llm_note"]["suites"] and "aci_eval" in reg["llm_both"]["suites"]
    assert icd.applies("knn", "aci_dev", reg) and not icd.applies("knn", "devp_discharge", reg) and icd.applies("lexical", "pii_gpt", reg)
    assert icd.applies("cmd:x {input} {output}", "pii_gpt", reg)
    assert SC.DEFAULT_ICD == ["llm_note", "knn", "lexical"]
    with pytest.raises(argparse.ArgumentTypeError):
        SC.coder_name("nope")
    assert SC.coder_name("cmd:x") == "cmd:x" and SC.coder_name("knn") == "knn"


def test_registry_rejects_unknown_kind(tmp_path):
    p = tmp_path / "coders.toml"
    p.write_text('[x]\nkind = "magic"\n')
    with pytest.raises(ValueError, match="kind"):
        icd.load_registry(p)


def test_evaluate_with_ollama_down(monkeypatch):
    def down(*a, **kw):
        raise OSError("refused")

    monkeypatch.setattr(ollama, "request", down)
    reg = {"llm": {"kind": "ollama", "suites": ["toy"]}}
    assert icd.evaluate("llm", "toy", ORIGINAL, ANONYMIZED, reg) == ("llm", {"error": f"Ollama at {ollama.URL}: unreachable"})
    assert icd.evaluate("llm", "other", ORIGINAL, ANONYMIZED, reg) is None  # not one of its suites
    assert icd.evaluate("llm", "toy", ORIGINAL[2:], ANONYMIZED[2:], reg) is None  # no gold codes: nothing to score


def test_evaluate_llm_coder_end_to_end(monkeypatch, cache):
    calls = []

    def fake(path, body=None, timeout=180):
        if path == "/api/tags":
            return {"models": [{"name": ollama.DEFAULT_MODEL, "digest": "d" * 64}]}
        if path == "/api/version":
            return {"version": "t"}
        if path == "/api/ps":
            return {"models": [{"name": ollama.DEFAULT_MODEL, "size": 10, "size_vram": 10}]}
        calls.append(body["prompt"])
        codes = ["I10"] if "hypertension" in body["prompt"] else ["E11.9"]
        return {"done": True, "done_reason": "stop", "response": json.dumps({"codes": codes})}

    monkeypatch.setattr(ollama, "request", fake)
    reg = {"llm_note": {"kind": "ollama", "inputs": "note", "retries": 0}}
    name, res = icd.evaluate("llm_note", "toy", ORIGINAL, ANONYMIZED, reg)
    assert name == "llm_note" and len(calls) == 3  # the unchanged anonymized note reuses the original's prediction
    assert res["original"]["code"]["f1"] == 0.8 and res["anonymized"]["code"]["f1"] == 0.8 and not res["ranked"]
    assert "endpoint" not in res and res["config"].startswith("ollama:qwen3.5:2b-q4_K_M@dddddddddddd:inputs=note:ctx=16384")
    assert {json.loads(f.read_text())["backend"] for f in (cache / "icd" / "llm_note").glob("*.json")} == {"gpu"}
    shape = {"coder", "config", "ranked", "inputs", "n_records", "n_codes", "original", "anonymized", "agreement", "f1_delta", "f1_retention",
             "errors", "invalid_codes", "valid_for_comparison", "changed_predictions"}  # fmt: skip
    assert set(res) == shape and set(res["f1_delta"]) == {"value", "ci95"} and set(res["errors"]) == {"original", "anonymized"}
    for arm in ("original", "anonymized", "agreement"):
        assert set(res[arm]) == {"code", "category"}
        assert all(set(res[arm][lv]) == {"precision", "recall", "f1", "mean_f1", "ci95"} for lv in ("code", "category"))
    icd.evaluate("llm_note", "toy", ORIGINAL, ANONYMIZED, reg)
    assert len(calls) == 3  # everything cached
