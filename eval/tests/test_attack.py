"""bonus/attack.py on the 3-record fixture: identity output recovers what the control recovers, the cleaned output leaves
A2 nothing, A3 and A5 have hand-computed answers, A1 and A4 run with a fake worker and a fake LLM."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from eval.bonus import attack as A
from eval.tests.conftest import item, rec


@pytest.fixture(autouse=True)
def lists(monkeypatch, tmp_path):
    """Fixed name lists, no drug list, the attack cache in tmp."""
    monkeypatch.setattr(A, "first_names", lambda: frozenset({"li", "wei"}))
    monkeypatch.setattr(A, "proper_nouns", lambda: frozenset({"andrew", "martha", "boston", "zanzibar", "friday"}))
    monkeypatch.setattr(A, "drug_words", lambda: frozenset())
    monkeypatch.setattr(A, "ACACHE", tmp_path / "cache")
    monkeypatch.setattr(A, "ner_config", lambda: "test")  # it hashes the worker and the VM detector code


def golds(labels: dict, level: str = "all") -> dict[str, list[dict]]:
    return {eid: A.gold_items(g, level) for eid, g in labels.items()}


def test_identity_recovers_the_control_and_clean_nothing_a2(original, labels, clean):
    gold = golds(labels)
    ident = A.a2(original, original, gold)
    assert ident["anonymized"] == ident["control"]
    # name tokens: andrew campbell x2 (e1), wei li x2 (e2), martha x2 (e3); `li` is in the roster through the public list
    assert ident["control"] == {"recovered": 1.0, "hits": 10, "total": 10, "by_category": {"PATIENT_NAME": 1.0}}
    assert A.a2(original, clean, gold)["anonymized"]["recovered"] == 0.0


def test_a2_counts_a_surviving_surname_and_ignores_its_uses_outside_the_name(original, labels, clean):
    gold = golds(labels)
    leaky = [dict(r) for r in clean]
    leaky[0]["note"] = leaky[0]["note"].replace("[PATIENT]", "[PATIENT] Campbell")
    assert A.a2(original, leaky, gold)["anonymized"]["hits"] == 1
    # `wei` used as an ordinary word elsewhere in the original note: its survival there is not a hit
    original[1]["note"] += " Wei is a word here."
    leaky[1]["note"] += " Wei is a word here."
    assert A.a2(original, leaky, gold)["anonymized"]["hits"] == 1


def test_a3_rare_identifier_like_tokens(original, labels, clean):
    original[1]["note"] += " Flew to Zanzibar on a Friday."
    anon = [dict(r) for r in clean]
    anon[1]["note"] += " Flew to Zanzibar on a Friday."
    res = A.a3(original, anon, golds(labels))
    # rare: andrew (e1; campbell is in the clinical vocabulary of the fixture), zanzibar (e2), martha (e3); boston is in
    # two encounters, friday is a calendar word, wei is too short
    assert res["control"] == {"recovered": 1.0, "hits": 3, "total": 3, "by_category": {"PATIENT_NAME": 1.0, "UNLABELLED": 1.0}}
    assert res["anonymized"] == {"recovered": round(1 / 3, 4), "hits": 1, "total": 3, "by_category": {"PATIENT_NAME": 0.0, "UNLABELLED": 1.0}}
    assert A.a3(original, original, golds(labels))["anonymized"] == res["control"]


def test_a5_linkage(original, clean):
    assert A.a5(original, original)["anonymized"] == {"recovered": 1.0, "hits": 3, "total": 3, "by_category": {}}
    assert A.a5(original, clean)["anonymized"]["recovered"] == 1.0
    swapped = [dict(clean[2], encounter_id="e1"), clean[1], dict(clean[0], encounter_id="e3")]
    assert A.linkage(original, swapped) == (1, 3)


def fake_worker(calls: list):
    names = re.compile(r"Andrew Campbell|Wei Li|Martha|Boston|\[PATIENT\]|59 years")

    def worker(texts: list[str], device: str) -> dict[str, list]:
        calls.append(len(texts))
        return {t: [[m.start(), m.end(), m.group(), "NAME", "ner", 0.9] for m in names.finditer(t)] for t in texts}

    return worker


def test_a1_with_a_fake_worker(original, labels, clean):
    calls = []
    texts = {r[f] for r in original + clean for f in ("dialogue", "note")}
    spans = A.ner_spans(texts, "cpu", fake_worker(calls))
    assert A.ner_spans(texts, "cpu", fake_worker(calls)) == spans and calls == [len(texts)]  # the second call is cached
    gold = golds(labels)
    ident = A.a1(original, original, gold, spans)
    assert ident["anonymized"] == ident["control"] and ident["control"]["recovered"] == 1.0 and ident["control"]["total"] == 9
    assert A.a1(original, clean, gold, spans)["anonymized"]["recovered"] == 0.0  # only placeholders are detected
    assert A.a1(original, clean, gold, {})["error"].startswith("12 texts were not scanned")
    assert A.attack({"original": original, "anonymized": clean, "gold": gold}, ["A1"], "anonymous venv not found")["A1"]["error"] == "anonymous venv not found"


def fake_llm(fail: bool = False):
    def generate(model, prompt, schema, **kw):
        if fail:
            from eval.coders.ollama import PredictionError

            raise PredictionError("timeout")
        g = {"first_name": "", "last_name": "", "age": 0, "city_or_state": "", "employer_or_job": ""}
        if "Andrew Campbell" in prompt:
            g.update(first_name="Andrew", last_name="Campbell")
        if "59 years" in prompt:
            g["age"] = 60
        if "Boston" in prompt:
            g["city_or_state"] = "Boston, MA"
        return g

    return generate


def test_a4_with_a_fake_llm(original, labels, clean):
    gold = golds(labels)
    ident = A.a4(original, original, gold, "toy", fake_llm())
    assert ident["anonymized"] == ident["control"]
    # e1: name and age (60 vs 59); e2: name missed, place; e3: name missed (Martha is never guessed), place (Boston)
    assert ident["control"]["by_category"] == {"age": 1.0, "name": 0.3333, "place": 1.0} and ident["control"]["recovered"] == round(4 / 6, 4)
    assert A.a4(original, clean, gold, "toy", fake_llm())["anonymized"]["recovered"] == 0.0
    assert len(list((A.ACACHE / "llm").iterdir())) == 6  # 3 originals + 3 anonymized prompts
    assert A.a4(original, clean, gold, "toy2", fake_llm(fail=True))["error"].startswith("3 consecutive failures")


def test_a4_claude_batches_prompts_and_caches_guesses(original, labels, clean, monkeypatch):
    gold, one = golds(labels), fake_llm()
    asked = []

    def many(model, prompts):
        asked.append(len(prompts))
        return [one(model, p, A.GUESS_SCHEMA) if "Boston" not in p else "refusal:general_harms" for p in prompts]

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert A.a4(original, clean, gold, "claude-x", generate_many=many)["anonymized"]["recovered"] == 0.0
    assert asked == [6]  # one batch: 3 anonymized + 3 original prompts
    res = A.a4(original, clean, gold, "claude-x", generate_many=many)
    assert asked == [6, 2] and res["errors"] == {"refusal:general_harms": 2}  # cached guesses are not asked again; the 2 refusals (Boston) are retried
    assert A.a4(original, clean, gold, "claude-x")["error"] == "ANTHROPIC_API_KEY not set"
    assert A.a4(original, clean, gold, "claude-y", generate_many=lambda m, ps: ["api:401"] * len(ps))["error"].startswith("every claude-y call failed")


def test_claude_worker_closes_the_schema():
    import importlib.util
    import sys
    import types

    sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))  # the worker's SDK is not in the harness interpreter
    spec = importlib.util.spec_from_file_location("_claude_worker", A.CLAUDE_WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    closed = worker.closed(A.GUESS_SCHEMA)
    assert closed["additionalProperties"] is False and closed["required"] == A.GUESS_SCHEMA["required"]
    assert "additionalProperties" not in A.GUESS_SCHEMA


@pytest.mark.parametrize(("text", "age"), [("52-year-old", 52), ("fifty two", 52), ("twenty-one years old", 21), ("6 months old", None), ("old", None)])
def test_age_of(text, age):
    assert A.age_of(text) == age


def test_consensus_gold_level():
    labels = {"e1": {"pii": [item("note", "Anna", "PATIENT_NAME", direct=True, votes=1), item("note", "teacher", "OCCUPATION", direct=False, votes=1),
                             item("note", "Ohio", "LOCATION", direct=False, votes=2)]}}  # fmt: skip
    assert A.gold_level(labels) == "consensus" and A.gold_level({"e1": {"pii": [item("note", "Anna", "PATIENT_NAME")]}}) == "all"
    assert [it["text"] for it in A.gold_items(labels["e1"], "consensus")] == ["Anna", "Ohio"]


def test_max_recovered_leaves_out_a5():
    attacks = {"A2": {"anonymized": {"recovered": 0.1}}, "A3": {"anonymized": {"recovered": 0.3}}, "A4": {"anonymized": None, "error": "x"},
               "A5": {"anonymized": {"recovered": 1.0}}}  # fmt: skip
    assert A.max_recovered(attacks) == 0.3 and A.max_recovered({"A5": attacks["A5"]}) is None


@pytest.fixture
def harness(tmp_path, monkeypatch, original, labels, clean):
    """A suite and two runs (identity, clean: every gold string replaced) on disk."""
    sdir, runs = tmp_path / "suites" / "aci_eval", tmp_path / "runs"
    sdir.mkdir(parents=True)
    (sdir / "encounters.jsonl").write_text("".join(json.dumps(r) + "\n" for r in original))
    (sdir / "labels.jsonl").write_text("".join(json.dumps(g) + "\n" for g in labels.values()))
    (sdir / "suite.json").write_text(json.dumps({"input_sha1": "i", "labels_sha1": "l"}))
    for model, recs in (("identity", original), ("clean", clean)):
        (runs / model / "aci_eval").mkdir(parents=True)
        (runs / model / "aci_eval" / "output.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (runs / model / "aci_eval" / "run.json").write_text('{"exit_code": 0}')
    (runs / "broken" / "aci_eval").mkdir(parents=True)
    (runs / "broken" / "aci_eval" / "run.json").write_text('{"exit_code": 1}')
    monkeypatch.setattr(A, "SUITES_DIR", tmp_path / "suites")
    monkeypatch.setattr(A, "RUNS_DIR", runs)
    monkeypatch.setattr(A, "ATTACKS_DIR", tmp_path / "attacks")
    return tmp_path / "attacks"


def test_main_writes_the_contract_and_skips_what_it_cannot_attack(harness, capsys):
    assert A.main(["--suites", "aci_eval", "sadiqguru"]) == 0
    out = capsys.readouterr().out
    assert f"{'broken':18s} {'aci_eval':16s} skipped: run failed (exit 1)" in out and f"{'sadiqguru':16s} skipped: suite not built" in out
    ident = json.loads((harness / "identity" / "aci_eval.json").read_text())
    assert list(ident) == ["model", "suite", "n_records", "gold_level", "attacks", "max_recovered", "sources"]
    assert ident["n_records"] == 3 and ident["gold_level"] == "all" and list(ident["attacks"]) == A.DEFAULT_ATTACKS
    for r in ident["attacks"].values():
        assert set(r) >= {"name", "anonymized", "control", "notes"} and r["anonymized"] == r["control"]
    clean = json.loads((harness / "clean" / "aci_eval.json").read_text())
    assert clean["attacks"]["A2"]["anonymized"]["recovered"] == 0.0 and clean["max_recovered"] == 0.0


def test_results_of_attacks_not_rerun_are_kept_while_the_output_is_unchanged(harness, monkeypatch, clean):
    monkeypatch.setattr(A, "a4", lambda o, a, g, model: A.result("A4", {"recovered": 0.5}, {"recovered": 1.0}, "fake"))  # never the API
    A.main(["--models", "clean", "--attacks", "A4"])
    A.main(["--models", "clean"])
    path = harness / "clean" / "aci_eval.json"
    res = json.loads(path.read_text())
    assert list(res["attacks"]) == ["A2", "A3", "A4", "A5"] and res["max_recovered"] == 0.5
    out = A.RUNS_DIR / "clean" / "aci_eval" / "output.jsonl"
    out.write_text("".join(json.dumps(dict(r, note=r["note"] + " ok")) + "\n" for r in clean))
    A.main(["--models", "clean"])
    assert list(json.loads(path.read_text())["attacks"]) == A.DEFAULT_ATTACKS


def test_rare_tokens_skip_clinical_dictionary_and_short_words(monkeypatch):
    recs = [rec("a", "Zanzibar cough", "classes"), rec("b", "Boston", ""), rec("c", "Boston", "")]
    assert A.rare_tokens(recs) == {"a": {"zanzibar"}, "b": set(), "c": set()}


FAKE_WORKER = """import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
with open(sys.argv[2], "w") as out:
    for r in rows[:2]:
        out.write(json.dumps({"key": r["key"], "spans": [[0, 3, r["text"][:3], "X", "fake", 1.0]]}) + "\\n")
sys.exit(0 if len(rows) <= 2 else 1)
"""


def test_run_worker_reads_what_the_worker_wrote(tmp_path, monkeypatch):
    (tmp_path / "worker.py").write_text(FAKE_WORKER)
    monkeypatch.setattr(A, "ANONYMOUS_PYTHON", Path(sys.executable))
    monkeypatch.setattr(A, "WORKER", tmp_path / "worker.py")
    assert A.run_worker(["abc d", "xyz w"], "cpu") == {"abc d": [[0, 3, "abc", "X", "fake", 1.0]], "xyz w": [[0, 3, "xyz", "X", "fake", 1.0]]}
    assert list(A.run_worker(["a1x", "b2y", "c3z"], "cpu")) == ["a1x", "b2y"]  # it failed after two texts: those are kept
    monkeypatch.setattr(A, "ANONYMOUS_PYTHON", tmp_path / "missing")
    with pytest.raises(RuntimeError, match="anonymous venv not found"):
        A.run_worker(["x"], "cpu")


def test_a1_title_alone_is_no_hit(monkeypatch):
    original = [rec("e1", "", "Ms. Hill has a cough.")]
    gold = {"e1": [item("note", "Ms. Hill", "PATIENT_NAME")]}
    anon = [rec("e1", "", "Ms. [PATIENT] has a cough.")]
    spans = {original[0]["note"]: [[0, 8, "Ms. Hill", "NAME", "ner", 0.9]], anon[0]["note"]: [[0, 3, "Ms.", "NAME", "ner", 0.6]]}
    res = A.a1(original, anon, gold, spans)
    assert res["anonymized"]["hits"] == 0 and res["control"]["hits"] == 1
    spans[anon[0]["note"]] = [[4, 8, "Hill", "NAME", "ner", 0.6]]
    assert A.a1(original, anon, gold, spans)["anonymized"]["hits"] == 1


def test_cache_survives_a_cut_file(tmp_path):
    p = tmp_path / "c" / "x.json"
    A.cache_write(p, {"a": 1})
    assert A.cache_read(p) == {"a": 1} and not list(p.parent.glob(".*.tmp"))
    p.write_text('{"a": ', encoding="utf-8")  # a write cut by a kill
    assert A.cache_read(p) is None and A.cache_read(tmp_path / "missing.json") is None
