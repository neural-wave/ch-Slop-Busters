"""coders/agent.py: the LLM judge by agents. The export (each distinct note once, never two versions of an encounter in
one batch, notes already cached skipped), the import into the coder cache, icd.score on the imported codes, and repeat
passes combined by majority vote."""

from __future__ import annotations

import json

import pytest

from eval import icd
from eval.coders import agent as J
from eval.tests.conftest import CLEAN, ORIGINAL


@pytest.fixture
def toy(tmp_path, monkeypatch):
    for attr, path in (("SUITES_DIR", tmp_path / "suites"), ("RUNS_DIR", tmp_path / "runs"), ("KEY_DIR", tmp_path / "key"), ("PACKAGE_DIR", tmp_path / "pkgs")):
        monkeypatch.setattr(J, attr, path)
    monkeypatch.setattr(icd, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "suites" / "toy").mkdir(parents=True)
    (tmp_path / "suites" / "toy" / "encounters.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ORIGINAL))
    for model, recs, rc in (("identity", ORIGINAL, 0), ("clean", CLEAN, 0), ("broken", CLEAN, 1)):
        (tmp_path / "runs" / model / "toy").mkdir(parents=True)
        (tmp_path / "runs" / model / "toy" / "output.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (tmp_path / "runs" / model / "toy" / "run.json").write_text(json.dumps({"exit_code": rc}))
    return tmp_path


def answer_all(pkg, key, codes_of) -> None:
    for name, rids in key["batches"].items():
        rows = [{"record_id": rid, "codes": codes_of(key["texts"][sha]["note"])} for rid, sha in rids.items()]
        (pkg / "answers" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "garbage\n")


def test_export_dedupes_notes_and_splits_versions(toy):
    coder = J.AgentCoder("llm_judge", "sonnet")
    key = J.export("toy", coder, toy / "pkg")
    assert key["notes_total"] == 6 and len(key["texts"]) == 6  # 3 originals (identity shares them) + 3 cleaned; broken run ignored
    shared = next(n for n in key["texts"].values() if n["note"] == ORIGINAL[0]["note"])
    assert sorted(shared["uses"]) == [["(original)", "e1"], ["identity", "e1"]]
    for rids in key["batches"].values():  # no batch holds two versions of one encounter
        eids = [key["texts"][sha]["uses"][0][1] for sha in rids.values()]
        assert len(eids) == len(set(eids))
    rows = [json.loads(line) for name in key["batches"] for line in (toy / "pkg" / f"{name}.jsonl").read_text().splitlines()]
    assert all(list(r) == ["record_id", "note"] for r in rows) and len(rows) == 6
    assert "batch_01.jsonl" in (toy / "pkg" / "prompts" / "batch_01.md").read_text() and not (toy / "pkg" / "key.json").exists()
    agent = (toy / "pkg" / "agents" / "batch_01.md").read_text()  # what the agent reads: the reading protocol, then the prompt
    assert f"{(toy / 'pkg').resolve()}/batch_01.jsonl r001" in agent and agent.endswith((toy / "pkg" / "prompts" / "batch_01.md").read_text())


def test_nothing_imported_is_a_coder_error_then_import_scores(toy):
    coder = J.AgentCoder("llm_judge", "sonnet")
    assert icd.score(coder, ORIGINAL, CLEAN)["error"].startswith("no agent answers imported yet")
    key = J.export("toy", coder, toy / "pkg")
    answer_all(toy / "pkg", key, lambda note: ["R05.9"] if "Cough" in note else (["I10"] if "Boston" in note else []))
    lines = J.import_("toy", coder)
    assert lines[0].startswith("pass 1 (") and lines[0].endswith("6/6 answers, 2 malformed")  # one garbage line per batch
    assert lines[-1].startswith("9 predictions cached")  # 3 originals x 2 uses (identity) + 3 cleaned
    res = icd.score(coder, ORIGINAL, CLEAN)  # e1 (R05.9) and e2 (I10) have gold codes; the cleaned e2 loses "Boston"
    assert res["original"]["code"]["f1"] == 1.0 and res["anonymized"]["code"]["recall"] == 0.5 and res["errors"] == {"original": 0, "anonymized": 0}
    assert J.export("toy", coder, toy / "pkg2")["batches"] == {}  # everything cached: nothing left to code


def test_a_missing_answer_is_a_failed_prediction(toy):
    coder = J.AgentCoder("llm_judge", "sonnet")
    key = J.export("toy", coder, toy / "pkg")
    answer_all(toy / "pkg", key, lambda note: ["R05.9"])
    first = next(iter(key["batches"]))
    (toy / "pkg" / "answers" / f"{first}.jsonl").unlink()
    J.import_("toy", coder)
    res = icd.score(coder, ORIGINAL, CLEAN)
    assert sum(res["errors"].values()) > 0 and res["valid_for_comparison"] is False
    with pytest.raises(SystemExit, match="earlier export"):
        J.export("toy", coder, toy / "pkg")


def test_majority_keeps_codes_most_answers_give():
    assert J.majority([["I10", "e11.9"], ["E11.9", "R05.9"], ["I10"]]) == ["I10", "E11.9"]
    assert J.majority([["I10", "E11.9"], ["I10"]]) == ["I10"]  # two passes: both must agree
    assert J.majority([["i10", "I10", "XYZ"]]) == ["I10", "XYZ"]  # one pass: its codes, canonical, an invalid one kept
    assert J.agreement([["I10"], ["E11.9"]], [["I10"], ["E11.65"]]) == {"notes": 2, "code": 0.5, "category": 1.0}


def test_repeat_passes_recode_every_note_and_vote(toy):
    coder = J.AgentCoder("llm_judge", "sonnet")
    k1 = J.export("toy", coder, J.package_dir("toy", 1))
    answer_all(J.package_dir("toy", 1), k1, lambda note: ["R05.9", "J20.9"] if "Cough" in note else [])
    J.import_("toy", coder)
    k2 = J.export("toy", coder, J.package_dir("toy", 2), pass_=2)
    assert len(k2["texts"]) == 6 and k2["seed"] == 1 and J.package_dir("toy", 2).name == "toy_pass2"  # cached notes coded again
    assert J.key_path("toy", 2).exists() and J.key_path("toy", 1).exists() and list(J.passes("toy")) == [1, 2]
    answer_all(J.package_dir("toy", 2), k2, lambda note: ["R05.9"] if "Cough" in note else [])
    k3 = J.export("toy", coder, J.package_dir("toy", 3), pass_=3)
    answer_all(J.package_dir("toy", 3), k3, lambda note: ["R05.9", "J20.9"] if "Cough" in note else [])
    lines = J.import_("toy", coder)
    assert any(ln.startswith("pass 1 vs pass 2: agreement 0.667 at code level") for ln in lines)  # {R05.9, J20.9} vs {R05.9}
    assert icd._load(icd._cache_path(coder, ORIGINAL[0])).codes == ["R05.9", "J20.9"]  # J20.9 in 2 of 3 passes
    stats = json.loads((J.KEY_DIR / "toy" / "passes.json").read_text())
    assert stats["votes_per_note"] == {"3": 6} and [p["passes"] for p in stats["agreement"]] == [[1, 2], [1, 3], [2, 3]]
    assert stats["agreement"][1]["code"] == 1.0 and sum(ln.startswith("pass ") for ln in J.status("toy")) == 3
