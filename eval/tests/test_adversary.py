"""bonus/adversary.py: the blinded export (no ids, no gold, shared sets for identical outputs, the key kept apart), the
prompt, the import scored like A4 (with the original-text set as control), stale and missing answers, and A6 surviving
automated attack runs."""

from __future__ import annotations

import json

import pytest

from eval.bonus import adversary as V
from eval.bonus import attack as A
from eval.tests.test_attack import harness, lists  # noqa: F401 (fixtures: a suite with identity and clean runs)


@pytest.fixture
def pkg(harness, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(V, "KEY_DIR", tmp_path / "key")
    key = V.export("aci_eval", ["identity", "clean", "broken"], tmp_path / "package")
    return tmp_path / "package", key


def answer(pkg_dir, key, models_in_set: str, guesses: dict[str, dict]) -> str:
    """Write the answers of the set holding `models_in_set`; `guesses` by encounter id."""
    name = next(n for n, s in key["sets"].items() if models_in_set in s["models"])
    rid = {eid: r for r, eid in key["sets"][name]["records"].items()}
    lines = [json.dumps({"record_id": rid[eid], **g}) for eid, g in guesses.items()]
    (pkg_dir / "answers" / f"{name}.jsonl").write_text("\n".join(lines + ["not json", json.dumps({"record_id": "r99"})]) + "\n")
    return name


def test_export_blinds_the_sets_and_keeps_the_key_apart(pkg, original):
    pkg_dir, key = pkg
    assert len(key["sets"]) == 2  # identity's output is the original text: one shared set, plus clean's
    ctrl = next(s for s in key["sets"].values() if s["control"])
    assert ctrl["models"] == [V.CONTROL, "identity"]
    files = sorted(p.name for p in pkg_dir.iterdir())
    assert files == ["PROMPT.md", "answers", "prompts", "set_01.jsonl", "set_02.jsonl"]  # no key, no gold, no ids
    for name, s in key["sets"].items():
        rows = [json.loads(line) for line in (pkg_dir / f"{name}.jsonl").read_text().splitlines()]
        assert all(list(r) == ["record_id", "dialogue", "note"] for r in rows) and [r["record_id"] for r in rows] == ["r01", "r02", "r03"]
        prompt = (pkg_dir / "prompts" / f"{name}.md").read_text()
        assert f"`{name}.jsonl`" in prompt and f"answers/{name}.jsonl" in prompt and "3 records" in prompt and "{" + "SET}" not in prompt
    text = "".join((pkg_dir / f"{n}.jsonl").read_text() for n in key["sets"])
    assert all(o["encounter_id"] not in text for o in original)
    assert json.loads(V.key_path("aci_eval").read_text())["sets"] == key["sets"]


def test_export_refuses_to_overwrite_answers(pkg):
    pkg_dir, key = pkg
    (pkg_dir / "answers" / "set_01.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="earlier export"):
        V.export("aci_eval", ["identity", "clean"], pkg_dir)
    V.export("aci_eval", ["identity", "clean"], pkg_dir, force=True)
    assert not any((pkg_dir / "answers").iterdir())


def test_import_scores_like_a4_with_the_original_set_as_control(pkg, harness):  # noqa: F811
    pkg_dir, key = pkg
    right = {"e1": {"first_name": "Andrew", "last_name": "Campbell", "age": 60}, "e2": {"first_name": "Wei", "city_or_state": "Boston, MA"},
             "e3": {"first_name": "Martha", "city_or_state": "Boston"}}  # fmt: skip
    answer(pkg_dir, key, "identity", right)
    answer(pkg_dir, key, "clean", {"e1": {"first_name": "John", "age": 30}, "e2": {}})  # e3 missing
    lines = V.import_("aci_eval", "agent-x")
    assert any(ln.startswith("clean") and "missing 1" in ln and "malformed 2" in ln for ln in lines)
    ident = json.loads((harness / "identity" / "aci_eval.json").read_text())["attacks"]["A6"]
    assert ident["anonymized"] == ident["control"] and ident["anonymized"]["recovered"] == 1.0 and ident["llm"] == "agent-x"
    clean = json.loads((harness / "clean" / "aci_eval.json").read_text())
    a6 = clean["attacks"]["A6"]
    assert a6["anonymized"]["recovered"] == 0.0 and a6["control"]["recovered"] == 1.0 and a6["missing"] == 1 and a6["malformed"] == 2
    assert clean["max_recovered"] == 0.0
    A.main(["--models", "clean", "--attacks", "all"])  # automated attacks never run A6 and keep it
    assert json.loads((harness / "clean" / "aci_eval.json").read_text())["attacks"]["A6"] == a6


def test_import_skips_a_model_whose_output_changed(pkg, clean):
    pkg_dir, key = pkg
    answer(pkg_dir, key, "clean", {"e1": {"first_name": "Andrew"}})
    out = A.RUNS_DIR / "clean" / "aci_eval" / "output.jsonl"
    out.write_text("".join(json.dumps(dict(r, note=r["note"] + " x")) + "\n" for r in clean))
    assert any("output changed since the export" in ln for ln in V.import_("aci_eval", "agent-x"))
    assert any("no answers" in ln for ln in V.import_("aci_eval", "agent-x") if ln.startswith("identity"))
