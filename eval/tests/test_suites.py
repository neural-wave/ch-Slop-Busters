"""suites.ensure: a suite is rebuilt when its suite.json is missing or old, its definition, source or build code changed, or
its files no longer match their sha1. The VM annotation dir a suite was built with is kept."""

from __future__ import annotations

import json

import pytest

from eval import suites as S

SPEC = {"source": "toy", "role": "kaggle", "about": "toy suite"}


@pytest.fixture
def toy(tmp_path, monkeypatch):
    src = tmp_path / "converted" / "toy"
    src.mkdir(parents=True)
    encs = [{"encounter_id": f"t{i}", "dataset": "toy", "dialogue": f"Hi Anna {i}", "note": "Fine.", "icd10_codes": []} for i in range(3)]
    labs = [{"encounter_id": e["encounter_id"], "pii_gold": "partial", "icd10_source": "none", "meta": {"language": "en"},
             "pii": [{"field": "dialogue", "text": "Anna", "category": "PATIENT_NAME", "risk": "high"}], "keep": []} for e in encs]  # fmt: skip
    (src / "encounters.jsonl").write_text("".join(json.dumps(e) + "\n" for e in encs))
    (src / "labels.jsonl").write_text("".join(json.dumps(lab) + "\n" for lab in labs))
    (src / "manifest.json").write_text('{"license": "test", "language": "en"}')
    monkeypatch.setattr(S, "CONVERTED", tmp_path / "converted")
    monkeypatch.setattr(S, "SUITES_DIR", tmp_path / "suites")
    monkeypatch.setattr(S, "SUITES", {"toy": dict(SPEC)})
    builds = []
    real = S.build
    monkeypatch.setattr(S, "build", lambda *a: builds.append(a[0]) or real(*a))
    return src, builds


def test_builds_once_then_reuses(toy):
    _, builds = toy
    info = S.ensure(["toy"])["toy"]
    assert builds == ["toy"] and info["records"] == 3 and len(info["source_sha1"]) == 40
    assert S.ensure(["toy"])["toy"] == info and builds == ["toy"]


def test_rebuilds_on_source_change(toy):
    src, builds = toy
    first = S.ensure(["toy"])["toy"]
    (src / "manifest.json").write_text('{"license": "other", "language": "en"}')
    second = S.ensure(["toy"])["toy"]
    assert builds == ["toy", "toy"] and second["source_sha1"] != first["source_sha1"] and second["license"] == "other"


def test_rebuilds_without_source_sha1_or_on_new_definition(toy, monkeypatch):
    _, builds = toy
    info = S.ensure(["toy"])["toy"]
    (S.SUITES_DIR / "toy" / "suite.json").write_text(json.dumps({k: v for k, v in info.items() if k != "source_sha1"}))
    S.ensure(["toy"])
    assert builds == ["toy", "toy"]
    monkeypatch.setitem(S.SUITES["toy"], "limit", 2)
    assert S.ensure(["toy"])["toy"]["records"] == 2 and len(builds) == 3


def test_rebuild_is_byte_identical(toy):
    S.ensure(["toy"])
    before = {f: (S.SUITES_DIR / "toy" / f).read_bytes() for f in ("encounters.jsonl", "labels.jsonl", "suite.json")}
    S.ensure(["toy"], force=True)
    assert before == {f: (S.SUITES_DIR / "toy" / f).read_bytes() for f in before}


def test_missing_source_keeps_the_suite(toy, capsys):
    src, builds = toy
    info = S.ensure(["toy"])["toy"]
    (src / "labels.jsonl").unlink()
    assert S.ensure(["toy"]) == {"toy": info} and builds == ["toy"]
    assert "keeping the existing suite" in capsys.readouterr().err
    (S.SUITES_DIR / "toy" / "suite.json").unlink()
    assert S.ensure(["toy"]) == {}


def test_rebuilds_damaged_suite_files(toy, capsys):
    _, builds = toy
    S.ensure(["toy"])
    enc = S.SUITES_DIR / "toy" / "encounters.jsonl"
    enc.write_text(enc.read_text().splitlines(keepends=True)[0])  # an interrupted build
    assert S.ensure(["toy"])["toy"]["records"] == 3 and len(enc.read_text().splitlines()) == 3 and builds == ["toy", "toy"]
    (S.SUITES_DIR / "toy" / "labels.jsonl").write_text("")
    S.ensure(["toy"])
    assert len(builds) == 3 and "suite toy: rebuilt (suite files changed)" in capsys.readouterr().out


def test_rebuilds_on_build_code_change(toy, monkeypatch):
    _, builds = toy
    S.ensure(["toy"])
    monkeypatch.setattr(S, "CODE_SHA1", "0" * 40)
    assert S.ensure(["toy"])["toy"]["code_sha1"] == "0" * 40 and builds == ["toy", "toy"]


def test_main_rebuilds_unless_if_stale(toy):
    _, builds = toy
    S.main(["--only", "toy"])
    S.main(["--only", "toy"])
    S.main(["--only", "toy", "--if-stale"])
    assert builds == ["toy", "toy"]


def test_vm_annotations_dir_is_kept(toy, tmp_path, monkeypatch):
    """`eval run` and `eval all` rebuild with the VM dir recorded in suite.json, not the default one."""
    src, builds = toy
    src.rename(src.parent / "aci_bench")
    monkeypatch.setitem(S.SUITES["toy"], "source", "aci_bench")
    default, mine = tmp_path / "vm_default", tmp_path / "vm_mine"
    default.mkdir()
    mine.mkdir()
    (mine / "t0.json").write_text(json.dumps({"phi": [{"field": "dialogue", "text": "Anna", "type": "NAME_PATIENT"}], "medical_concepts": []}))
    monkeypatch.setattr(S, "VM_ANNOTATIONS", default)
    info = S.ensure(["toy"], mine, force=True)["toy"]
    assert info["pii_alt_records"] == 1 and info["vm_annotations"] == str(mine.resolve())
    assert S.ensure(["toy"])["toy"] == info and len(builds) == 1
    (mine / "t0.json").write_text(json.dumps({"phi": [], "medical_concepts": []}))
    assert S.ensure(["toy"])["toy"]["vm_annotations"] == str(mine.resolve()) and len(builds) == 2
    S.main(["--only", "toy", "--vm-annotations", str(default)])  # back to the default: nothing recorded
    assert "vm_annotations" not in json.loads((S.SUITES_DIR / "toy" / "suite.json").read_text())
    assert S.ensure(["toy"])["toy"]["pii_alt_records"] == 0 and len(builds) == 3
