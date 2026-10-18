from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import unified_harness
from unified_harness.environment import REVISION_FILE, check_hf_model, check_ollama_model, doctor, load_manifest

ROOT = Path(__file__).resolve().parents[2]
UNREACHABLE = "http://127.0.0.1:9"  # discard port: connection refused, no network involved


def test_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == unified_harness.__version__


def test_manifest_pins_exact_revisions():
    manifest = load_manifest()
    for entry in manifest["huggingface"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry["repo"]
        assert entry["files"]
    assert manifest["ollama"]["detector"]["digest"]


def test_rules_mode_needs_nothing(tmp_path):
    report = doctor("rules", root=tmp_path, url=UNREACHABLE)
    assert report.ok


def test_full_mode_fails_clearly_without_models_or_ollama(tmp_path):
    report = doctor("full", root=tmp_path, url=UNREACHABLE)
    assert not report.ok
    details = " | ".join(c.detail for c in report.checks if not c.ok)
    assert "run scripts/download_models.py" in details
    assert "Ollama is not reachable" in details


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        doctor("regex-only")


def test_hf_model_check_requires_the_pinned_revision(tmp_path):
    entry = load_manifest()["huggingface"]["deid_roberta"]
    folder = tmp_path / entry["dir"]
    folder.mkdir()
    for f in entry["files"]:
        (folder / f).write_text("x")
    (folder / REVISION_FILE).write_text("0" * 40)
    assert not check_hf_model(entry, tmp_path).ok
    (folder / REVISION_FILE).write_text(entry["revision"])
    assert check_hf_model(entry, tmp_path).ok


def test_ollama_check_compares_digests():
    entry = {"name": "m:tag", "digest": "abc"}
    assert check_ollama_model(entry, {"m:tag": "abc"}, "u").ok
    assert not check_ollama_model(entry, {"m:tag": "def"}, "u").ok
    assert not check_ollama_model(entry, {}, "u").ok
