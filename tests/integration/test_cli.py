from __future__ import annotations

import json
from pathlib import Path

from unified_harness.cli import main
from unified_harness.io import dumps_record

FIXTURE = str(Path(__file__).resolve().parents[1] / "fixtures" / "encounters.jsonl")


def _write(path, records):
    path.write_text("".join(dumps_record(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def test_validate_accepts_a_text_only_change(tmp_path, redacted, capsys):
    assert main(["validate", "--original", FIXTURE, "--anonymized", _write(tmp_path / "a.jsonl", redacted)]) == 0
    assert "0 contract violation(s)" in capsys.readouterr().out


def test_validate_rejects_changed_codes(tmp_path, redacted, capsys):
    redacted[1]["icd10_codes"].pop()
    assert main(["validate", "--original", FIXTURE, "--anonymized", _write(tmp_path / "a.jsonl", redacted)]) == 1
    assert "icd10_codes changed" in capsys.readouterr().err


def test_doctor_json_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("UNIFIED_HARNESS_MODELS", str(tmp_path))
    monkeypatch.setenv("UNIFIED_HARNESS_OLLAMA_URL", "http://127.0.0.1:9")
    assert main(["doctor", "--mode", "full", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "full" and report["ok"] is False
    assert main(["doctor", "--mode", "rules"]) == 0
