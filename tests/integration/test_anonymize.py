"""End-to-end `unified-harness anonymize` on the synthetic fixture (rules mode, no challenge data)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from unified_harness import pipeline
from unified_harness.cli import main
from unified_harness.io import output_paths, read_jsonl, validate_output

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encounters.jsonl"
DAY = "2026-10-17"


def _anonymize(tmp_path, *extra: str) -> int:
    return main(["anonymize", "--input", str(FIXTURE), "--out-root", str(tmp_path), "--date", DAY, *extra])


def test_rules_mode_publishes_the_three_dated_files(tmp_path):
    assert _anonymize(tmp_path, "--mode", "rules") == 0
    out = output_paths(tmp_path, date.fromisoformat(DAY))
    assert out.records.name == f"anonymized-{DAY}-unified-harness.jsonl"
    original, anonymized = read_jsonl(FIXTURE), read_jsonl(out.records)
    assert validate_output(original, anonymized) == []
    text = " ".join(r["dialogue"] + " " + r["note"] for r in anonymized).lower()
    for identifier in ("lindqvist", "okafor", "4417823", "01/14/2026"):
        assert identifier not in text
    for clinical in ("140/90", "metformin 500 mg", "7.2%", "right knee"):
        assert clinical in text
    summary = json.loads(out.summary.read_text(encoding="utf-8"))
    assert summary["mode"] == "rules" and summary["n_records"] == 2 and summary["shield_edit_overlaps"] == 0


def test_decisions_sidecar_has_no_original_text(tmp_path):
    assert _anonymize(tmp_path, "--mode", "rules") == 0
    out = output_paths(tmp_path, date.fromisoformat(DAY))
    decisions = read_jsonl(out.decisions)
    assert decisions and all("text" not in d for d in decisions)
    assert "Lindqvist" not in out.decisions.read_text(encoding="utf-8")


def test_existing_output_is_not_overwritten(tmp_path, capsys):
    assert _anonymize(tmp_path, "--mode", "rules") == 0
    assert _anonymize(tmp_path, "--mode", "rules") == 1
    assert "already exists" in capsys.readouterr().err
    assert _anonymize(tmp_path, "--mode", "rules", "--overwrite") == 0


def test_full_mode_fails_closed_when_not_ready(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("UNIFIED_HARNESS_MODELS", str(tmp_path / "no-models"))
    monkeypatch.setenv("UNIFIED_HARNESS_OLLAMA_URL", "http://127.0.0.1:9")
    assert _anonymize(tmp_path, "--mode", "full") == 1
    assert "full mode is not ready" in capsys.readouterr().err
    assert not (tmp_path / DAY).exists()


def test_a_failing_record_stops_the_run_and_publishes_nothing(tmp_path, monkeypatch, capsys):
    real = pipeline.Anonymizer.anonymize

    def flaky(self, rec):
        if rec["encounter_id"] == "SYN0002":
            raise RuntimeError("simulated detector crash")
        return real(self, rec)

    monkeypatch.setattr(pipeline.Anonymizer, "anonymize", flaky)
    assert _anonymize(tmp_path, "--mode", "rules") == 1
    assert "SYN0002" in capsys.readouterr().err
    assert not (tmp_path / DAY).exists()


@pytest.mark.parametrize("preset", ["conservative", "balanced", "aggressive"])
def test_every_preset_keeps_the_contract(tmp_path, preset):
    assert _anonymize(tmp_path, "--mode", "rules", "--preset", preset) == 0
