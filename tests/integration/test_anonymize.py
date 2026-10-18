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


def test_a_failing_record_is_passed_through_and_the_run_still_publishes(tmp_path, monkeypatch, capsys):
    """Fail open: a record every pipeline crashes on is published unchanged and listed in the summary."""
    real = pipeline.Anonymizer.anonymize

    def flaky(self, rec):
        if rec["encounter_id"] == "SYN0002":
            raise RuntimeError("simulated detector crash")
        return real(self, rec)

    monkeypatch.setattr(pipeline.Anonymizer, "anonymize", flaky)
    assert _anonymize(tmp_path, "--mode", "rules") == 0
    assert "warning: SYN0002: RuntimeError: simulated detector crash" in capsys.readouterr().err
    out = output_paths(tmp_path, date.fromisoformat(DAY))
    original, anonymized = read_jsonl(FIXTURE), read_jsonl(out.records)
    assert validate_output(original, anonymized) == [] and len(anonymized) == len(original)
    by_id = {r["encounter_id"]: r for r in anonymized}
    assert by_id["SYN0002"] == next(r for r in original if r["encounter_id"] == "SYN0002")  # unchanged
    assert by_id["SYN0001"] != next(r for r in original if r["encounter_id"] == "SYN0001")  # the others anonymized
    summary = json.loads(out.summary.read_text(encoding="utf-8"))
    assert [f["encounter_id"] for f in summary["failed_records"]] == ["SYN0002"]
    assert summary["failed_records"][0]["outcome"].startswith("passed through")


def test_a_record_the_llm_fails_on_falls_back_to_the_rules(tmp_path, monkeypatch, capsys):
    """Only the configured pipeline fails: the record is anonymized by the rules alone and the run publishes."""
    real = pipeline.Anonymizer.anonymize

    def flaky(self, rec):
        if rec["encounter_id"] == "SYN0002" and "llm" in self.cfg.detectors:
            raise RuntimeError("simulated LLM outage")
        return real(self, rec)

    monkeypatch.setattr(pipeline.Anonymizer, "anonymize", flaky)
    monkeypatch.setattr(pipeline.Anonymizer, "_model_spans", lambda self, docs: [])  # no Ollama in the tests
    assert _anonymize(tmp_path, "--detectors", "rules,llm") == 0
    assert "anonymized with the rules alone" in capsys.readouterr().err
    out = output_paths(tmp_path, date.fromisoformat(DAY))
    summary = json.loads(out.summary.read_text(encoding="utf-8"))
    assert [(f["encounter_id"], f["outcome"]) for f in summary["failed_records"]] == [
        ("SYN0002", "anonymized with the rules alone")
    ]
    text = next(r for r in read_jsonl(out.records) if r["encounter_id"] == "SYN0002")
    assert text != next(r for r in read_jsonl(FIXTURE) if r["encounter_id"] == "SYN0002")


@pytest.mark.parametrize("preset", ["conservative", "balanced", "aggressive", "privacy"])
def test_every_preset_keeps_the_contract(tmp_path, preset):
    assert _anonymize(tmp_path, "--mode", "rules", "--preset", preset) == 0


def test_llm_detector_unreachable_falls_back_to_the_rules(tmp_path, monkeypatch, capsys):
    # explicit detectors skip the doctor check: every record then fails in the LLM and is done by the rules alone
    monkeypatch.setenv("UNIFIED_HARNESS_OLLAMA_URL", "http://127.0.0.1:9")
    assert _anonymize(tmp_path, "--detectors", "rules,llm") == 0
    assert "anonymized with the rules alone" in capsys.readouterr().err
    out = output_paths(tmp_path, date.fromisoformat(DAY))
    summary = json.loads(out.summary.read_text(encoding="utf-8"))
    assert (
        len(summary["failed_records"]) == summary["n_records"]
        and validate_output(read_jsonl(FIXTURE), read_jsonl(out.records)) == []
    )


@pytest.mark.parametrize(
    "line",
    [
        '{"encounter_id": "X1", "dataset": "s", "dialogue": "hi"',
        '{"encounter_id": "X1", "dataset": "s", "dialogue": "hi"}',
    ],
    ids=["broken-json", "missing-note"],
)
def test_malformed_input_publishes_nothing(tmp_path, line):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(FIXTURE.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
    try:
        code = main(
            ["anonymize", "--input", str(bad), "--out-root", str(tmp_path / "out"), "--date", DAY, "--mode", "rules"]
        )
    except Exception:  # noqa: BLE001  (an input error may surface as an exception or an exit code)
        code = 1
    assert code != 0
    assert not (tmp_path / "out" / DAY).exists()


def test_publish_failure_leaves_no_records_file(tmp_path, monkeypatch):
    import os

    real = os.replace

    def failing(src, dst):
        if str(dst).endswith("-unified-harness.jsonl"):
            raise OSError("simulated disk failure")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError):
        _anonymize(tmp_path, "--mode", "rules")
    assert not output_paths(tmp_path, date.fromisoformat(DAY)).records.exists()


def test_identifier_over_a_number_shield_is_removed_and_published(tmp_path):
    rec = {
        "encounter_id": "SYN0003",
        "dataset": "synthetic",
        "dialogue": "",
        "note": "Lives at 4 Elm Row, Glasgow G11 6BQ. Seen on 10 Oct 2026. Sodium 140 mmol/L.",
        "icd10_codes": [{"code": "E87.1", "description": "Hypo-osmolality and hyponatremia"}],
    }
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    code = main(["anonymize", "--input", str(src), "--out-root", str(tmp_path / "o"), "--date", DAY, "--mode", "rules"])
    assert code == 0
    out = output_paths(tmp_path / "o", date.fromisoformat(DAY))
    note = read_jsonl(out.records)[0]["note"]
    assert "G11 6BQ" not in note and "10 Oct 2026" not in note and "140 mmol/L" in note
    assert json.loads(out.summary.read_text(encoding="utf-8"))["shield_edit_overlaps"] == 0
