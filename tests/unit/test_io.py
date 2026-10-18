from __future__ import annotations

import copy
import json
from datetime import date

import pytest

from unified_harness.io import (
    ContractError,
    InputError,
    output_paths,
    publish_run,
    read_jsonl,
    validate_input,
    validate_output,
)


def test_fixture_is_valid_input(encounters):
    assert len(encounters) == 2
    assert validate_input(encounters) == []


def test_read_jsonl_reports_the_bad_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\n\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(InputError, match=":3: expected a JSON object"):
        read_jsonl(p)


def test_validate_input_flags_duplicates_and_bad_codes(encounters):
    bad = copy.deepcopy(encounters)
    bad[1]["encounter_id"] = bad[0]["encounter_id"]
    bad[1]["icd10_codes"] = ["R05.9"]
    errors = validate_input(bad)
    assert any("duplicate encounter_id" in e for e in errors)
    assert any("icd10_codes must be" in e for e in errors)


def test_text_only_changes_pass(encounters, redacted):
    assert validate_output(encounters, redacted) == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.pop(), "record count changed"),
        (lambda r: r.reverse(), "encounter_id changed"),
        (lambda r: r[0]["icd10_codes"].reverse(), "icd10_codes changed"),
        (lambda r: r[0]["icd10_codes"][0].update(description="Knee pain"), "icd10_codes changed"),
        (lambda r: r[1].update(dataset="other"), "dataset changed"),
        (lambda r: r[0].update(note=None), "note is not a string"),
        (lambda r: r[0].update(extra="x"), "keys or key order changed"),
    ],
)
def test_contract_violations_are_reported(encounters, redacted, mutate, message):
    mutate(redacted)
    assert any(message in e for e in validate_output(encounters, redacted))


def test_output_paths_follow_the_naming_contract(tmp_path):
    paths = output_paths(tmp_path, date(2026, 10, 17))
    assert paths.records == tmp_path / "2026-10-17" / "anonymized-2026-10-17-unified-harness.jsonl"
    assert paths.decisions.name == "anonymized-2026-10-17-unified-harness.decisions.jsonl"
    assert paths.summary.name == "anonymized-2026-10-17-unified-harness.summary.json"


def test_publish_writes_all_files_in_input_style(tmp_path, encounters, redacted):
    paths = output_paths(tmp_path, date(2026, 10, 17))
    publish_run(encounters, redacted, [{"action": "REPLACE"}], {"n_records": 2}, paths)
    lines = paths.records.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == redacted
    assert lines[0] == json.dumps(redacted[0], ensure_ascii=False)
    assert json.loads(paths.summary.read_text(encoding="utf-8")) == {"n_records": 2}
    assert not list(paths.records.parent.glob(".tmp-*"))


def test_publish_refuses_a_contract_violation(tmp_path, encounters, redacted):
    redacted[0]["icd10_codes"] = []
    paths = output_paths(tmp_path, date(2026, 10, 17))
    with pytest.raises(ContractError, match="icd10_codes changed"):
        publish_run(encounters, redacted, [], {}, paths)
    assert not paths.records.parent.exists()


def test_publish_never_overwrites_by_default(tmp_path, encounters, redacted):
    paths = output_paths(tmp_path, date(2026, 10, 17))
    publish_run(encounters, redacted, [], {"run": 1}, paths)
    with pytest.raises(FileExistsError):
        publish_run(encounters, redacted, [], {"run": 2}, paths)
    publish_run(encounters, redacted, [], {"run": 2}, paths, overwrite=True)
    assert json.loads(paths.summary.read_text(encoding="utf-8")) == {"run": 2}
