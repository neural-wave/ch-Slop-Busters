"""JSONL input/output, the output contract and atomic publishing.

The anonymizer may change only `dialogue` and `note`. Every other key, the record order and count, and
`icd10_codes` (codes, descriptions and their order) must be identical to the input. Results are written
to temporary files and moved into place only after `validate_output` reports no error, so a failed run
never leaves a partial deliverable behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

TEXT_FIELDS = ("dialogue", "note")
REQUIRED_KEYS = ("encounter_id", "dataset", "dialogue", "note", "icd10_codes")
OUTPUT_STEM = "anonymized-{date}-unified-harness"


class InputError(ValueError):
    """The input file is not valid challenge JSONL."""


class ContractError(RuntimeError):
    """An anonymized result violates the output contract; nothing was published."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        shown = "; ".join(self.errors[:5]) + (f" (+{len(self.errors) - 5} more)" if len(self.errors) > 5 else "")
        super().__init__(f"output contract violated: {shown}")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file strictly: every non-blank line must be a JSON object."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise InputError(f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}")
            records.append(obj)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Write records in the input files' style (development files; deliverables go through publish_run)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(dumps_record(r) + "\n")
            n += 1
    return n


def validate_input(records: Sequence[dict]) -> list[str]:
    """Schema problems of challenge records (empty list when valid)."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        where = f"record {i}"
        missing = [k for k in REQUIRED_KEYS if k not in rec]
        if missing:
            errors.append(f"{where}: missing keys {missing}")
            continue
        eid = rec["encounter_id"]
        where = f"record {i} ({eid})"
        if eid in seen:
            errors.append(f"{where}: duplicate encounter_id")
        seen.add(eid)
        for f in TEXT_FIELDS:
            if not isinstance(rec[f], str):
                errors.append(f"{where}: {f} is not a string")
        codes = rec["icd10_codes"]
        if not isinstance(codes, list) or not all(
            isinstance(c, dict) and isinstance(c.get("code"), str) and isinstance(c.get("description"), str)
            for c in codes
        ):
            errors.append(f"{where}: icd10_codes must be a list of {{code, description}} strings")
    return errors


def validate_output(original: Sequence[dict], anonymized: Sequence[dict]) -> list[str]:
    """Contract violations of an anonymized result against its input (empty list when valid)."""
    errors: list[str] = []
    if len(original) != len(anonymized):
        errors.append(f"record count changed: {len(original)} -> {len(anonymized)}")
    for i, (src, out) in enumerate(zip(original, anonymized, strict=False)):
        where = f"record {i} ({src.get('encounter_id')})"
        if list(out.keys()) != list(src.keys()):
            errors.append(f"{where}: keys or key order changed")
        for key, value in src.items():
            if key in TEXT_FIELDS:
                if not isinstance(out.get(key), str):
                    errors.append(f"{where}: {key} is not a string")
            elif out.get(key) != value:  # deep, order-sensitive comparison (covers icd10_codes)
                errors.append(f"{where}: {key} changed")
    return errors


def dumps_record(record: dict) -> str:
    """One JSONL line in the input files' style (UTF-8, no ASCII escaping)."""
    return json.dumps(record, ensure_ascii=False)


@dataclass(frozen=True)
class OutputPaths:
    """The three files of one anonymization run."""

    records: Path
    decisions: Path
    summary: Path

    def all(self) -> tuple[Path, Path, Path]:
        return (self.records, self.decisions, self.summary)


def output_paths(out_root: str | Path, run_date: date) -> OutputPaths:
    """`<out_root>/<YYYY-MM-DD>/anonymized-<YYYY-MM-DD>-unified-harness{.jsonl,.decisions.jsonl,.summary.json}`."""
    day = run_date.isoformat()
    folder = Path(out_root) / day
    stem = OUTPUT_STEM.format(date=day)
    return OutputPaths(folder / f"{stem}.jsonl", folder / f"{stem}.decisions.jsonl", folder / f"{stem}.summary.json")


def _write_temp(folder: Path, lines: Iterable[str]) -> Path:
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".part")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return Path(tmp)


def publish_run(
    original: Sequence[dict],
    anonymized: Sequence[dict],
    decisions: Sequence[dict],
    summary: dict,
    paths: OutputPaths,
    *,
    overwrite: bool = False,
) -> OutputPaths:
    """Validate, then publish the three output files atomically.

    Raises ContractError (nothing written) when the result breaks the contract, and FileExistsError when
    an output already exists and `overwrite` is False.
    """
    errors = validate_output(original, anonymized)
    if errors:
        raise ContractError(errors)
    existing = [p for p in paths.all() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {existing[0]}")
    folder = paths.records.parent
    folder.mkdir(parents=True, exist_ok=True)
    temps: list[tuple[Path, Path]] = []
    try:
        temps.append((_write_temp(folder, (dumps_record(r) + "\n" for r in anonymized)), paths.records))
        temps.append((_write_temp(folder, (dumps_record(d) + "\n" for d in decisions)), paths.decisions))
        temps.append((_write_temp(folder, [json.dumps(summary, indent=1, ensure_ascii=False) + "\n"]), paths.summary))
    except BaseException:
        for tmp, _ in temps:
            tmp.unlink(missing_ok=True)
        raise
    # The records file is moved last, so its presence means the whole run was published.
    for tmp, final in sorted(temps, key=lambda t: t[1] == paths.records):
        os.replace(tmp, final)
    return paths
