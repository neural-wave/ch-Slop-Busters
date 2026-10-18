"""JSONL I/O and output schema validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

TEXT_FIELDS = ("dialogue", "note")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:  # tolerate a UTF-8 BOM
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Write records exactly like the input files (`ensure_ascii=False`, default separators)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def validate_output(original: list[dict], anonymized: list[dict]) -> list[str]:
    """Schema check: same count/order/keys/types; `icd10_codes` byte-identical; ids unchanged."""
    errs = []
    if len(original) != len(anonymized):
        errs.append(f"record count differs: {len(original)} vs {len(anonymized)}")
    for i, (o, a) in enumerate(zip(original, anonymized)):
        if list(o.keys()) != list(a.keys()):
            errs.append(f"#{i} keys differ: {list(o)} vs {list(a)}")
        for k in o:
            if k in TEXT_FIELDS:
                if not isinstance(a.get(k), str):
                    errs.append(f"#{i} {k} is not a string")
            elif json.dumps(o[k], ensure_ascii=False) != json.dumps(a.get(k), ensure_ascii=False):
                errs.append(f"#{i} field {k!r} changed (must be preserved byte-for-byte)")
    return errs
