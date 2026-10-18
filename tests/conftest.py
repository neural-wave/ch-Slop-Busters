from __future__ import annotations

import copy
from pathlib import Path

import pytest

from unified_harness.io import read_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def encounters() -> list[dict]:
    """Two synthetic encounters (invented people, no challenge data)."""
    return read_jsonl(FIXTURES / "encounters.jsonl")


@pytest.fixture
def redacted(encounters) -> list[dict]:
    """A contract-valid stand-in for an anonymized result: only the text fields differ."""
    out = copy.deepcopy(encounters)
    for rec in out:
        rec["dialogue"] = rec["dialogue"].replace("marta", "[PATIENT]")
        rec["note"] = rec["note"].replace("Marta Lindqvist", "[PATIENT]")
    return out
