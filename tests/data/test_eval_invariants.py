"""Pipeline invariants on the challenge eval split (skipped when data/ is absent)."""

from __future__ import annotations

import json

import pytest

from unified_harness import paths
from unified_harness.config import make_config
from unified_harness.io import read_jsonl, validate_output
from unified_harness.pipeline import Anonymizer

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not paths.split_file("eval").exists(), reason="challenge data not in data/"),
]


@pytest.fixture(scope="module")
def EVAL() -> list[dict]:
    return read_jsonl(paths.split_file("eval"))


def test_schema_and_codes_preserved_byte_for_byte(EVAL):
    an = Anonymizer(make_config("aggressive"))
    outs = [an.anonymize(r)[0] for r in EVAL[:15]]
    assert validate_output(EVAL[:15], outs) == []
    for o, a in zip(EVAL[:15], outs, strict=True):
        assert json.dumps(o["icd10_codes"], ensure_ascii=False) == json.dumps(a["icd10_codes"], ensure_ascii=False)
        assert list(o) == list(a)


def test_deterministic(EVAL):
    an1, an2 = Anonymizer(make_config("balanced")), Anonymizer(make_config("balanced"))
    for r in EVAL[:5]:
        assert an1.anonymize(r)[0] == an2.anonymize(r)[0]
