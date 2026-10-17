"""Regression suite: the 167 hard cases from the dataset analysis (`labels/hard_cases.jsonl`, local only).

- `keep*` cases: the span must still be present after anonymization (clinical look-alikes, eponyms,
  scribe products, numbers, typos that must not be "repaired").
- replace / remove / generalize cases: the span must be gone, under the preset whose policy covers
  that category (low-risk quasi-identifiers such as hobbies are only generalized by `aggressive`).
- Cases the policy deliberately does not implement are `xfail` with the reason (documented gaps).
"""

from __future__ import annotations

import json
import re
from functools import cache

import pytest

from unified_harness import paths
from unified_harness.config import make_config
from unified_harness.io import read_jsonl
from unified_harness.pipeline import Anonymizer

HARD_CASES = paths.labels_dir() / "hard_cases.jsonl"
HAVE_DATA = HARD_CASES.exists() and paths.split_file("eval").exists() and paths.split_file("train").exists()
CASES = read_jsonl(HARD_CASES) if HAVE_DATA else []
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not HAVE_DATA, reason="hard cases or challenge data not available locally"),
]

# Categories whose generalization only the aggressive preset performs (balanced keeps them on purpose).
AGGRESSIVE_ONLY = {
    "relative+hobby", "event", "marital_duration", "brand", "brand_vs_weekday", "school_grade",
}  # fmt: skip
EXPECTATIONS = paths.labels_dir() / "hard_cases_expectations.json"


def _expectations() -> tuple[dict, dict]:
    """Documented gaps (xfail reasons) and policy overrides, keyed by (encounter, span).

    They quote encounter text, so they live in the git-ignored labels/ folder next to the hard cases.
    """
    if not EXPECTATIONS.exists():
        return {}, {}
    data = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    xfail = {(x["encounter_id"], x["span"]): x["reason"] for x in data.get("xfail", [])}
    overrides = {(x["encounter_id"], x["span"]): x["action"] for x in data.get("policy_overrides", [])}
    return xfail, overrides


XFAIL, POLICY_OVERRIDES = _expectations()


@cache
def records() -> dict[str, dict]:
    recs = read_jsonl(paths.split_file("train")) + read_jsonl(paths.split_file("eval"))
    return {r["encounter_id"]: r for r in recs}


@cache
def anonymized(eid: str, preset: str) -> dict:
    out, _, _ = Anonymizer(make_config(preset)).anonymize(records()[eid])
    return out


def present(text: str, needle: str) -> bool:
    n = re.escape(needle.lower()).replace(r"\ ", r"\s+")
    left = r"(?<!\w)" if needle[:1].isalnum() else ""
    right = r"(?!\w)" if needle[-1:].isalnum() else ""
    return re.search(left + n + right, text.lower()) is not None


def _id(c: dict) -> str:
    return f"{c['encounter_id']}-{c['field']}-{c['category']}-{c['span_text'][:25]}"


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_hard_case(case: dict) -> None:
    eid, field, span = case["encounter_id"], case["field"], case["span_text"]
    if (eid, span) in XFAIL:
        pytest.xfail(XFAIL[(eid, span)])
    rec = records()[eid]
    assert present(rec[field], span), "hard case not found in the source text"
    action = POLICY_OVERRIDES.get((eid, span), case["expected_action"]).lower()
    preset = "aggressive" if case["category"] in AGGRESSIVE_ONLY else "balanced"
    out = anonymized(eid, preset)[field]
    main_action = re.sub(r"\(.*?\)", "", action)  # parentheticals are alternatives, not the action
    pure_keep = main_action.startswith("keep") and not re.search(r"\b(?:generalize|replace|remove)\b", main_action)
    if pure_keep:
        assert present(out, span), f"over-anonymized: {span!r} should survive ({case['why_hard']})"
    else:
        assert not present(out, span), f"leak: {span!r} still present ({case['why_hard']})"
        # the clinical part of mixed cases must survive
        for keep in re.findall(r"keep ([0-9/]+|skiing|alcohol|syndrome|ecg)", action):
            if keep in span.lower():
                assert present(out, keep), f"clinical part {keep!r} lost"
