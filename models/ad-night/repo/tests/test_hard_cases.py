"""Regression suite: the 167 hard cases from the dataset analysis (`outputs/eda/hard_cases.jsonl`).

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
from pathlib import Path

import pytest

from neural_wave.config import make_config
from neural_wave.io import read_jsonl
from neural_wave.pipeline import Anonymizer

ROOT = Path(__file__).resolve().parents[1]
CASES = [json.loads(line) for line in (ROOT / "outputs/eda/hard_cases.jsonl").open(encoding="utf-8")]

# Categories whose generalization only the aggressive preset performs (balanced keeps them on purpose).
AGGRESSIVE_ONLY = {
    "relative+hobby", "event", "marital_duration", "brand", "brand_vs_weekday", "school_grade",
}  # fmt: skip
# Known, documented gaps: (encounter, span) -> reason.
XFAIL = {
    ("D2N138", "that was my first fall here"): "doctor biography: no detector for first-person clinician life details",
    ("D2N038", "i used to party a lot and even did use iv drugs"): "sensitive narrative is kept verbatim (no rewriting of clinical history)",
    ("D2N113", "iv drug use"): "sensitive history is clinical content and is kept verbatim (policy: never rewrite)",
    ("D2N074", "Lives with wife, recently purchased a new house"): "living situation is kept (functional/social history)",
    ("D2N180", "car wreck last december"): "month names alone are relative timing and are kept by every preset",
    ("D2N138", "enlisted early spring of 2022"): "only the year and the military service are generalized; 'enlisted early spring' remains",
    ("D2N129", "working in a bakery this summer"): "generic workplace without a 'work at/own' frame is not detected",
    ("D2N058", "the rex center"): "ASR-garbled generic place ('rec center') is not a gazetteer entry",
    ("D2N154", "detroit is a hockey town"): "city mentioned without a locative cue (needs a model detector)",
    ("D2N099", "dino"): "pet name / nickname with no cue (needs a model detector)",
    ("D2N021", "Walmart"): "store name used as a place to buy medicine is kept by the cue-gated employer rule",
}


# Where the implemented policy deliberately differs from the EDA's first suggestion.
POLICY_OVERRIDES = {
    # Infant ages are generalized to `[infant]` (age band) instead of kept: the exact month count plus the
    # visit date re-derives the DOB, and no code in the data depends on the month count.
    ("D2N099", "9-month-old"): "generalize to [infant]",
}


@cache
def records() -> dict[str, dict]:
    recs = read_jsonl(ROOT / "resources/data/patient_doctor_encounters.jsonl") + read_jsonl(ROOT / "resources/data/eval_patient_doctor_encounters.jsonl")
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
