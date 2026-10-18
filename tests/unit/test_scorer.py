"""Unified scorer, acceptance gate and ICD judge hook on a synthetic encounter (invented names)."""

from __future__ import annotations

import json

import pytest

from unified_harness.eval import gate, icd_judge, privacy

ORIGINAL = {
    "encounter_id": "SYN7",
    "dataset": "synthetic",
    "dialogue": "[doctor] hi , mrs . nadia orlov . how is the knee ?\n[patient] sore since the bake-off in elmsford .",
    "note": "Nadia Orlov, a baker from Elmsford, has right knee osteoarthritis.",
    "icd10_codes": [{"code": "M17.11", "description": "Unilateral primary osteoarthritis, right knee"}],
}
GOLD = {
    "SYN7": {
        "encounter_id": "SYN7",
        "pii": [
            {"field": "dialogue", "text": "nadia orlov", "category": "PATIENT_NAME", "risk": "high"},
            {"field": "note", "text": "Nadia Orlov", "category": "PATIENT_NAME", "risk": "high"},
            {"field": "dialogue", "text": "elmsford", "category": "LOCATION", "risk": "medium"},
            {"field": "note", "text": "Elmsford", "category": "LOCATION", "risk": "medium"},
            {"field": "note", "text": "baker", "category": "OCCUPATION", "risk": "medium"},
            {"field": "dialogue", "text": "bake-off", "category": "OTHER_QI", "risk": "low"},
        ],
        "keep": [{"field": "note", "text": "right knee", "category": "CODE_EVIDENCE"}],
        "concepts": [
            {"field": "note", "text": "osteoarthritis", "icd_relevant": True},
            {"field": "note", "text": "knee", "icd_relevant": False},
        ],
    }
}


def anon(dialogue: str, note: str) -> dict:
    return {**ORIGINAL, "dialogue": dialogue, "note": note}


def test_identity_scores_zero_privacy_and_full_utility():
    for matcher in privacy.MATCHERS:
        r = privacy.score([ORIGINAL], [ORIGINAL], GOLD, matcher=matcher)
        assert r["removed"]["ALL"] == 0.0
        assert r["name_token_removed"] == 0.0
        assert r["keep_survival"]["ALL"] == 1.0
        assert r["labeled_concept_recall"]["ALL"] == 1.0


def test_tiers_name_tokens_and_entities():
    out = anon(
        "[doctor] hi , mrs . [PATIENT] orlov . how is the knee ?\n[patient] sore since the bake-off in [CITY] .",
        "[PATIENT], a [OCCUPATION] from [CITY], has right knee osteoarthritis.",
    )
    r = privacy.score([ORIGINAL], [out], GOLD)
    # the full name string is gone from both fields, the surname token is not
    assert r["removed"]["cat:PATIENT_NAME"] == 1.0
    assert r["name_token_removed"] == pytest.approx(3 / 4)
    assert r["removed"]["tier:direct"] == 1.0
    assert r["removed"]["tier:quasi"] == pytest.approx(3 / 4)
    assert r["removed"]["risk:low"] == 0.0
    assert r["entity_removed"]["ALL"] == round(5 / 6, 4)
    assert r["keep_survival"]["ALL"] == 1.0


def test_normalized_matcher_sees_across_hyphens():
    out = anon(ORIGINAL["dialogue"].replace("bake-off", "bake off"), ORIGINAL["note"])
    assert privacy.score([ORIGINAL], [out], GOLD, matcher="word")["removed"]["risk:low"] == 1.0
    assert privacy.score([ORIGINAL], [out], GOLD, matcher="normalized")["removed"]["risk:low"] == 0.0
    with pytest.raises(ValueError):
        privacy.score([ORIGINAL], [out], GOLD, matcher="fuzzy")


def _result(**kw) -> dict:
    base = {
        "name": "run",
        "n_records": 10,
        "privacy_high": 0.99,
        "privacy_direct": 0.99,
        "privacy_quasi": 0.70,
        "privacy_low": 0.30,
        "keep_survival": 0.998,
        "concept_recall": 0.998,
        "labeled_concept_recall": None,
        "icd_agreement_f1": 0.96,
        "schema_errors": [],
    }
    return base | kw


def test_gate_passes_only_a_strict_improvement():
    best = _result()
    assert all(c.passed for c in gate.compare(_result(privacy_quasi=0.75), best))
    assert not all(c.passed for c in gate.compare(best, best))  # no gain
    assert not all(c.passed for c in gate.compare(_result(privacy_quasi=0.75, privacy_high=0.98), best))
    assert not all(c.passed for c in gate.compare(_result(privacy_quasi=0.75, keep_survival=0.99), best))
    assert not all(c.passed for c in gate.compare(_result(privacy_quasi=0.75, icd_agreement_f1=0.94), best))
    assert not all(c.passed for c in gate.compare(_result(privacy_quasi=0.75, schema_errors=["x"]), best))
    with pytest.raises(ValueError):
        gate.compare(_result(n_records=11), best)


def test_icd_judge_export_is_blinded_and_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_HARNESS_HOME", str(tmp_path))
    orig, run = tmp_path / "orig.jsonl", tmp_path / "run.jsonl"
    orig.write_text(json.dumps(ORIGINAL) + "\n")
    run.write_text(json.dumps(anon(ORIGINAL["dialogue"], "[PATIENT] has right knee osteoarthritis.")) + "\n")
    out = tmp_path / "outputs" / "judge"
    info = icd_judge.export(orig, {"balanced": run}, out)
    assert info["cases"] == 2
    batch = (out / "batches" / "batch_01.jsonl").read_text()
    assert "SYN7" not in batch and "M17" not in batch and "balanced" not in batch
    with pytest.raises(SystemExit):
        icd_judge.score(out)  # no predictions yet
    plan = json.loads((out / "plan.json").read_text())
    rows = [{"case_id": c, "codes": ["M17.11", "not-a-code"]} for c in plan["cases"]]
    (out / "predictions" / "p.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = icd_judge.score(out, resamples=50)
    assert res["arms"]["original"]["micro_f1"] == 1.0
    assert res["arms"]["balanced"]["delta"] == 0.0
    with pytest.raises(SystemExit):
        icd_judge.export(orig, {}, tmp_path / "elsewhere")  # outside outputs/
