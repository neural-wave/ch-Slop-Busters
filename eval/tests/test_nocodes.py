"""No-codes bonus: pairing blind-codes models with their base, and the paired values and deltas."""

from __future__ import annotations

from eval.bonus import nocodes as N

REGISTRY = {
    "ad_night": {"cmd": ["x"]}, "ad_night_nocodes": {"cmd": ["x"], "blind_codes": True},
    "fable": {"cmd": ["x"]}, "fable_blind": {"cmd": ["x"], "blind_codes": True, "base": "fable"},
    "orphan_nocodes": {"cmd": ["x"], "blind_codes": True},          # no `orphan` entry
    "ketai_nocodes": {"cmd": ["x"]},                                # not blind: no pair
    "self": {"cmd": ["x"], "blind_codes": True, "base": "self"},    # its own base: no pair
}  # fmt: skip


def score(privacy, high, keep, concepts, icd_rel, icd):
    return {"privacy": {"removed": privacy, "removed_high_risk": high}, "keep": {"kept": keep},
            "concepts": {"kept": concepts, "breakdown": {"icd_relevant": icd_rel}}, "icd": icd}  # fmt: skip


def coder(f1):
    arm = {"precision": f1, "recall": f1, "f1": f1, "mean_f1": f1, "ci95": [f1, f1]}
    return {"coder": "llm_note", "original": {"code": arm, "category": arm}, "anonymized": {"code": arm, "category": arm}}


def test_pairs_by_suffix_and_base_key():
    assert N.pairs(REGISTRY) == [("ad_night", "ad_night_nocodes"), ("fable", "fable_blind")]
    assert N.pairs({}) == [] and N.pairs({"a_nocodes": {"blind_codes": True}}) == []


def test_rows_values_and_deltas():
    scores = {
        "ad_night": {"aci_eval": score(0.8388, 0.9981, 1.0, 1.0, 1.0, {"llm_note": coder(0.5), "knn": {"error": "unreachable"}})},
        "ad_night_nocodes": {"aci_eval": score(0.8428, 0.9981, 0.9935, 0.999, 1.0, {"llm_note": coder(0.45), "lexical": {"f1_anonymized": 0.16}})},
    }
    rows = N.rows(scores, REGISTRY)
    assert [(r["base"], r["nocodes"], r["suite"]) for r in rows] == [("ad_night", "ad_night_nocodes", "aci_eval"), ("fable", "fable_blind", "aci_eval")]
    m = rows[0]["metrics"]
    assert list(m) == ["privacy", "privacy_high", "keep", "concepts", "icd_concepts", "icd_f1:llm_note", "icd_f1:lexical"]  # no row for a failed coder
    assert m["privacy"] == {"base": 0.8388, "nocodes": 0.8428, "delta": 0.004}
    assert m["keep"]["delta"] == -0.0065 and m["privacy_high"]["delta"] == 0.0 and m["concepts"]["delta"] == -0.001
    assert m["icd_f1:llm_note"] == {"base": 0.5, "nocodes": 0.45, "delta": -0.05}
    assert m["icd_f1:lexical"] == {"base": None, "nocodes": 0.16, "delta": None}  # one side only (old score format)
    assert all(v == {"base": None, "nocodes": None, "delta": None} for v in rows[1]["metrics"].values())  # never scored


def test_failed_run_and_other_suite():
    scores = {"ad_night": {"aci_eval": {"error": "run failed"}, "aci_dev": score(0.9, None, 1.0, None, None, {})},
              "ad_night_nocodes": {"aci_dev": score(0.8, 0.7, 1.0, None, None, {})}}  # fmt: skip
    assert N.rows(scores, REGISTRY)[0]["metrics"]["privacy"] == {"base": None, "nocodes": None, "delta": None}
    dev = N.rows(scores, REGISTRY, "aci_dev")[0]["metrics"]
    assert dev["privacy"]["delta"] == -0.1 and dev["privacy_high"] == {"base": None, "nocodes": 0.7, "delta": None}
