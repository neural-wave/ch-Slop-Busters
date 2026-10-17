"""coders.bootstrap: exact set scores and the paired bootstrap, ported from daniel-eval's metric tests. The pinned
numbers below were produced by daniel-eval's metrics.evaluate on the same fixture (200 resamples, seed 0)."""

from __future__ import annotations

import pytest

from eval.coders.bootstrap import aggregate, counts, interval, paired, rates
from eval.coders.codes import normalize_set

GOLD = {"a": ["I10"], "b": ["E11.9", "I11.0"], "c": ["Z00.00"], "d": ["J45.909", "R05.9"], "e": ["M17.0"]}
ORIG = {"a": ["I10"], "b": ["E11.9"], "c": [], "d": ["J45.909", "R05.9", "R06.2"], "e": ["M17.0", "M25.561"]}
ANON = {"a": ["I10"], "b": [], "c": ["Z00.00"], "d": ["J45.909"], "e": ["M17.9"]}


def rows(preds: dict, gold: dict = GOLD) -> list[tuple[int, int, int]]:
    return [counts(set(normalize_set(preds[k])), set(normalize_set(gold[k]))) for k in gold]


def test_rates_empty_denominators_score_zero():
    assert rates(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert rates(1, 1, 3) == {"precision": 0.5, "recall": 0.25, "f1": pytest.approx(1 / 3)}


def test_micro_and_mean_per_note_and_hallucinations():
    gold = {"one": ["I10"], "two": ["E11.9", "I11.0", "Z00.00"]}
    s = aggregate(rows({"one": ["I10"], "two": ["A99.9999"]}, gold))  # a syntactically valid hallucination is a false positive
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 3)
    assert s["micro"]["f1"] == pytest.approx(1 / 3)
    assert s["mean_per_note"]["f1"] == 0.5
    with pytest.raises(ValueError):
        aggregate([])


def test_failed_prediction_scored_as_empty():
    s = aggregate(rows({"one": []}, {"one": ["I10"]}))
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 1)


def test_paired_bootstrap_identical_arms_zero_and_empty_policy():
    same = rows({"a": ["I10"], "b": []}, {"a": ["I10"], "b": []})
    assert aggregate(same)["mean_per_note"]["f1"] == 0.5  # a note with no gold and no prediction scores 0
    ci, delta = paired({"original": same, "anonymized": same}, [("anonymized", "original")], resamples=30)
    assert delta[("anonymized", "original")] == [0, 0]
    assert ci["original"] == ci["anonymized"]


def test_paired_bootstrap_reproduces_daniel_eval():
    arms = {"original": rows(ORIG), "anonymized": rows(ANON)}
    assert aggregate(arms["original"])["micro"]["f1"] == 0.7142857142857143
    assert aggregate(arms["anonymized"])["micro"]["f1"] == 0.5454545454545454
    assert aggregate(arms["original"])["mean_per_note"]["f1"] == pytest.approx(0.6266666666666667)
    ci, delta = paired(arms, [("anonymized", "original")], resamples=200, seed=0)
    assert ci == {"original": [0.5, 0.8571428571428571], "anonymized": [0.2, 0.9090909090909091]}
    assert delta == {("anonymized", "original"): [-0.5142857142857142, 0.2187500000000002]}


def test_more_arms_do_not_change_the_resamples():
    base = {"original": rows(ORIG), "anonymized": rows(ANON)}
    ci, _ = paired(base, resamples=100)
    ci3, _ = paired({**base, "agreement": rows(ANON, ORIG)}, resamples=100)
    assert ci3["original"] == ci["original"] and ci3["anonymized"] == ci["anonymized"]


def test_interval_interpolates():
    assert interval([float(i) for i in range(101)]) == [2.5, 97.5]
    assert interval([0.3]) == [0.3, 0.3]


def test_paired_rejects_bad_input():
    with pytest.raises(ValueError, match="resamples"):
        paired({"a": [(1, 0, 0)]}, resamples=0)
    with pytest.raises(ValueError, match="same nonzero"):
        paired({"a": [(1, 0, 0)], "b": []})
