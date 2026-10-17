"""Exact code-set scores and a paired bootstrap over encounters (not over codes).

Adapted from daniel-eval/src/icd_evaluation/metrics.py: `rates` (an empty denominator scores 0), `aggregate` (micro
and mean-per-note), `interval` (linear-interpolated 2.5 and 97.5 percentiles) and the paired bootstrap: 1000
resamples of the encounters with replacement, seed 0, every arm scored on the same resample, so an arm's CI and the
delta CI of two arms reproduce daniel-eval's numbers exactly.

Usage:
    rows = {"original": [(tp, fp, fn), ...], "anonymized": [...]}      one row per encounter, same order
    ci, delta = paired(rows, [("anonymized", "original")])            micro-F1 CI per arm, CI of anonymized - original
"""

from __future__ import annotations

import random

Row = tuple[int, int, int]  # tp, fp, fn of one encounter


def counts(predicted: set, gold: set) -> Row:
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}  # fmt: skip


def aggregate(rows: list[Row]) -> dict:
    """Micro rates over the pooled counts, and the mean of the per-encounter rates."""
    if not rows:
        raise ValueError("no encounters to score")
    tp, fp, fn = (sum(r[i] for r in rows) for i in range(3))
    per_note = [rates(*r) for r in rows]
    return {"tp": tp, "fp": fp, "fn": fn, "micro": rates(tp, fp, fn), "mean_per_note": {k: sum(r[k] for r in per_note) / len(rows) for k in per_note[0]}}


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)

    def quantile(q: float) -> float:
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return [quantile(0.025), quantile(0.975)]


def _micro_f1(tps: list[int], dens: list[int], idx: list[int]) -> float:
    tp = sum(tps[i] for i in idx)
    den = sum(dens[i] for i in idx)
    return 2 * tp / den if den else 0.0


def paired(arms: dict[str, list[Row]], pairs: list[tuple[str, str]] = (), resamples: int = 1000, seed: int = 0) -> tuple[dict, dict]:
    """95% CI of the micro F1 of every arm, and of the difference a - b for each pair (a, b), on shared resamples."""
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be positive")
    n = {len(rows) for rows in arms.values()}
    if len(n) != 1 or not n.pop():
        raise ValueError("every arm needs the same nonzero number of encounters")
    size = len(next(iter(arms.values())))
    cols = {a: ([r[0] for r in rows], [2 * r[0] + r[1] + r[2] for r in rows]) for a, rows in arms.items()}
    samples: dict[str, list[float]] = {a: [] for a in arms}
    deltas: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}
    rng = random.Random(seed)
    for _ in range(resamples):
        idx = [rng.randrange(size) for _ in range(size)]
        for a, (tps, dens) in cols.items():
            samples[a].append(_micro_f1(tps, dens, idx))
        for a, b in pairs:
            deltas[(a, b)].append(samples[a][-1] - samples[b][-1])
    return {a: interval(v) for a, v in samples.items()}, {p: interval(v) for p, v in deltas.items()}
