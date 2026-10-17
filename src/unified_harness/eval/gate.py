"""Acceptance gate: is a candidate run better than the current best on the same split?

A change is kept only when, on the tune split:
1. direct / high-risk removal does not drop;
2. quasi or low-risk removal rises;
3. must-keep traps and medical concepts stay at or above 99.5%;
4. local ICD proxy agreement does not drop by more than its run-to-run noise;
5. the output has no contract violation (fields other than dialogue and note, icd10_codes included,
   are unchanged).

Inputs are the result files written by `unified-harness evaluate` / `run` (outputs/eval/<name>.json).
"""

from __future__ import annotations

from dataclasses import dataclass

FLOOR = 0.995
ICD_NOISE = 0.01


@dataclass
class Check:
    name: str
    candidate: str
    baseline: str
    passed: bool


def _v(res: dict, key: str) -> float | None:
    v = res.get(key)
    return float(v) if v is not None else None


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def compare(candidate: dict, baseline: dict, floor: float = FLOOR, icd_noise: float = ICD_NOISE) -> list[Check]:
    if candidate.get("n_records") != baseline.get("n_records"):
        raise ValueError(
            f"runs are on different inputs: {candidate.get('n_records')} vs {baseline.get('n_records')} records"
        )
    checks: list[Check] = []
    for key in ("privacy_high", "privacy_direct"):
        c, b = _v(candidate, key), _v(baseline, key)
        if c is None or b is None:
            raise ValueError(f"{key} missing: both runs must be scored against silver labels")
        checks.append(Check(f"{key} does not drop", _fmt(c), _fmt(b), c >= b))
    rises = {k: (_v(candidate, k), _v(baseline, k)) for k in ("privacy_quasi", "privacy_low")}
    checks.append(
        Check(
            "quasi or low-risk removal rises",
            " / ".join(_fmt(c) for c, _ in rises.values()),
            " / ".join(_fmt(b) for _, b in rises.values()),
            any(c is not None and b is not None and c > b for c, b in rises.values()),
        )
    )
    for key in ("keep_survival", "concept_recall", "labeled_concept_recall"):
        c = _v(candidate, key)
        if c is None and key == "labeled_concept_recall":
            continue  # only the validation labels carry concept annotations
        checks.append(Check(f"{key} >= {floor}", _fmt(c), _fmt(_v(baseline, key)), c is not None and c >= floor))
    c, b = _v(candidate, "icd_agreement_f1"), _v(baseline, "icd_agreement_f1")
    checks.append(
        Check(
            f"ICD proxy agreement within {icd_noise} of baseline",
            _fmt(c),
            _fmt(b),
            c is not None and b is not None and c >= b - icd_noise,
        )
    )
    errors = candidate.get("schema_errors")
    checks.append(Check("no output contract violation", str(len(errors or [])), "0", errors == []))
    return checks


def table(checks: list[Check], candidate: str, baseline: str) -> str:
    lines = [f"| criterion | {candidate} | {baseline} | result |", "|---|---:|---:|---|"]
    lines += [f"| {c.name} | {c.candidate} | {c.baseline} | {'pass' if c.passed else 'FAIL'} |" for c in checks]
    return "\n".join(lines)
