"""Robustness to unseen formats (bonus task e): build the three robust suites and compare the models on them.

Usage: python3 -I eval robustness [--models ...] [--force] [--run] [--icd ...]

R1 robust_discharge  devp discharge summaries (the same Kaggle records as devp_discharge) as lowercase dictation, with
                     injected admission/discharge dates, age, physician and phone; exhaustive gold, 5 gold codes
R2 robust_medconv    real web medical Q&A (cased, punctuated) as [patient]/[doctor] dialogue with injected patient PII;
                     partial gold (the posts' own names are unlabelled), no codes
R3 robust_synthetic  24 synthetic hospital documents (triage, discharge letter, referral e-mail, handoff, reports, phone
                     triage) written by the A+D team; exhaustive gold, hand-assigned codes

The sets are A+D's (models/ad-night/repo/src/neural_wave/eval/robustness.py), ported to scripts/convert/robust_*.py.
This command converts them into data/converted/robust_* when missing (`--force`: always, and rebuilds the suites),
builds the suites data/eval/suites/robust_* (role `robust`) when missing or stale, and prints one row per model:
privacy (share of gold PII occurrences removed), high-risk privacy and medical terms kept on each robust suite, read
from data/eval/scores/. The header gives each suite's complexity rank from data/eval/drift.json (`eval drift`: 1 =
closest to ACI train by unigram JSD), the bonus task's "complexity of additional data". `--run` first runs and scores
the models on the robust suites (`eval run`, `eval score`); without it, a model with no score shows `-`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from eval import suites as S
from eval.common import CONVERTED, OUT, REPO, SCORES_DIR

ROBUST = ("robust_discharge", "robust_medconv", "robust_synthetic")
CONVERT = REPO / "scripts" / "convert"
DRIFT = OUT / "drift.json"
COLUMNS = (("privacy", "removed"), ("high", "removed_high_risk"), ("terms", "kept"))


def convert(force: bool = False) -> list[str]:
    """Run the converters of the robust sets that are missing (all with `force`), then validate them; returns failures."""
    todo = [s for s in ROBUST if force or not all((CONVERTED / S.SUITES[s]["source"] / f).exists() for f in S.SOURCE_FILES)]
    failed = []
    for s in todo:
        print(f"converting {s}: scripts/convert/{s}.py", flush=True)
        if subprocess.run([sys.executable, "-I", "-B", str(CONVERT / f"{s}.py")], cwd=REPO).returncode:
            failed.append(s)
    done = [str(CONVERTED / S.SUITES[s]["source"]) for s in todo if s not in failed]
    if done and subprocess.run([sys.executable, "-I", "-B", str(CONVERT / "validate.py"), *done], cwd=REPO).returncode:
        failed.append("validate")
    return failed


def cell(score: dict | None, block: str, key: str) -> str:
    if not score or score.get("error"):
        return "-" if not score else "err"
    v = (score.get("terms" if block == "terms" else "privacy") or {}).get(key)
    return "-" if v is None else f"{v:.3f}"


def table(models: list[str]) -> str:
    ranks = {}
    if DRIFT.exists():
        try:
            ranks = json.loads(DRIFT.read_text(encoding="utf-8")).get("complexity_rank", {})
        except json.JSONDecodeError:
            pass
    head = ["model"] + [f"{s} {c}" for s in ROBUST for c, _ in COLUMNS]
    rows = [head, ["complexity rank"] + [str(ranks.get(s, "-")) if i == 0 else "" for s in ROBUST for i in range(len(COLUMNS))]]
    for m in models:
        row = [m]
        for s in ROBUST:
            p = SCORES_DIR / m / f"{s}.json"
            score = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            row += [cell(score, c, k) for c, k in COLUMNS]
        rows.append(row)
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|---|" + "---:|" * (len(head) - 1))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eval.run import load_models

    registry = load_models()
    p = argparse.ArgumentParser(prog="python3 -I eval robustness", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--models", nargs="+", help="rows of the table (default: every model with a robust score; with --run: every enabled model)")
    p.add_argument("--force", action="store_true", help="reconvert the robust sets and rebuild their suites")
    p.add_argument("--run", action="store_true", help="run and score the models on the robust suites first")
    p.add_argument("--icd", nargs="+", help="ICD coders for the score step (default: score's)")
    a = p.parse_args(argv)
    failed = convert(a.force)
    built = S.ensure(list(ROBUST), force=a.force)
    for s in ROBUST:
        if s in built:
            info = built[s]
            print(f"{s:17s} {info['records']:4d} records  pii items {info['pii_items']:5d} ({'/'.join(info['pii_gold'])})  keep {info['keep_items']:4d}"
                  f"  icd {info['with_icd10']}")  # fmt: skip
        else:
            failed.append(s)
    suites = [s for s in ROBUST if s in built]
    if a.run and suites:
        from eval import run, score

        models = a.models or [m for m, spec in registry.items() if spec.get("enabled", True)]
        rc = run.main(["--models", *models, "--suites", *suites])
        rc |= score.main(["--models", *models, "--suites", *suites] + (["--icd", *a.icd] if a.icd else []))
        if rc:
            failed.append("run/score")
    scored = sorted(d.name for d in SCORES_DIR.iterdir() if d.is_dir() and any((d / f"{s}.json").exists() for s in ROBUST)) if SCORES_DIR.is_dir() else []
    models = a.models or scored
    print()
    print(table(models) if models else "no scores on the robust suites yet: python3 -I eval robustness --run (or eval run/score --suites robust_...)")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
