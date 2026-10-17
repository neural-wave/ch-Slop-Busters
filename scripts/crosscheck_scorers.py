"""Check that the unified scorer reproduces both original scorers on the same outputs.

    uv run python scripts/crosscheck_scorers.py \
        --ketai ../solutions/ketai-harness --upstream ../solutions/alind+davide-harness/neural_wave \
        RUN.jsonl [RUN.jsonl ...]

Each RUN is an anonymized eval-split file. For every run the script computes:

- Ketai's own `evaluate.py` privacy and concept metrics on Ketai's eval labels, and the unified scorer
  with the `normalized` matcher on the same labels;
- the Alind+Davide `eval/privacy.py` score on the upstream eval labels, and the unified scorer with the
  `word` matcher on the same labels.

It prints the largest absolute difference per run and exits 1 when any difference exceeds the rounding
tolerance. The original solutions are not part of this repository; their paths are arguments.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from unified_harness import paths
from unified_harness.eval import privacy
from unified_harness.eval.labels import ketai_as_gold, load_ketai
from unified_harness.io import read_jsonl

TOLERANCE = 0.0006  # the scorers round to 4 decimals


def _import_from(root: Path, module: str):
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.pop(0)


def ketai_pairs(kev, original: list[dict], anonymized: list[dict], gold: dict[str, dict]) -> dict[str, tuple]:
    """{metric: (ketai value, unified value)} for one run."""
    anon = {r["encounter_id"]: r for r in anonymized}

    def prep(r):
        return {**r, **{f + "_n": kev.norm(r.get(f) or "") for f in kev.FIELDS}}

    ids = [r["encounter_id"] for r in original if r["encounter_id"] in anon and r["encounter_id"] in gold]
    by_orig = {r["encounter_id"]: r for r in original}
    per_p, per_c = [], []
    for i in ids:
        o, a = prep(by_orig[i]), prep(anon[i])
        per_p.append(kev.eval_privacy(o, a, gold[i]))
        per_c.append(kev.eval_concepts(o, a, gold[i]))
    kp, kc = kev.aggregate_privacy(per_p), kev.aggregate_concepts(per_c)
    u = privacy.score(original, anonymized, {i: ketai_as_gold(gold[i]) for i in ids}, matcher="normalized")
    pairs = {
        "mention_removal": (kp["overall"]["mention_removal"], u["removed"]["ALL"]),
        "entity_removal": (kp["overall"]["entity_removal"], u["entity_removed"]["ALL"]),
        "name_token_removal": (kp["name_token_removal"], u["name_token_removed"]),
        "concept_recall": (kc["overall"]["concept_recall"], u["labeled_concept_recall"]["ALL"]),
        "concept_recall_icd": (kc["icd_relevant"]["concept_recall"], u["labeled_concept_recall"]["icd_relevant"]),
    }
    for group, prefix in (("by_tier", "tier"), ("by_category", "cat"), ("by_field", "field")):
        for k, v in kp[group].items():
            pairs[f"{prefix}:{k}"] = (v["mention_removal"], u["removed"].get(f"{prefix}:{k}"))
    return pairs


def upstream_pairs(upriv, original: list[dict], anonymized: list[dict], gold: dict[str, dict]) -> dict[str, tuple]:
    a = upriv.score(original, anonymized, gold)
    u = privacy.score(original, anonymized, gold, matcher="word")
    pairs = {f"removed {k}": (v, u["removed"].get(k)) for k, v in a["removed"].items()}
    pairs.update({f"keep {k}": (v, u["keep_survival"].get(k)) for k, v in a["keep_survival"].items()})
    pairs["leaks"] = (len(a["leaks"]), len(u["leaks"]))
    return pairs


def worst(pairs: dict[str, tuple]) -> tuple[str, float]:
    diffs = {
        k: (abs(x - y) if x is not None and y is not None else (0.0 if x is None and y is None else 1.0))
        for k, (x, y) in pairs.items()
    }
    k = max(diffs, key=diffs.get)
    return k, diffs[k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ketai", type=Path, required=True, help="folder with Ketai's evaluate.py")
    ap.add_argument("--upstream", type=Path, required=True, help="Alind+Davide project folder (with src/)")
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    kev = _import_from(args.ketai.resolve(), "evaluate")
    upriv = _import_from((args.upstream / "src").resolve(), "neural_wave.eval.privacy")
    original = read_jsonl(paths.split_file("eval"))
    ketai_gold = load_ketai(paths.labels_dir() / "ketai" / "eval_gold.jsonl")
    upstream_gold = privacy.load_gold(paths.labels_dir() / "upstream" / "eval")

    failed = 0
    print("| run | Ketai: all PII | unified | worst diff | Alind+Davide: all PII | unified | worst diff |")
    print("|---|---:|---:|---|---:|---:|---|")
    for run in args.runs:
        anonymized = read_jsonl(run)
        kp = ketai_pairs(kev, original, anonymized, ketai_gold)
        up = upstream_pairs(upriv, original, anonymized, upstream_gold)
        (kk, kd), (uk, ud) = worst(kp), worst({k: v for k, v in up.items() if k != "leaks"})
        bad = kd > TOLERANCE or ud > TOLERANCE or up["leaks"][0] != up["leaks"][1]
        failed += bad
        km, um = kp["mention_removal"], up["removed ALL"]
        print(
            f"| {run.parent.name}/{run.name} | {km[0]:.4f} | {km[1]:.4f} | {kd:.4f} ({kk}) "
            f"| {um[0]:.4f} | {um[1]:.4f} | {ud:.4f} ({uk}) |" + (" MISMATCH" if bad else "")
        )
    print(f"\n{len(args.runs) - failed} of {len(args.runs)} runs reproduce both scorers within {TOLERANCE}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
