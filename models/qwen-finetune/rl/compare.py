"""Compare anonymization runs on a split with paired encounter-level bootstrap CIs (TRAINING_PLAN.md §8, §9).

  python3 rl/compare.py runs/pilot_sft runs/pilot_ppo --split test
  python3 rl/compare.py runs/vllm4gb-qwen3.5-2b_generalize runs/pilot_ppo --split dev --n-boot 5000

For every run: PII removal (all / direct / quasi / per category), name-token removal, concept recall (all /
ICD-relevant), the share of encounters where any direct identifier survives. For every run after the first: the
difference to the first run with a 95% CI from resampling encounters (the same resample for both runs). ICD-10 F1 needs
the coder model and comes from ../term-generalizer/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import evaluate as E  # noqa: E402
from common import EVAL_DATA, GOLD_EVAL, norm, read_jsonl  # noqa: E402

SPLITS = Path(__file__).resolve().parent / "splits.json"


def per_encounter(anon_path: Path, orig: dict, gold: dict, ids: list[str]) -> dict:
    anon = {r["encounter_id"]: r for r in read_jsonl(anon_path)}
    missing = [i for i in ids if i not in anon]
    if missing:
        raise SystemExit(f"{anon_path}: {len(missing)} encounters of the split missing, e.g. {missing[:3]}")
    out = {}
    for i in ids:
        o = {**orig[i], **{f + "_n": norm(orig[i].get(f) or "") for f in E.FIELDS}}
        a = {**anon[i], **{f + "_n": norm(anon[i].get(f) or "") for f in E.FIELDS}}
        p, c = E.eval_privacy(o, a, gold[i]), E.eval_concepts(o, a, gold[i])
        rows = p["rows"]
        out[i] = {
            "pii": (sum(r["orig"] for r in rows), sum(r["orig"] - r["left"] for r in rows)),
            "direct": (sum(r["orig"] for r in rows if r["tier"] == "direct"),
                       sum(r["orig"] - r["left"] for r in rows if r["tier"] == "direct")),
            "quasi": (sum(r["orig"] for r in rows if r["tier"] == "quasi"),
                      sum(r["orig"] - r["left"] for r in rows if r["tier"] == "quasi")),
            "name_tokens": (sum(r["orig"] for r in p["token_rows"]),
                            sum(r["orig"] - r["left"] for r in p["token_rows"])),
            "concepts": (len(c["rows"]), sum(r["left"] > 0 for r in c["rows"])),
            "icd_concepts": (sum(r["icd_relevant"] for r in c["rows"]),
                             sum(r["icd_relevant"] and r["left"] > 0 for r in c["rows"])),
            "direct_leak": (1, int(any(r["left"] and r["tier"] == "direct" for r in rows))),
            "cats": {cat: (sum(r["orig"] for r in rows if r["category"] == cat),
                           sum(r["orig"] - r["left"] for r in rows if r["category"] == cat))
                     for cat in {r["category"] for r in rows}},
        }
    return out


METRICS = ["pii", "direct", "quasi", "name_tokens", "concepts", "icd_concepts", "direct_leak"]   # direct_leak: share of encounters with a surviving direct identifier (lower is better)


def ratio(enc: dict, ids, key) -> float | None:
    d = sum(enc[i][key][0] for i in ids)
    return sum(enc[i][key][1] for i in ids) / d if d else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (with anonymized.jsonl) or .jsonl files")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--orig", default=str(EVAL_DATA))
    ap.add_argument("--gold", default=str(GOLD_EVAL))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the numbers to this file")
    args = ap.parse_args()
    orig = {r["encounter_id"]: r for r in read_jsonl(args.orig)}
    gold = {g["encounter_id"]: g for g in read_jsonl(args.gold)}
    ids = sorted(gold) if args.split == "all" else json.loads(SPLITS.read_text())[args.split]
    paths = [Path(r) / "anonymized.jsonl" if Path(r).is_dir() else Path(r) for r in args.runs]
    encs = [per_encounter(p, orig, gold, ids) for p in paths]
    rng = random.Random(args.seed)
    boots = [[rng.choice(ids) for _ in ids] for _ in range(args.n_boot)]
    result = {"split": args.split, "n_encounters": len(ids), "runs": {}}
    names = [str(Path(r).name if Path(r).is_dir() else Path(r).parent.name) for r in args.runs]
    print(f"split {args.split}: {len(ids)} encounters; differences vs {names[0]} with paired bootstrap 95% CI "
          f"({args.n_boot} resamples)\n")
    print("| run | " + " | ".join(METRICS) + " |\n|---|" + "---:|" * len(METRICS))
    for name, enc in zip(names, encs):
        row = {k: ratio(enc, ids, k) for k in METRICS}
        result["runs"][name] = {"metrics": row,
                                "by_category": {c: ratio({i: {c: enc[i]["cats"].get(c, (0, 0))} for i in ids}, ids, c)
                                                for c in sorted({c for i in ids for c in enc[i]["cats"]})}}
        print(f"| {name} | " + " | ".join("n/a" if v is None else f"{100 * v:.1f}" for v in row.values()) + " |")
    print()
    for name, enc in zip(names[1:], encs[1:]):
        diffs = {}
        for k in METRICS:
            ds = sorted(d for b in boots
                        if (x := ratio(enc, b, k)) is not None and (y := ratio(encs[0], b, k)) is not None
                        for d in [x - y])
            point = (ratio(enc, ids, k) or 0) - (ratio(encs[0], ids, k) or 0)
            lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]
            diffs[k] = {"diff": point, "ci95": [lo, hi], "excludes_0": lo > 0 or hi < 0}
        result["runs"][name]["vs_" + names[0]] = diffs
        print(f"{name} - {names[0]}:")
        for k, d in diffs.items():
            print(f"  {k:13s} {100 * d['diff']:+6.1f}  [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"
                  + ("  *" if d["excludes_0"] else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
