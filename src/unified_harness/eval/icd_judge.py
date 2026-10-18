"""External ICD-10 judge hook (milestone checkpoints only; the local proxy covers every other run).

The judge is a frozen, blinded coder outside the pipeline, for example a separate LLM run by the
development team. Two steps:

    export  write every distinct note text once, shuffled, under surrogate case ids, in batches. No
            encounter ids, gold codes or run names are written to the batches.
    score   read the judge's prediction files, map them back to every run and encounter, and report
            micro-F1 against `icd10_codes` per run, with a paired bootstrap 95% interval of the change
            from the original text. An optional re-run of the original notes measures the judge's own
            round-to-round noise.

Prediction rows: {"case_id": "case_0001", "codes": ["I10", ...]}. A missing case, a duplicate or an
unparseable row is an error, never an empty prediction. Everything is written under outputs/ (git-ignored),
since the batches contain note text.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from .. import paths
from ..io import read_jsonl

CODE = re.compile(r"[A-TV-Z][0-9][0-9A-Z](\.?[0-9A-Z]{1,4})?")
PROMPT = (
    "You are a certified medical coder. Read the physician note and list the ICD-10-CM codes for the "
    'conditions diagnosed or addressed in this encounter. Answer JSON only: {"codes": ["..."]}'
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())


def _inside_outputs(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(paths.outputs_dir().resolve()):
        raise SystemExit("judge files contain note text; keep them under outputs/ (git-ignored)")
    return path


def export(original: Path, runs: dict[str, Path], out_dir: Path, batch_size: int = 25, seed: int = 0) -> dict:
    out = _inside_outputs(out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    orig = read_jsonl(original)
    arms = {"original": orig} | {name: read_jsonl(p) for name, p in runs.items()}
    ids = [r["encounter_id"] for r in orig]
    for name, recs in arms.items():
        if [r["encounter_id"] for r in recs] != ids:
            raise SystemExit(f"{name}: encounter ids differ from the original file")
    unique = sorted({r["note"] for recs in arms.values() for r in recs})
    random.Random(seed).shuffle(unique)
    cases = {f"case_{i:04d}": t for i, t in enumerate(unique, 1)}
    (out / "batches").mkdir(parents=True)
    (out / "predictions").mkdir()
    order = list(cases)
    for b, start in enumerate(range(0, len(order), batch_size), 1):
        with (out / "batches" / f"batch_{b:02d}.jsonl").open("w", encoding="utf-8") as fh:
            for c in order[start : start + batch_size]:
                fh.write(json.dumps({"case_id": c, "note": cases[c]}, ensure_ascii=False) + "\n")
    plan = {
        "prompt": PROMPT,
        "original": str(Path(original).resolve()),
        "runs": {k: str(Path(v).resolve()) for k, v in runs.items()},
        "cases": {c: _digest(t) for c, t in cases.items()},
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return {"encounters": len(ids), "runs": len(arms), "cases": len(cases), "out": str(out)}


def load_predictions(out: Path, plan: dict) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = {}
    for f in sorted((out / "predictions").glob("*.jsonl")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            cid, codes = row.get("case_id"), row.get("codes")
            if cid not in plan["cases"] or plan["cases"][cid] in by_hash:
                raise SystemExit(f"{f.name}:{n}: unknown or duplicate case_id {cid!r}")
            if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
                raise SystemExit(f"{f.name}:{n}: codes must be a list of strings")
            by_hash[plan["cases"][cid]] = sorted({norm_code(c) for c in codes if CODE.fullmatch(c.strip().upper())})
    missing = len(set(plan["cases"].values()) - set(by_hash))
    if missing:
        raise SystemExit(f"{missing} of {len(plan['cases'])} cases have no prediction; finish the judge first")
    return by_hash


def _counts(pred: list[str], gold: set[str]) -> tuple[int, int, int]:
    p = set(pred)
    return len(p & gold), len(p - gold), len(gold - p)


def _f1(rows: list[tuple[int, int, int]]) -> float:
    tp, fp, fn = (sum(r[i] for r in rows) for i in range(3))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _interval(values: list[float]) -> list[float]:
    v = sorted(values)
    return [round(v[int(0.025 * (len(v) - 1))], 4), round(v[int(0.975 * (len(v) - 1))], 4)]


def score(out_dir: Path, rerun_dir: Path | None = None, resamples: int = 1000, seed: int = 0) -> dict:
    out = _inside_outputs(out_dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    preds = load_predictions(out, plan)
    orig = read_jsonl(Path(plan["original"]))
    gold = [{norm_code(c["code"]) for c in r["icd10_codes"]} for r in orig]
    arms = {"original": orig} | {k: read_jsonl(Path(v)) for k, v in plan["runs"].items()}
    rows = {
        k: [_counts(preds[_digest(r["note"])], g) for r, g in zip(recs, gold, strict=True)] for k, recs in arms.items()
    }
    if rerun_dir is not None:
        rerun = _inside_outputs(rerun_dir)
        rerun_preds = load_predictions(rerun, json.loads((rerun / "plan.json").read_text(encoding="utf-8")))
        rows["control_rerun_original"] = [
            _counts(rerun_preds[_digest(r["note"])], g) for r, g in zip(orig, gold, strict=True)
        ]
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {k: [] for k in rows}
    n = len(gold)
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        for k, r in rows.items():
            samples[k].append(_f1([r[i] for i in idx]))
    res = {"n_encounters": n, "cases": len(plan["cases"]), "arms": {}}
    for k, r in rows.items():
        arm = {"micro_f1": round(_f1(r), 4), "f1_ci95": _interval(samples[k])}
        if k != "original":
            arm["delta"] = round(_f1(r) - _f1(rows["original"]), 4)
            arm["delta_ci95"] = _interval([a - o for a, o in zip(samples[k], samples["original"], strict=True)])
        res["arms"][k] = arm
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
