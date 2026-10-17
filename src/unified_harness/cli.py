"""Command line interface: `unified-harness <command>`.

anonymize   anonymize a JSONL file; publishes outputs/<date>/anonymized-<date>-unified-harness.* atomically
evaluate    score an anonymized file (privacy against silver labels, utility proxies, efficiency)
run         anonymize a split in a clean process and evaluate it (one row in outputs/eval/results.md)
sweep       presets x codes/no-codes: the privacy/utility curve
attack      adversarial re-identification attacks on an anonymized file
robustness  build and evaluate the unseen-format robustness sets
show        print one anonymized encounter with every decision (debug view, contains original text)
doctor      check that a mode (rules | full) has everything it needs
validate    check an anonymized file against its input (output contract)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from . import __version__, paths
from .environment import MODES, doctor, load_manifest
from .io import (
    ContractError,
    InputError,
    output_paths,
    publish_run,
    read_jsonl,
    validate_input,
    validate_output,
)

MODE_DETECTORS = {"rules": ("rules",), "llm": ("rules", "llm"), "full": ("rules", "deid", "gliner", "llm")}
PRESETS = ("conservative", "balanced", "aggressive", "privacy")


def _detectors(s: str | None) -> tuple[str, ...] | None:
    return tuple(x.strip() for x in s.split(",") if x.strip()) if s else None


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------- anonymize
def cmd_anonymize(a: argparse.Namespace) -> int:
    from .config import make_config
    from .eval.resources import ollama_vram_mb, peak_rss_mb, torch_vram_mb
    from .pipeline import Anonymizer

    detectors = _detectors(a.detectors) or MODE_DETECTORS[a.mode]
    if a.mode != "rules" and not a.detectors:
        report = doctor(a.mode, detector_model=a.detector_model)
        if not report.ok:
            for c in report.checks:
                if not c.ok:
                    print(f"error: {c.name}: {c.detail}", file=sys.stderr)
            return _fail(
                f"{a.mode} mode is not ready (see `unified-harness doctor --mode {a.mode}`); nothing was anonymized"
            )

    try:
        records = read_jsonl(a.input)
    except InputError as exc:
        return _fail(str(exc))
    errors = validate_input(records)
    if errors:
        return _fail(f"invalid input ({len(errors)} problem(s)): {errors[0]}")
    if a.ids:
        wanted = set(json.loads(Path(a.ids).read_text(encoding="utf-8")))
        records = [r for r in records if r["encounter_id"] in wanted]
    if a.limit:
        records = records[: a.limit]

    run_date = date.fromisoformat(a.date) if a.date else date.today()
    out = output_paths(a.out_root, run_date)
    if not a.overwrite and any(p.exists() for p in out.all()):
        return _fail(f"output already exists (use --overwrite): {out.records}")

    cfg = make_config(
        a.preset,
        use_codes=not a.no_codes,
        detectors=detectors,
        explain_text=a.explain_text,
        device=a.device,
        features=_features(a.features),
        llm_model=a.detector_model,
    )
    t0 = time.perf_counter()
    an = Anonymizer(cfg)
    outs, stats, decisions = [], [], []
    expl = {"action": Counter(), "category": Counter(), "layer": Counter(), "keep_reason": Counter(), "with_reason": 0}
    vram_peak = 0.0
    for rec in records:
        try:
            if cfg.use_codes:
                anon, dec, st = an.anonymize(rec)
            else:
                anon, dec, st = an.anonymize({**rec, "icd10_codes": []})  # the pipeline never sees the codes
                anon["icd10_codes"] = rec["icd10_codes"]
        except Exception as exc:  # fail closed: one failed record stops the run, nothing is published
            return _fail(f"{rec['encounter_id']}: {type(exc).__name__}: {exc}; nothing was published")
        for d in dec:
            d["encounter_id"] = rec["encounter_id"]
            expl["action"][d["action"]] += 1
            expl["category"][d["category"]] += 1
            expl["layer"][d["source"].split(":")[0]] += 1
            expl["with_reason"] += bool(d.get("policy_reason")) and bool(d.get("detector_reason"))
            if d["action"] == "KEEP":
                expl["keep_reason"][d["policy_reason"].split(" (")[0]] += 1
        decisions += dec
        outs.append(anon)
        stats.append(st)
        vram_peak = max(vram_peak, an.vram_peak_mb)
    elapsed = time.perf_counter() - t0

    from .detect.encoders import place

    n_keep = sum(d["action"] == "KEEP" for d in decisions)
    llm_vram = ollama_vram_mb(cfg.llm_model) if "llm" in cfg.detectors else 0.0
    summary = {
        "tool": "unified-harness",
        "version": __version__,
        "run_date": run_date.isoformat(),
        "mode": "custom" if a.detectors else a.mode,
        "input": str(a.input),
        "input_sha256": _sha256(a.input),
        "config": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in cfg.__dict__.items()},
        "models": load_manifest() if any(d != "rules" for d in cfg.detectors) else {},
        "n_records": len(outs),
        "runtime_s": round(elapsed, 2),
        "ms_per_encounter": round(1000 * elapsed / max(1, len(outs)), 1),
        "peak_ram_mb": peak_rss_mb(),
        "peak_vram_mb": round(max(vram_peak, torch_vram_mb() + (llm_vram if llm_vram == llm_vram else 0.0)), 1),
        "vram_budget_mb": 4096,
        "vram_events": an.vram_events,
        "detectors": an.detector_stats(),
        "device_placement": place(cfg.detectors, cfg.device) if len(cfg.detectors) > 1 else {},
        "n_decisions": len(decisions),
        "n_edits": len(decisions) - n_keep,
        "n_kept_with_reason": n_keep,
        "explained_decisions": round(expl["with_reason"] / max(1, len(decisions)), 4),
        "explanations": {k: (dict(v.most_common()) if isinstance(v, Counter) else v) for k, v in expl.items()},
        "residual_leaks": sum(s["residual_leaks"] for s in stats),
        "shield_edit_overlaps": sum(s["shield"]["edit_overlaps"] for s in stats),
        "shields_overridden_by_identifier": sum(s["shield"].get("overridden_by_identifier", 0) for s in stats),
        "shield_spans": sum(s["shield"]["shield_spans"] for s in stats),
        "per_encounter": stats,
    }
    if summary["shield_edit_overlaps"]:
        return _fail(f"{summary['shield_edit_overlaps']} edit(s) overlap the medical shield; nothing was published")
    try:
        publish_run(records, outs, decisions, summary, out, overwrite=a.overwrite)
    except (ContractError, FileExistsError) as exc:
        return _fail(str(exc))
    print(
        f"anonymized {len(outs)} records in {elapsed:.1f}s ({summary['ms_per_encounter']} ms/encounter), "
        f"peak RAM {summary['peak_ram_mb']} MB, peak VRAM {summary['peak_vram_mb']} MB, {summary['n_edits']} edits, "
        f"{n_keep} explained keeps"
    )
    print("\n".join(f"-> {p}" for p in out.all()))
    return 0


# --------------------------------------------------------------------------- evaluation
def cmd_evaluate(a: argparse.Namespace) -> int:
    from .eval import report

    anon = read_jsonl(a.anonymized)
    ids = {r["encounter_id"] for r in anon}
    orig = [r for r in read_jsonl(a.original) if r["encounter_id"] in ids]
    dec = read_jsonl(a.decisions) if a.decisions and Path(a.decisions).exists() else None
    summ = json.loads(Path(a.summary).read_text()) if a.summary and Path(a.summary).exists() else None
    res = report.evaluate(
        orig, anon, Path(a.gold) if a.gold else None, dec, summ, icd=a.icd, name=a.name, llm_model=a.llm_model
    )
    report.save(res, Path(a.out_dir))
    print(report.table([res]))
    if res.get("schema_errors"):
        print("SCHEMA ERRORS:", res["schema_errors"][:5])
    if res.get("icd_error"):
        print("ICD proxy skipped:", res["icd_error"])
    return 0


def _split_paths(split: str) -> tuple[Path, Path | None, list[str] | None]:
    """(input file, silver-label directory or None, encounter ids or None) of a split name or a file path."""
    if split in paths.DATA_FILES:
        gold = paths.gold_dir(split)
        ids = paths.split_ids(split) if split == "dev" else None
        if split == "dev" and ids is None:
            raise SystemExit("error: configs/splits.json has no 'dev' ids")
        return paths.split_file(split), gold if gold.exists() else None, ids
    return Path(split), None, None


FEATURES = ("F0", "F2", "F4")


def _features(value: str | None) -> frozenset[str]:
    feats = frozenset(f.strip().upper() for f in (value or "").split(",") if f.strip())
    unknown = feats - set(FEATURES)
    if unknown:
        raise SystemExit(f"error: unknown feature(s) {sorted(unknown)}; available: {', '.join(FEATURES)}")
    return feats


def run_one(
    split: str,
    name: str,
    preset: str,
    no_codes: bool,
    detectors: str | None,
    icd: str,
    extra: list[str] | None = None,
    gold: str | None = None,
    llm_model: str | None = None,
    device: str = "auto",
    mode: str = "rules",
    features: str | None = None,
    detector_model: str | None = None,
) -> dict:
    """Anonymize one split in a separate process (clean peak-RAM measurement), then evaluate it."""
    inp, gold_dir, ids = _split_paths(split)
    if gold:
        gold_dir = Path(gold)
    run_root = paths.runs_dir() / name
    run_date = date.today()
    out = output_paths(run_root, run_date)
    cmd = [sys.executable, "-m", "unified_harness.cli", "anonymize", "--input", str(inp), "--out-root", str(run_root)]
    cmd += ["--date", run_date.isoformat(), "--overwrite", "--preset", preset, "--device", device, "--mode", mode]
    if no_codes:
        cmd.append("--no-codes")
    if detectors:
        cmd += ["--detectors", detectors]
    if features:
        cmd += ["--features", features]
    if detector_model:
        cmd += ["--detector-model", detector_model]
    if ids:
        ids_file = run_root / "ids.json"
        ids_file.parent.mkdir(parents=True, exist_ok=True)
        ids_file.write_text(json.dumps(ids), encoding="utf-8")
        cmd += ["--ids", str(ids_file)]
    cmd += extra or []
    print("$", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)
    ns = argparse.Namespace(
        original=str(inp),
        anonymized=str(out.records),
        gold=str(gold_dir) if gold_dir else None,
        decisions=str(out.decisions),
        summary=str(out.summary),
        icd=icd,
        name=name,
        out_dir=str(paths.eval_dir()),
        llm_model=llm_model,
    )
    cmd_evaluate(ns)
    return json.loads((paths.eval_dir() / f"{name}.json").read_text())


def _run_name(
    split: str,
    preset: str,
    no_codes: bool,
    mode: str,
    detectors: str | None,
    features: str | None,
    detector_model: str | None = None,
) -> str:
    det = detectors.replace(",", "+") if detectors else mode
    feats = "".join(f"_{f}" for f in sorted(_features(features)))
    model = "_" + re.sub(r"[^A-Za-z0-9.]+", "-", detector_model) if detector_model else ""
    return f"{split}_{preset}{'_nocodes' if no_codes else ''}_{det}{feats}{model}"


def cmd_run(a: argparse.Namespace) -> int:
    name = a.name or _run_name(a.split, a.preset, a.no_codes, a.mode, a.detectors, a.features, a.detector_model)
    run_one(
        a.split,
        name,
        a.preset,
        a.no_codes,
        a.detectors,
        a.icd,
        gold=a.gold,
        llm_model=a.llm_model,
        device=a.device,
        mode=a.mode,
        features=a.features,
        detector_model=a.detector_model,
    )
    return 0


SWEEP = [("conservative", False), ("balanced", False), ("aggressive", False), ("balanced", True)]


def cmd_sweep(a: argparse.Namespace) -> int:
    from .eval.report import table

    rows = []
    for preset, no_codes in SWEEP:
        name = "sweep_" + _run_name(a.split, preset, no_codes, a.mode, None, None)
        rows.append(run_one(a.split, name, preset, no_codes, None, a.icd, mode=a.mode))
    md = table(rows)
    out = paths.eval_dir() / f"sweep_{a.split}_{a.mode}.md"
    out.write_text(f"# Privacy / utility sweep on `{a.split}` ({a.mode} mode)\n\n{md}\n", encoding="utf-8")
    (paths.eval_dir() / f"sweep_{a.split}_{a.mode}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(md)
    print(f"-> {out}")
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    from .eval import gate

    cand, base = (json.loads(Path(f).read_text(encoding="utf-8")) for f in (a.candidate, a.baseline))
    checks = gate.compare(cand, base, floor=a.floor, icd_noise=a.icd_noise)
    print(gate.table(checks, cand.get("name", "candidate"), base.get("name", "baseline")))
    ok = all(c.passed for c in checks)
    print("\nGATE PASS" if ok else "\nGATE FAIL")
    return 0 if ok else 1


def cmd_icd_judge(a: argparse.Namespace) -> int:
    from .eval import icd_judge

    if a.step == "export":
        runs = dict(r.split("=", 1) for r in a.run)
        print(json.dumps(icd_judge.export(Path(a.original), {k: Path(v) for k, v in runs.items()}, Path(a.out_dir))))
    else:
        res = icd_judge.score(Path(a.out_dir), Path(a.rerun) if a.rerun else None)
        for k, v in res["arms"].items():
            change = f"  change {v['delta']} {v['delta_ci95']}" if "delta" in v else ""
            print(f"{k}: micro-F1 {v['micro_f1']} {v['f1_ci95']}{change}")
    return 0


def cmd_attack(a: argparse.Namespace) -> int:
    from .eval import attack

    return attack.main(a)


def cmd_robustness(a: argparse.Namespace) -> int:
    from .eval import robustness

    return robustness.main(a)


def cmd_show(a: argparse.Namespace) -> int:
    """Side-by-side view of one encounter with its decisions (debug view: prints original text)."""
    from .config import make_config
    from .pipeline import Anonymizer

    rec = next(r for r in read_jsonl(a.input) if r["encounter_id"] == a.id)
    cfg = make_config(a.preset, explain_text=True, use_codes=not a.no_codes, detectors=MODE_DETECTORS[a.mode])
    out, dec, _ = Anonymizer(cfg).anonymize(rec)
    for f in ("dialogue", "note"):
        print(f"===== {f} (anonymized)\n{out[f]}\n")
    print("===== decisions")
    for d in sorted(dec, key=lambda d: (d["field"], d["start"])):
        print(
            f"{d['field']:8s} {d['action']:10s} {d['category']:15s} {d.get('text', '')!r:28s} -> "
            f"{d.get('replacement') or '':14s} [{d['source']}] {d['policy_reason']}"
        )
    return 0


# --------------------------------------------------------------------------- environment and contract
def cmd_doctor(a: argparse.Namespace) -> int:
    report = doctor(a.mode, detector_model=a.detector_model)
    if a.json:
        print(json.dumps(report.as_dict(), indent=1))
    else:
        for c in report.checks:
            print(f"[{'ok' if c.ok else 'FAIL':>4}] {c.name}: {c.detail}")
        print(f"mode {a.mode}: {'ready' if report.ok else 'NOT ready'}")
    return 0 if report.ok else 1


def cmd_validate(a: argparse.Namespace) -> int:
    try:
        original, anonymized = read_jsonl(a.original), read_jsonl(a.anonymized)
    except InputError as exc:
        return _fail(str(exc))
    errors = validate_input(original) + validate_output(original, anonymized)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"{len(anonymized)} records checked, {len(errors)} contract violation(s)")
    return 1 if errors else 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified-harness", description="Offline, ICD-aware anonymization of patient-doctor encounters."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("anonymize", help="anonymize a JSONL file (dialogue + note); codes preserved")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--out-root", default=str(paths.outputs_dir()), help="results go to <out-root>/<date>/")
    p.add_argument("--date", help="run date for the output names (YYYY-MM-DD, default: today)")
    p.add_argument(
        "--mode",
        choices=MODES,
        default="llm",
        help="llm (default): rules + local Ollama model; rules: no ML; full: + PyTorch encoders",
    )
    p.add_argument("--preset", default="privacy", choices=PRESETS)
    p.add_argument("--no-codes", action="store_true", help="do not use icd10_codes at inference (bonus mode)")
    p.add_argument("--detectors", help="ablation only: comma list of rules, deid, gliner, llm (overrides --mode)")
    p.add_argument("--features", help="M5 feature flags, comma list: " + ", ".join(FEATURES))
    p.add_argument("--detector-model", help="Ollama model for the LLM detector (pinned in models.toml)")
    p.add_argument("--ids", help="JSON list of encounter ids to process")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="auto", help="encoders: auto (GPU within the 4 GB budget) | cpu | mps | cuda")
    p.add_argument("--overwrite", action="store_true", help="replace existing outputs of the same date")
    p.add_argument(
        "--explain-text", action="store_true", help="debug only: copy original text into the sidecar (contains PII)"
    )
    p.set_defaults(func=cmd_anonymize)

    p = sub.add_parser("evaluate", help="score an anonymized file (privacy, utility proxies, efficiency)")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.add_argument("--gold", help="silver-label directory (labels/upstream/eval)")
    p.add_argument("--decisions")
    p.add_argument("--summary")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.add_argument("--llm-model")
    p.add_argument("--name", default="run")
    p.add_argument("--out-dir", default=str(paths.eval_dir()))
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("run", help="anonymize a split and evaluate it (one row in outputs/eval/results.md)")
    p.add_argument("--split", default="dev", help="dev | eval | train | path to a JSONL file")
    p.add_argument("--mode", choices=MODES, default="rules")
    p.add_argument("--preset", default="balanced", choices=PRESETS)
    p.add_argument("--no-codes", action="store_true")
    p.add_argument("--detectors", help="ablation only (overrides --mode)")
    p.add_argument("--features", help="M5 feature flags, comma list: " + ", ".join(FEATURES))
    p.add_argument("--detector-model", help="Ollama model for the LLM detector (pinned in models.toml)")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.add_argument("--llm-model")
    p.add_argument("--gold")
    p.add_argument("--name")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="presets x codes/no-codes: the privacy/utility curve")
    p.add_argument("--split", default="dev")
    p.add_argument("--mode", choices=MODES, default="rules")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("attack", help="adversarial re-identification attacks on an anonymized file")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.add_argument("--gold", help="silver-label directory with the removed values")
    p.add_argument("--llm-model", default="qwen3.5:2b-q4_K_M")
    p.add_argument("--limit", type=int)
    p.add_argument("--name", default="attack")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--skip-ner", action="store_true")
    p.set_defaults(func=cmd_attack)

    p = sub.add_parser("robustness", help="build and evaluate the unseen-format robustness sets")
    p.add_argument("--build", action="store_true", help="(re)build the datasets under data/robustness/")
    p.add_argument("--preset", default="balanced")
    p.add_argument("--detectors")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.set_defaults(func=cmd_robustness)

    p = sub.add_parser("show", help="print one anonymized encounter with every decision (contains original text)")
    p.add_argument("--input", default=str(paths.split_file("eval")))
    p.add_argument("--id", required=True)
    p.add_argument("--mode", choices=MODES, default="rules")
    p.add_argument("--preset", default="balanced", choices=PRESETS)
    p.add_argument("--no-codes", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("doctor", help="check the environment for a run mode")
    p.add_argument("--mode", choices=MODES, default="llm")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.add_argument("--detector-model", help="Ollama model for the LLM detector (pinned in models.toml)")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("gate", help="acceptance gate: compare a scored run with the current best on one split")
    p.add_argument("candidate", help="result file of the new run (outputs/eval/<name>.json)")
    p.add_argument("baseline", help="result file of the current best run on the same split")
    p.add_argument("--floor", type=float, default=0.995, help="minimum keep-trap and concept recall")
    p.add_argument("--icd-noise", type=float, default=0.01, help="tolerated drop of ICD proxy agreement")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("icd-judge", help="external ICD judge hook: blinded export, then score its predictions")
    p.add_argument("step", choices=("export", "score"))
    p.add_argument("--out-dir", required=True, help="a new folder under outputs/")
    p.add_argument("--original", help="export: the input split file")
    p.add_argument("--run", action="append", default=[], metavar="NAME=PATH", help="export: an anonymized file")
    p.add_argument("--rerun", help="score: an export of the original notes coded again (noise control)")
    p.set_defaults(func=cmd_icd_judge)

    p = sub.add_parser("validate", help="check an anonymized file against its input")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
