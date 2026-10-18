"""Command line: `neural-wave anonymize | evaluate | run | sweep | attack | robustness | show`."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/runs"
EVAL_OUT = ROOT / "outputs/eval"


def _detectors(s: str | None) -> tuple[str, ...] | None:
    return tuple(x.strip() for x in s.split(",") if x.strip()) if s else None


def cmd_anonymize(a: argparse.Namespace) -> int:
    from .config import make_config
    from .eval.resources import ollama_vram_mb, peak_rss_mb, torch_vram_mb
    from .io import iter_jsonl, read_jsonl, validate_output, write_jsonl
    from .pipeline import Anonymizer

    cfg = make_config(a.preset, use_codes=not a.no_codes, detectors=_detectors(a.detectors), explain_text=a.explain_text, device=a.device)
    out_path = Path(a.output)
    dec_path = Path(a.decisions) if a.decisions else out_path.with_suffix(".decisions.jsonl")
    sum_path = Path(a.summary) if a.summary else out_path.with_suffix(".summary.json")
    ids = set(json.loads(Path(a.ids).read_text())) if a.ids else None
    t0 = time.perf_counter()
    an = Anonymizer(cfg)
    outs, stats, n_dec, kept, vram_peak = [], [], 0, 0, 0.0
    from collections import Counter

    expl = {"action": Counter(), "category": Counter(), "layer": Counter(), "keep_reason": Counter(), "with_reason": 0}
    originals = []
    dec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dec_path, "w", encoding="utf-8") as dfh:
        for i, rec in enumerate(iter_jsonl(a.input)):
            if ids is not None and rec.get("encounter_id") not in ids:
                continue
            if a.limit is not None and len(outs) >= a.limit:
                break
            if not cfg.use_codes:
                rec_in = dict(rec)
                rec_in.pop("icd10_codes", None)  # the pipeline never sees the codes in this mode
                out, dec, st = an.anonymize(rec_in)
                out = {k: (rec[k] if k == "icd10_codes" else out.get(k, rec[k])) for k in rec}  # same keys/order as input
            else:
                out, dec, st = an.anonymize(rec)
            for d in dec:
                d["encounter_id"] = rec.get("encounter_id", "")
                dfh.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_dec += 1
                kept += d["action"] == "KEEP"
                expl["action"][d["action"]] += 1
                expl["category"][d["category"]] += 1
                expl["layer"][d["source"].split(":")[0]] += 1
                expl["with_reason"] += bool(d.get("policy_reason")) and bool(d.get("detector_reason"))
                if d["action"] == "KEEP":
                    expl["keep_reason"][d["policy_reason"].split(" (")[0]] += 1
            originals.append(rec)
            outs.append(out)
            stats.append(st)
            vram_peak = max(vram_peak, an.vram_peak_mb)
    write_jsonl(out_path, outs)
    elapsed = time.perf_counter() - t0
    errors = validate_output(originals, outs)
    summary = {
        "input": str(a.input),
        "output": str(out_path),
        "config": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in cfg.__dict__.items()},
        "n_records": len(outs),
        "runtime_s": round(elapsed, 2),
        "ms_per_encounter": round(1000 * elapsed / max(1, len(outs)), 1),
        "peak_ram_mb": peak_rss_mb(),
        "peak_vram_mb": round(max(vram_peak, torch_vram_mb() + (ollama_vram_mb(cfg.llm_model) if "llm" in cfg.detectors else 0.0)), 1),
        "vram_budget_mb": 4096,
        "vram_events": an.vram_events,
        "device_placement": {"rules": "cpu", **{n: getattr(det, "device", "cpu") for n, det in an._models.items()}},
        "n_decisions": n_dec,
        "n_edits": n_dec - kept,
        "n_kept_with_reason": kept,
        "explained_decisions": round(expl["with_reason"] / max(1, n_dec), 4),
        "explanations": {k: (dict(v.most_common()) if isinstance(v, Counter) else v) for k, v in expl.items()},
        "residual_leaks": sum(s["residual_leaks"] for s in stats),
        "shield_edit_overlaps": sum(s["shield"]["edit_overlaps"] for s in stats),
        "shield_spans": sum(s["shield"]["shield_spans"] for s in stats),
        "schema_errors": errors,
        "per_encounter": stats,
    }
    Path(sum_path).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"anonymized {len(outs)} records in {elapsed:.1f}s ({summary['ms_per_encounter']} ms/encounter), "
          f"peak RAM {summary['peak_ram_mb']} MB, peak VRAM {summary['peak_vram_mb']} MB, {summary['n_edits']} edits, "
          f"{kept} explained keeps, schema errors: {len(errors)}, shield overlaps: {summary['shield_edit_overlaps']}")
    print(f"-> {out_path}\n-> {dec_path}\n-> {sum_path}")
    return 1 if errors else 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from .eval import report
    from .io import read_jsonl

    anon = read_jsonl(a.anonymized)
    orig = read_jsonl(a.original)
    if getattr(a, "ids", None):  # explicit subset (dev split); otherwise a dropped record is a schema error
        keep = set(json.loads(Path(a.ids).read_text()))
        orig = [r for r in orig if r["encounter_id"] in keep]
    dec = read_jsonl(a.decisions) if a.decisions and Path(a.decisions).exists() else None
    summ = json.loads(Path(a.summary).read_text()) if a.summary and Path(a.summary).exists() else None
    res = report.evaluate(orig, anon, Path(a.gold) if a.gold else None, dec, summ, icd=a.icd, name=a.name, llm_model=a.llm_model)
    report.save(res, Path(a.out_dir))
    print(report.table([res]))
    if res.get("schema_errors"):
        print("SCHEMA ERRORS:", res["schema_errors"][:5])
    if res.get("icd_error"):
        print("ICD proxy skipped:", res["icd_error"])
    return 0


def _split_paths(split: str) -> tuple[Path, Path | None, Path | None]:
    from .eval.report import SPLIT_FILES

    if split in SPLIT_FILES:
        gold = ROOT / f"outputs/gold/{split}"
        ids = ROOT / "outputs/gold/dev_ids.json" if split == "dev" else None
        return SPLIT_FILES[split], gold if gold.exists() else None, ids
    return Path(split), None, None


def run_one(split: str, name: str, preset: str, no_codes: bool, detectors: str | None, icd: str, extra: list[str] | None = None, gold: str | None = None, llm_model: str | None = None, device: str = "auto") -> dict:
    inp, gold_dir, ids = _split_paths(split)
    if gold:
        gold_dir = Path(gold)
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "neural_wave.cli", "anonymize", "-i", str(inp), "-o", str(d / "anonymized.jsonl"), "--preset", preset, "--explain-text", "--device", device]
    if no_codes:
        cmd.append("--no-codes")
    if detectors:
        cmd += ["--detectors", detectors]
    if ids:
        cmd += ["--ids", str(ids)]
    cmd += extra or []
    print("$", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)  # separate process: clean peak-RAM measurement
    ns = argparse.Namespace(
        original=str(inp), anonymized=str(d / "anonymized.jsonl"), gold=str(gold_dir) if gold_dir else None,
        decisions=str(d / "anonymized.decisions.jsonl"), summary=str(d / "anonymized.summary.json"),
        icd=icd, name=name, out_dir=str(EVAL_OUT), llm_model=llm_model, ids=str(ids) if ids else None,
    )
    cmd_evaluate(ns)
    return json.loads((EVAL_OUT / f"{name}.json").read_text())


def cmd_run(a: argparse.Namespace) -> int:
    from .eval.report import SPLIT_FILES

    base = a.split if a.split in SPLIT_FILES else Path(a.split).stem  # a path never escapes outputs/
    name = a.name or f"{base}_{a.preset}{'_nocodes' if a.no_codes else ''}{'_' + a.detectors.replace(',', '+') if a.detectors else ''}"
    name = re.sub(r"[^A-Za-z0-9_+.-]", "_", name)
    run_one(a.split, name, a.preset, a.no_codes, a.detectors, a.icd, gold=a.gold, llm_model=a.llm_model, device=a.device)
    return 0


SWEEP = [
    ("conservative", False, None),
    ("balanced", False, None),
    ("aggressive", False, None),
    ("balanced", True, None),
]


def cmd_sweep(a: argparse.Namespace) -> int:
    from .eval.report import table

    configs = SWEEP
    if a.detectors:
        configs = configs + [(p, nc, a.detectors) for p, nc, _ in SWEEP]
    rows = []
    for preset, nc, det in configs:
        name = f"sweep_{a.split}_{preset}{'_nocodes' if nc else ''}{'_' + det.replace(',', '+') if det else ''}"
        rows.append(run_one(a.split, name, preset, nc, det, a.icd))
    md = table(rows)
    out = EVAL_OUT / f"sweep_{a.split}.md"
    out.write_text(f"# Privacy / utility sweep on `{a.split}`\n\n{md}\n", encoding="utf-8")
    (EVAL_OUT / f"sweep_{a.split}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(md)
    print(f"-> {out}")
    return 0


def cmd_attack(a: argparse.Namespace) -> int:
    from .eval import attack

    return attack.main(a)


def cmd_robustness(a: argparse.Namespace) -> int:
    from .eval import robustness

    return robustness.main(a)


def cmd_show(a: argparse.Namespace) -> int:
    """Side-by-side view of one encounter with its decisions (for explanations / demos)."""
    from .config import make_config
    from .io import read_jsonl
    from .pipeline import Anonymizer

    from .eval.report import SPLIT_FILES

    pool = read_jsonl(a.input) if a.input else [r for f in dict.fromkeys(SPLIT_FILES.values()) for r in read_jsonl(f)]
    rec = next((r for r in pool if r["encounter_id"] == a.id), None)
    if rec is None:
        print(f"encounter {a.id!r} not found", file=sys.stderr)
        return 2
    out, dec, st = Anonymizer(make_config(a.preset, explain_text=True, use_codes=not a.no_codes)).anonymize(rec)
    for f in ("dialogue", "note"):
        print(f"===== {f} (anonymized)\n{out[f]}\n")
    print("===== decisions")
    for d in sorted(dec, key=lambda d: (d["field"], d["start"])):
        print(f"{d['field']:8s} {d['action']:10s} {d['category']:15s} {d.get('text', '')!r:28s} -> {d.get('replacement') or '':14s} [{d['source']}] {d['policy_reason']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neural-wave", description="Shield-first, code-aware anonymization of patient-doctor encounters")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("anonymize", help="anonymize a JSONL file (dialogue + note); codes preserved")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--preset", default="balanced", choices=["conservative", "balanced", "aggressive"])
    p.add_argument("--no-codes", action="store_true", help="do not use icd10_codes at inference (bonus mode)")
    p.add_argument("--detectors", help="comma list: rules (default), deid, gliner, llm")
    p.add_argument("--decisions", help="explanations sidecar (default: <output>.decisions.jsonl)")
    p.add_argument("--summary", help="run summary JSON (default: <output>.summary.json)")
    p.add_argument("--explain-text", action="store_true", help="include original surface text in the sidecar (debug; contains PII)")
    p.add_argument("--ids", help="JSON list of encounter ids to process")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="auto", help="transformer detectors: auto (GPU if the 4 GB VRAM budget allows) | cpu | mps | cuda")
    p.set_defaults(fn=cmd_anonymize)

    p = sub.add_parser("evaluate", help="score an anonymized file (privacy vs gold, utility proxies, efficiency)")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.add_argument("--gold", help="gold directory (outputs/gold/eval)")
    p.add_argument("--decisions")
    p.add_argument("--summary")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.add_argument("--llm-model")
    p.add_argument("--name", default="run")
    p.add_argument("--ids", help="JSON list of encounter ids to score (default: all original records)")
    p.add_argument("--out-dir", default=str(EVAL_OUT))
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("run", help="anonymize a split and evaluate it (one row in outputs/eval/results.md)")
    p.add_argument("--split", default="eval", help="eval | dev | train | path to a JSONL file")
    p.add_argument("--preset", default="balanced", choices=["conservative", "balanced", "aggressive"])
    p.add_argument("--no-codes", action="store_true")
    p.add_argument("--detectors")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.add_argument("--llm-model")
    p.add_argument("--gold")
    p.add_argument("--name")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("sweep", help="presets x codes/no-codes: the privacy/utility curve")
    p.add_argument("--split", default="eval")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.add_argument("--detectors", help="also run balanced with these detectors")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("attack", help="adversarial re-identification attacks on an anonymized file")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.add_argument("--gold", help="gold directory with the removed values")
    p.add_argument("--llm-model", default="qwen3.5:2b-q4_K_M")
    p.add_argument("--limit", type=int)
    p.add_argument("--name", default="attack")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--skip-ner", action="store_true")
    p.set_defaults(fn=cmd_attack)

    p = sub.add_parser("robustness", help="build and evaluate the unseen-format robustness sets")
    p.add_argument("--build", action="store_true", help="(re)build the datasets under outputs/robustness/")
    p.add_argument("--preset", default="balanced", choices=["conservative", "balanced", "aggressive"])
    p.add_argument("--detectors")
    p.add_argument("--icd", default="embed", choices=["none", "embed", "llm"])
    p.set_defaults(fn=cmd_robustness)

    p = sub.add_parser("show", help="print one anonymized encounter with every decision")
    p.add_argument("--input", help="JSONL file (default: search the train and eval files)")
    p.add_argument("--id", required=True)
    p.add_argument("--preset", default="balanced", choices=["conservative", "balanced", "aggressive"])
    p.add_argument("--no-codes", action="store_true")
    p.set_defaults(fn=cmd_show)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
