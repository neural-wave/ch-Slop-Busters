"""Score finished runs: every metric for every (model, suite) that has an output.

Usage: python3 -I eval score [--models ...] [--suites ...] [--icd llm_note knn ... "cmd:<command>"]

`--icd` takes coder names from eval/coders.toml (default: every enabled coder with default = true) or an ad-hoc
"cmd:<command>". The LLM coders need a running Ollama server; their predictions are cached by text, so a re-score
only codes new texts. See eval/icd.py for the ICD block of a score file.

Writes `data/eval/scores/<model>/<suite>.json`. The leak and loss lists in it quote the gold, so they contain PII:
keep them under data/eval/ (gitignored) and share only the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import icd
from eval import metrics as M
from eval import suites as S
from eval.bonus import explain
from eval.common import RUNS_DIR, SCORES_DIR, SUITES_DIR, load_jsonl, write_json
from eval.run import load_models

DEFAULT_ICD = icd.default_names()  # every enabled coder of coders.toml with default = true


def coder_name(name: str) -> str:
    """--icd value: a coders.toml entry or "cmd:<command>"."""
    reg = icd.load_registry()
    if name.startswith("cmd:") or name in reg:
        return name
    raise argparse.ArgumentTypeError(f"unknown coder {name!r} (coders.toml: {', '.join(reg)}, or cmd:<command>)")


def score_run(model: str, suite: str, predictors: list[str]) -> dict | None:
    run_dir = RUNS_DIR / model / suite
    if not (run_dir / "run.json").exists():
        return None
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    out = {"model": model, "suite": suite, "run": run}
    if run["exit_code"] != 0 or not (run_dir / "output.jsonl").exists():
        out["error"] = f"run failed (exit {run['exit_code']}), see {run_dir / 'run.log'}"
        write_json(SCORES_DIR / model / f"{suite}.json", out)
        return out
    sdir = SUITES_DIR / suite
    original = load_jsonl(sdir / "encounters.jsonl")
    labels = {g["encounter_id"]: g for g in load_jsonl(sdir / "labels.jsonl")}
    try:
        anonymized = load_jsonl(run_dir / "output.jsonl")
    except json.JSONDecodeError as e:
        out["error"] = f"output is not valid JSONL: {e}"
        write_json(SCORES_DIR / model / f"{suite}.json", out)
        return out
    out["validity"] = M.validity(original, anonymized)
    out["privacy"] = M.privacy(original, anonymized, labels, "pii")  # consensus level + levels + by_source (merged gold)
    out["keep"] = M.survival(original, anonymized, labels, "keep", "category")
    out["concepts"] = M.survival(original, anonymized, labels, "concepts", "type")
    out["edits"] = M.edits(original, anonymized, labels)
    out["terms"] = M.terms(original, anonymized, labels)
    out["placeholders_per_1k_words"] = M.placeholder_rate(anonymized)
    out["explain"] = explain.score(run_dir, original, anonymized)
    out["icd"] = {}
    for name in predictors:  # a coder only scores the suites of its coders.toml entry; a failed coder gives {"error"}
        if (res := icd.evaluate(name, suite, original, anonymized)) is not None:
            out["icd"][res[0]] = res[1]
    write_json(SCORES_DIR / model / f"{suite}.json", out)
    return out


def main(argv: list[str] | None = None) -> int:
    models = load_models()
    p = argparse.ArgumentParser(prog="python3 -I eval score", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(models), choices=list(models))
    p.add_argument("--suites", nargs="+", default=list(S.SUITES), choices=list(S.SUITES))
    p.add_argument("--icd", nargs="+", type=coder_name, default=DEFAULT_ICD,
                   help=f'ICD coders of coders.toml (default: {" ".join(DEFAULT_ICD)}) or "cmd:<command with {{input}} {{output}}>"')
    a = p.parse_args(argv)
    registry = icd.load_registry()
    for name in [n for n in a.icd if not n.startswith("cmd:") and not registry[n].get("enabled", True)]:
        print(f"{name}: disabled in coders.toml, skipped")
        a.icd.remove(name)
    for m in a.models:
        for s in a.suites:
            res = score_run(m, s, a.icd)
            if res is None:
                continue
            if "error" in res:
                print(f"{m:18s} {s:15s} {res['error']}")
                continue
            pv = res["privacy"] or {}
            print(f"{m:18s} {s:15s} valid {res['validity']['valid_rate']}  privacy {pv.get('removed')}  "
                  f"keep {(res['keep'] or {}).get('kept')}  concepts {(res['concepts'] or {}).get('kept')}  terms {(res['terms'] or {}).get('kept')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
