"""No-codes bonus: each model run without ICD-10 codes at inference beside the same model with codes.

Usage: python3 -I eval/bonus/nocodes.py [--suite aci_eval]   (prints the pairs from data/eval/scores/)

The bonus task asks for a model that works without the codes. `blind_codes = true` in models.toml hides them from
the model (the runner puts them back before scoring). Such an entry is paired with its base model: the entry named
in its `base` key, else the entry named without the `_nocodes` suffix (`ad_night_nocodes` -> `ad_night`). An entry
without an existing base is left out.

`rows()` gives, per pair, the base value, the no-codes value and the delta (no-codes minus base) of privacy (the
headline consensus level, `privacy.removed`), high-risk privacy, keep traps, concepts, ICD concepts and the
anonymized-text ICD F1 of every coder. The report renders them as "without ICD-10 codes at inference".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import SCORES_DIR
from eval.run import load_models

SUFFIX = "_nocodes"
METRICS = {
    "privacy": ("privacy", "removed"), "privacy_high": ("privacy", "removed_high_risk"), "keep": ("keep", "kept"),
    "concepts": ("concepts", "kept"), "icd_concepts": ("concepts", "breakdown", "icd_relevant"),
}  # fmt: skip


def pairs(registry: dict) -> list[tuple[str, str]]:
    """(base, nocodes) model pairs in registry order."""
    out = []
    for name, spec in registry.items():
        if spec.get("blind_codes"):
            base = spec.get("base") or (name.removesuffix(SUFFIX) if name.endswith(SUFFIX) else None)
            if base and base != name and base in registry:
                out.append((base, name))
    return out


def _get(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def _icd_f1(res: dict) -> float | None:
    """Code-level micro F1 on the anonymized text (score files from before the coder registry: `f1_anonymized`)."""
    return _get(res, "anonymized", "code", "f1") if "anonymized" in res else _get(res, "f1_anonymized")


def values(sc: dict | None) -> dict[str, float | None]:
    """The paired metrics of one score file, `icd_f1:<coder>` for every coder without an error."""
    if not sc or "error" in sc:
        return {}
    out = {k: _get(sc, *path) for k, path in METRICS.items()}
    for coder, res in (sc.get("icd") or {}).items():
        if isinstance(res, dict) and "error" not in res:
            out[f"icd_f1:{coder}"] = _icd_f1(res)
    return out


def rows(scores: dict, registry: dict, suite: str = "aci_eval") -> list[dict]:
    """One row per pair: {"base", "nocodes", "suite", "metrics": {metric: {"base", "nocodes", "delta"}}}; `scores` is
    {model: {suite: score}}. A value is None when the model has no usable score; the delta when either value is."""
    out = []
    for base, blind in pairs(registry):
        v0, v1 = values(scores.get(base, {}).get(suite)), values(scores.get(blind, {}).get(suite))
        keys = list(METRICS) + [k for k in dict.fromkeys([*v0, *v1]) if k not in METRICS]
        metrics = {}
        for k in keys:
            a, b = v0.get(k), v1.get(k)
            metrics[k] = {"base": a, "nocodes": b, "delta": None if a is None or b is None else round(b - a, 4)}
        out.append({"base": base, "nocodes": blind, "suite": suite, "metrics": metrics})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval/bonus/nocodes.py", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--suite", default="aci_eval")
    a = p.parse_args(argv)
    scores: dict[str, dict[str, dict]] = {}
    for path in sorted(SCORES_DIR.glob(f"*/{a.suite}.json")):
        scores.setdefault(path.parent.name, {})[a.suite] = json.loads(path.read_text(encoding="utf-8"))
    for r in rows(scores, load_models(), a.suite):
        print(f"{r['nocodes']} vs {r['base']} ({r['suite']})")
        for k, m in r["metrics"].items():
            print(f"  {k:22s} {m['base']!s:>8s} -> {m['nocodes']!s:>8s}  delta {m['delta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
