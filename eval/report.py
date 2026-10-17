"""Collect every score into one comparison: report.md, report.json, the CSV exports, tradeoff.svg, report_discord.md.

Usage: python3 -I eval report

Reads only data/eval/scores/<model>/<suite>.json, runs/*/*/run.json, attacks/<model>/<suite>.json, drift.json,
suites/*/suite.json, gold/aci_eval/merged_stats.json, eval/models.toml and eval/coders.toml. Data may be partial: a
missing value renders as "–", a failed run as "failed", a failed coder or attack as "error", and nothing crashes.

Writes into data/eval/ (aggregate numbers only, no leaked strings, so every file is safe to paste into a chat or a PR):

  report.md             every table; the headline is aci_eval, identity and oracle at the bottom as bounds
  report.json           the same numbers, plus "exports" (the files below)
  results.csv           long format, one number per row: model, suite, category, metric, value, n, ci_low, ci_high,
                        coder, level, notes (UTF-8 with BOM, comma). Every number of report.md is a row, plus the
                        per-category/field/language breakdowns the markdown leaves out. Full stored precision.
  results_headline.csv  wide: one row per model, one column per headline metric
  tradeoff.svg          consensus privacy vs ICD F1 retained (primary coder of coders.toml; ICD concepts kept while
                        no model has a result from it), Pareto-optimal models filled, identity and oracle as squares
  report_discord.md     the headline only, short enough for a chat message

results.csv rows: `category` is the result family (privacy, privacy_by_category, privacy_by_field,
privacy_by_language, keep, concepts, edits, terms, icd, attack, robustness, drift, efficiency, explain, validity).
`level` is the privacy gold level (`consensus` = `privacy.removed`, the headline; `union`, `direct`; `gold:<source>` =
one source gold on its own list), or a breakdown key; `coder` the ICD coder. Suite-level rows (drift, gold counts)
have an empty model. `n` is the denominator (occurrences, items, records, tokens) when the score file has it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple
from xml.sax.saxutils import escape

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import icd
from eval import suites as S
from eval.bonus import nocodes
from eval.common import CATEGORY_RISK, HERE, OUT, REPO, write_json
from eval.run import VRAM_BUDGET_MB, load_models

MODELS_TOML = HERE / "models.toml"
CODERS_TOML = HERE / "coders.toml"
REFERENCE = ("identity", "oracle")
HEAD = "aci_eval"  # the headline suite
MAIN = ("aci_eval", "aci_dev")
DASH = "–"
CSV_FIELDS = ("model", "suite", "category", "metric", "value", "n", "ci_low", "ci_high", "coder", "level", "notes")
CONSENSUS = "consensus"


class Col(NamedTuple):
    key: str
    header: str
    category: str
    metric: str
    level: str = ""
    primary: bool = False  # the metric of the primary ICD coder
    digits: int | str = 3  # "bool": a yes/no cell (1/0 in the CSV)


HEADLINE = (
    Col("privacy", "privacy (consensus)", "privacy", "removed", CONSENSUS),
    Col("direct", "direct", "privacy", "removed", "direct"),
    Col("entity", "entity-level", "privacy", "entity_removed", CONSENSUS),
    Col("privacy_harness", "privacy (harness gold)", "privacy", "removed", "gold:harness"),
    Col("concepts", "concepts kept", "concepts", "kept"),
    Col("icd_concepts", "ICD concepts kept", "concepts", "icd_relevant"),
    Col("icd_f1_anon", "ICD F1 anon", "icd", "anonymized.code.f1", primary=True),
    Col("icd_f1_retained", "ICD F1 retained", "icd", "f1_retention", primary=True),
    Col("icd_agree", "ICD agree", "icd", "agreement.code.f1", primary=True),
    Col("ms_per_encounter", "ms/enc", "efficiency", "ms_per_encounter", digits=1),
    Col("peak_ram_mb", "peak RAM MB", "efficiency", "peak_ram_mb", digits=1),
    Col("vram_ok", f"peak VRAM ≤ {VRAM_BUDGET_MB} MB", "efficiency", "within_vram_budget", digits="bool"),
    Col("explained", "explained", "explain", "explained"),
    Col("attack_max", "attack recovered (max)", "attack", "max_recovered"),
)


# --- reading -------------------------------------------------------------------------------------------------------


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or d.get(p) is None:
            return default
        d = d[p]
    return d


def num(x) -> float | int | None:
    """A finite number (bool as 1/0), else None."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and math.isfinite(x):
        return x
    return None


def short(msg, limit: int = 120) -> str:
    s = " ".join(str(msg).split()).replace("|", "/")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:  # a file being written right now, or damaged: skip it
        print(f"report: skipped {path} ({short(e, 80)})", file=sys.stderr)
        return None


def load_tree(base: Path) -> dict[str, dict[str, dict]]:
    """{model: {suite: json}} from <base>/<model>/<suite>.json."""
    out: dict[str, dict[str, dict]] = {}
    for p in sorted(base.glob("*/*.json")):
        if isinstance(d := read_json(p), dict):
            out.setdefault(p.parent.name, {})[p.stem] = d
    return out


def _safe(fn, default, what: str):
    try:
        return fn()
    except (OSError, ValueError) as e:  # tomllib.TOMLDecodeError is a ValueError
        print(f"report: cannot read {what} ({short(e, 80)})", file=sys.stderr)
        return default


class Data:
    """Everything the report reads, loaded once."""

    def __init__(self) -> None:
        self.registry: dict = _safe(lambda: load_models(MODELS_TOML), {}, MODELS_TOML.name)
        self.coders: dict = _safe(lambda: icd.load_registry(CODERS_TOML), {}, CODERS_TOML.name)
        self.primary: str | None = icd.primary(self.coders) if self.coders else None
        self.scores = load_tree(OUT / "scores")
        self.attacks = load_tree(OUT / "attacks")
        self.runs: dict[str, dict[str, dict]] = {}
        for p in sorted((OUT / "runs").glob("*/*/run.json")):
            if isinstance(r := read_json(p), dict):
                self.runs.setdefault(p.parent.parent.name, {})[p.parent.name] = r
        drift = read_json(OUT / "drift.json")
        self.drift: dict = drift if isinstance(drift, dict) else {}
        self.info: dict[str, dict] = {}
        for p in sorted((OUT / "suites").glob("*/suite.json")):
            if isinstance(i := read_json(p), dict):
                self.info[p.parent.name] = i
        gold = read_json(OUT / "gold" / "aci_eval" / "merged_stats.json")
        self.gold: dict = gold if isinstance(gold, dict) else {}
        present = set(self.info) | {s for per in (self.scores, self.attacks, self.runs) for m in per.values() for s in m}
        self.suites = [s for s in S.SUITES if s in present] + sorted(present - set(S.SUITES))
        seen = set(self.scores) | set(self.runs) | set(self.attacks)
        names = [m for m, spec in self.registry.items() if m in seen or spec.get("enabled", True)]
        self.models = [m for m in names if m not in REFERENCE] + [m for m in REFERENCE if m in names]
        self.pending = [m for m, spec in self.registry.items() if not spec.get("enabled", True) and m not in seen]
        found = sorted({c for per in self.scores.values() for sc in per.values() for c in (g(sc, "icd") or {})} - set(self.coders))
        self.coder_names = ([self.primary] if self.primary else []) + [c for c in self.coders if c != self.primary] + found

    def role(self, s: str) -> str:
        return (self.info.get(s) or {}).get("role") or S.SUITES.get(s, {}).get("role", "")

    def suites_of(self, *roles: str) -> list[str]:
        return [s for s in self.suites if self.role(s) in roles]

    def score(self, m: str, s: str) -> dict | None:
        return self.scores.get(m, {}).get(s)

    def failed(self, m: str, s: str) -> bool:
        return bool((self.score(m, s) or {}).get("error"))

    def run(self, m: str, s: str) -> dict | None:
        """The run behind the score (its `run` block), else runs/<m>/<s>/run.json; only a finished, successful run."""
        r = g(self.score(m, s), "run")
        if not isinstance(r, dict) or r.get("wall_s") is None:
            r = self.runs.get(m, {}).get(s)
        return r if isinstance(r, dict) and r.get("exit_code") == 0 and num(r.get("wall_s")) is not None else None


# --- the long table ------------------------------------------------------------------------------------------------


class Results:
    """The rows of results.csv, with a lookup by (model, suite, category, metric, coder, level). First row wins."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.index: dict[tuple, dict] = {}

    def add(self, model, suite, category, metric, value, n=None, ci=None, coder="", level="", notes="") -> None:
        value = num(value)
        if value is None and not notes:
            return
        key = (model, suite, category, metric, coder or "", level or "")
        if key in self.index:
            return
        lo, hi = (num(ci[0]), num(ci[1])) if isinstance(ci, (list, tuple)) and len(ci) == 2 else (None, None)
        row = {"model": model, "suite": suite, "category": category, "metric": metric, "value": value,
               "n": num(n) if not isinstance(n, bool) else None, "ci_low": lo, "ci_high": hi, "coder": coder or "",
               "level": level or "", "notes": notes}  # fmt: skip
        self.rows.append(row)
        self.index[key] = row

    def row(self, model, suite, category, metric, coder="", level="") -> dict | None:
        return self.index.get((model, suite, category, metric, coder or "", level or ""))

    def get(self, model, suite, category, metric, coder="", level=""):
        r = self.row(model, suite, category, metric, coder, level)
        return r["value"] if r else None


def total(counts: dict, key: str):
    c = (counts or {}).get(key)
    return c[1] if isinstance(c, list) and len(c) == 2 else None


SUMMARY = ("removed", "removed_high_risk", "entity_removed")


def privacy_rows(R: Results, m: str, s: str, pv: dict) -> None:
    levels, bd, counts = pv.get("levels") or {}, pv.get("breakdown") or {}, pv.get("counts") or {}
    cons = levels.get(CONSENSUS) or {}
    n_occ = total(counts, "ALL") or cons.get("occurrences")
    for k in SUMMARY:  # the classic keys are the consensus level (every level, on a suite without vote metadata)
        R.add(m, s, "privacy", k, pv.get(k), n=cons.get("items") if k == "entity_removed" else (total(counts, "risk:high") if k == "removed_high_risk" else n_occ),
              level=CONSENSUS)  # fmt: skip
    for lv, d in levels.items():
        for k in SUMMARY:
            R.add(m, s, "privacy", k, g(d, k), n=g(d, "items" if k == "entity_removed" else "occurrences"), level=lv)
    for src, d in (pv.get("by_source") or {}).items():
        for k in SUMMARY:
            R.add(m, s, "privacy", k, g(d, k), n=g(d, "items" if k == "entity_removed" else "occurrences"), level=f"gold:{src}")
    for k, v in bd.items():
        kind, _, name = k.partition(":")
        fam = {"cat": "privacy_by_category", "field": "privacy_by_field", "lang": "privacy_by_language"}.get(kind)
        if fam:
            R.add(m, s, fam, name, v, n=total(counts, k), level=CONSENSUS)
        elif kind in ("risk", "votes", "src"):  # votes:* and src:* are over the union items
            R.add(m, s, "privacy", k, v, n=total(counts, k), level="union" if kind != "risk" and levels else CONSENSUS)
        # ALL, level:*, entity:*, items_fully_removed repeat the rows above


def icd_rows(R: Results, m: str, s: str, coder: str, res) -> None:
    if not isinstance(res, dict):
        return
    if res.get("error"):
        R.add(m, s, "icd", "error", None, coder=coder, notes=f"error: {short(res['error'])}")
        return
    n = res.get("n_records")
    for arm in ("original", "anonymized", "agreement"):
        for unit in ("code", "category"):
            d = g(res, arm, unit) or {}
            for stat in ("f1", "precision", "recall", "mean_f1"):
                R.add(m, s, "icd", f"{arm}.{unit}.{stat}", d.get(stat), n=n, ci=d.get("ci95") if stat == "f1" else None, coder=coder)
    for old, new in (("f1_original", "original.code.f1"), ("f1_anonymized", "anonymized.code.f1"), ("agreement", "agreement.code.f1")):
        if num(res.get(old)) is not None:  # score files from before the coder registry
            R.add(m, s, "icd", new, res[old], n=n, coder=coder, notes="old score format")
    R.add(m, s, "icd", "f1_delta", g(res, "f1_delta", "value"), n=n, ci=g(res, "f1_delta", "ci95"), coder=coder,
          notes="anonymized minus original, paired bootstrap CI95")  # fmt: skip
    R.add(m, s, "icd", "f1_retention", res.get("f1_retention"), n=n, coder=coder)
    errs = [num(g(res, "errors", a)) for a in ("original", "anonymized")]
    for a, e in zip(("original", "anonymized"), errs, strict=True):
        R.add(m, s, "icd", f"errors.{a}", e, coder=coder)
        R.add(m, s, "icd", f"invalid_codes.{a}", g(res, "invalid_codes", a), coder=coder)
    if any(e is not None for e in errs):
        R.add(m, s, "icd", "errors", sum(e or 0 for e in errs), coder=coder, notes="failed predictions, both arms")
    R.add(m, s, "icd", "valid_for_comparison", res.get("valid_for_comparison"), coder=coder)
    R.add(m, s, "icd", "n_codes", res.get("n_codes"), n=n, coder=coder)


def score_rows(R: Results, m: str, s: str, sc: dict) -> None:
    if sc.get("error"):
        R.add(m, s, "validity", "error", None, notes="run failed or output unreadable")
        return
    v = sc.get("validity") or {}
    R.add(m, s, "validity", "valid_rate", v.get("valid_rate"))
    R.add(m, s, "validity", "records_changed", v.get("records_changed"))
    if isinstance(v.get("errors"), list):
        R.add(m, s, "validity", "errors", len(v["errors"]))
    if isinstance(sc.get("privacy"), dict):
        privacy_rows(R, m, s, sc["privacy"])
    for fam in ("keep", "concepts"):
        b = sc.get(fam) or {}
        R.add(m, s, fam, "kept", b.get("kept"), n=total(b.get("counts"), "ALL"))
        for k, val in (b.get("breakdown") or {}).items():
            if k != "ALL":
                R.add(m, s, fam, k, val, n=total(b.get("counts"), k))
    e = sc.get("edits") or {}
    for k in ("non_pii_tokens_kept", "pii_tokens_changed", "edit_precision", "tokens_changed"):
        R.add(m, s, "edits", k, e.get(k))
    R.add(m, s, "edits", "placeholders_per_1k_words", sc.get("placeholders_per_1k_words"))
    t = sc.get("terms") or {}
    R.add(m, s, "terms", "kept", t.get("kept"), n=t.get("n_terms"))
    x = sc.get("explain") or {}
    R.add(m, s, "explain", "explained", x.get("explained"), n=x.get("changed_tokens"), notes=f"decisions: {x['source']}" if x.get("source") else "")
    for k in ("changed_tokens", "covered_tokens", "decisions", "with_reason", "unaligned"):
        R.add(m, s, "explain", k, x.get(k))
    for coder, res in (sc.get("icd") or {}).items():
        icd_rows(R, m, s, coder, res)


RUN_KEYS = ("ms_per_encounter", "peak_ram_mb", "peak_vram_mb", "within_vram_budget", "wall_s", "n_records")


def attack_rows(R: Results, m: str, s: str, at: dict) -> None:
    for key, a in sorted((at.get("attacks") or {}).items()):
        if not isinstance(a, dict):
            continue
        if a.get("error"):
            R.add(m, s, "attack", f"{key}.error", None, notes=f"{a.get('name', key)} error: {short(a['error'])}")
        for arm in ("anonymized", "control"):
            d = a.get(arm) or {}
            R.add(m, s, "attack", f"{key}.{arm}.recovered", d.get("recovered"), n=d.get("total"), notes=a.get("name", ""))
            for cat, v in sorted((d.get("by_category") or {}).items()):
                R.add(m, s, "attack", f"{key}.{arm}.recovered", v, level=f"cat:{cat}")
    R.add(m, s, "attack", "max_recovered", at.get("max_recovered"), notes="largest anonymized share, A5 excluded")


NOCODES = {"privacy": ("privacy", "removed", CONSENSUS), "privacy_high": ("privacy", "removed_high_risk", CONSENSUS), "keep": ("keep", "kept", ""),
           "concepts": ("concepts", "kept", ""), "icd_concepts": ("concepts", "icd_relevant", "")}  # fmt: skip


NOCODES_LABEL = {"privacy": "privacy (consensus)", "privacy_high": "high-risk", "keep": "keep traps", "concepts": "concepts kept",
                 "icd_concepts": "ICD concepts kept"}  # fmt: skip


def nocodes_key(k: str) -> tuple[str, str, str, str]:
    """nocodes.rows() metric -> (category, metric, coder, level) of results.csv."""
    if k.startswith("icd_f1:"):
        return "icd", "anonymized.code.f1", k.partition(":")[2], ""
    cat, metric, level = NOCODES.get(k, ("validity", k, ""))
    return cat, metric, "", level


def collect(d: Data) -> tuple[Results, list[dict]]:
    R = Results()
    for m in d.models:
        for s in d.suites:
            if (sc := d.score(m, s)) is not None:
                score_rows(R, m, s, sc)
            if r := d.run(m, s):
                for k in RUN_KEYS:
                    R.add(m, s, "efficiency", k, r.get(k), n=r.get("n_records") if k == "ms_per_encounter" else None)
            if at := d.attacks.get(m, {}).get(s):
                attack_rows(R, m, s, at)
        runs = [r for s in d.suites if (r := d.run(m, s))]
        if runs:
            n = sum(num(r.get("n_records")) or 0 for r in runs)
            wall = math.fsum(num(r.get("wall_s")) or 0 for r in runs)
            vram = max(num(r.get("peak_vram_mb")) or 0 for r in runs)
            R.add(m, "all", "efficiency", "suites", len(runs))
            R.add(m, "all", "efficiency", "n_records", n)
            R.add(m, "all", "efficiency", "wall_s", round(wall, 3))
            R.add(m, "all", "efficiency", "ms_per_encounter", round(1000 * wall / n, 3) if n else None, n=n)
            R.add(m, "all", "efficiency", "peak_ram_mb", max(num(r.get("peak_ram_mb")) or 0 for r in runs))
            R.add(m, "all", "efficiency", "peak_vram_mb", vram)
            R.add(m, "all", "efficiency", "within_vram_budget", vram <= VRAM_BUDGET_MB)
        medical = d.suites_of("kaggle")
        for fam, metric, level in (("privacy", "removed", CONSENSUS), ("terms", "kept", "")):
            vals = [R.get(m, s, fam, metric, level=level) for s in medical]
            if medical and all(v is not None for v in vals):
                R.add(m, "kaggle_medical_mean", fam, metric, round(math.fsum(vals) / len(vals), 6), n=len(vals), level=level,
                      notes="mean over " + " ".join(medical))  # fmt: skip
    ranks = d.drift.get("complexity_rank") or {}
    for s in d.suites_of("robust"):
        for m in d.models:
            for metric, (fam, key, level) in (("privacy", ("privacy", "removed", CONSENSUS)), ("high_risk", ("privacy", "removed_high_risk", CONSENSUS)),
                                              ("terms", ("terms", "kept", ""))):  # fmt: skip
                R.add(m, s, "robustness", metric, R.get(m, s, fam, key, level=level), notes=f"complexity rank {ranks[s]}" if s in ranks else "")
    for s, vals in sorted((d.drift.get("suites") or {}).items()):
        if isinstance(vals, dict):
            R.add("", s, "drift", "complexity_rank", ranks.get(s))
            for k, v in vals.items():
                R.add("", s, "drift", k, v)
    ref = d.drift.get("reference") or {}
    for k in ("records", "tokens"):
        R.add("", "reference", "drift", k, ref.get(k), notes=str(ref.get("name", "")))
    for k, v in (ref.get("values") or {}).items():
        R.add("", "reference", "drift", k, v)
    pii = d.gold.get("pii") or {}
    for lv, v in (pii.get("levels") or {}).items():
        R.add("", HEAD, "privacy", "gold_items", v, level=lv, notes="merged gold")
    for lv, v in (pii.get("occurrences") or {}).items():
        R.add("", HEAD, "privacy", "gold_occurrences", v, level=lv, notes="merged gold")
    for votes, v in sorted((pii.get("votes") or {}).items()):
        R.add("", HEAD, "privacy", "gold_items", v, level=f"votes:{votes}", notes="merged gold")
    nc = nocodes.rows(d.scores, d.registry, HEAD) if d.registry else []
    for row in nc:
        for k, v in row["metrics"].items():
            cat, metric, coder, level = nocodes_key(k)
            R.add(row["nocodes"], HEAD, cat, f"delta_vs_base:{metric}", v["delta"], coder=coder, level=level, notes=f"{row['nocodes']} minus {row['base']}")
    return R, nc


# --- rendering helpers ---------------------------------------------------------------------------------------------


def fmt(x, digits: int | str | None = 3) -> str:
    """A table cell: "–" for no value, 3 decimals for shares, thousands separators from 1000 on."""
    if x is None:
        return DASH
    if digits == "bool" or isinstance(x, bool):
        return "yes" if x else "**no**"
    if not isinstance(x, (int, float)):
        return str(x)
    if digits is None:
        return f"{x:,}" if isinstance(x, int) else f"{x:,.0f}"
    return f"{x:,.{digits}f}" if abs(x) < 1000 else f"{x:,.0f}"


def signed(x) -> str:
    return DASH if x is None else f"{x:+.3f}"


def table(header: list[str], rows: list[list[str]], left: int = 1) -> list[str]:
    """A markdown table; the first `left` columns are text (left-aligned), the others numbers."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i < left else "---:" for i in range(len(header))) + "|"]
    return out + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def name(m: str) -> str:
    return f"`{m}` (ref)" if m in REFERENCE else f"`{m}`"


def section(md: list[str], title: str, *lines: str) -> None:
    md += [f"## {title}", ""] + [ln for ln in lines] + ([""] if lines else [])


def headline(d: Data, R: Results, m: str) -> dict:
    out = {}
    for c in HEADLINE:
        v = R.get(m, HEAD, c.category, c.metric, coder=d.primary if c.primary else "", level=c.level)
        out[c.key] = (bool(v) if v is not None else None) if c.digits == "bool" else v
    return out


def headline_cell(d: Data, m: str, c: Col, v) -> str:
    if v is not None:
        return fmt(v, c.digits)
    if d.failed(m, HEAD):
        return "failed"
    if c.primary and g(d.score(m, HEAD), "icd", d.primary or "", "error"):
        return "error"
    return DASH


def pareto(points: dict[str, tuple]) -> dict[str, bool | None]:
    """True when no other model is >= on both axes and > on one; None without both values."""
    out: dict[str, bool | None] = {}
    for m, (x, y) in points.items():
        if x is None or y is None:
            out[m] = None
            continue
        out[m] = not any(o != m and ox is not None and oy is not None and ox >= x and oy >= y and (ox > x or oy > y) for o, (ox, oy) in points.items())
    return out


def tradeoff(d: Data, R: Results) -> dict:
    """x = consensus privacy, y = primary-coder ICD F1 retained, or ICD concepts kept when no real model has the coder."""
    real = [m for m in d.models if m not in REFERENCE]
    has = bool(d.primary) and any(R.get(m, HEAD, "icd", "f1_retention", coder=d.primary) is not None for m in real)
    if has:
        ykey, ylabel = ("icd", "f1_retention", d.primary), f"ICD F1 retained ({d.primary}: anonymized F1 / original F1)"
    else:
        ykey = ("concepts", "icd_relevant", "")
        ylabel = f"ICD concepts kept (fallback: no {d.primary or 'primary'} coder result yet)"
    pts = {m: (R.get(m, HEAD, "privacy", "removed", level=CONSENSUS), R.get(m, HEAD, ykey[0], ykey[1], coder=ykey[2])) for m in d.models}
    flags = pareto({m: p for m, p in pts.items() if m not in REFERENCE})
    return {"x": "privacy (consensus)", "y": ylabel, "y_metric": {"category": ykey[0], "metric": ykey[1], "coder": ykey[2]}, "fallback": not has,
            "points": {m: {"x": x, "y": y, "reference": m in REFERENCE, "pareto": flags.get(m)} for m, (x, y) in pts.items()}}  # fmt: skip


# --- SVG -----------------------------------------------------------------------------------------------------------

LIGHT = {"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "lead": "#a3a299", "series": "#2a78d6", "ref": "#6f6e69"}
DARK = {"bg": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#3a3a37", "lead": "#6f6e69", "series": "#3987e5", "ref": "#a3a299"}


def nice_step(raw: float) -> float:
    e = 10 ** math.floor(math.log10(raw))
    f = raw / e
    return (1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10) * e


def axis(vals: list[float], lo: float | None = None, hi: float | None = None) -> tuple[float, float, float]:
    a, b = min(vals + ([lo] if lo is not None else [])), max(vals + ([hi] if hi is not None else []))
    if b - a < 1e-9:
        a, b = a - 0.05, b + 0.05
    pad = 0 if lo is not None and hi is not None and a >= lo and b <= hi else (b - a) * 0.08
    step = nice_step((b - a + 2 * pad) / 5)
    return math.floor((a - pad) / step + 1e-9) * step, math.ceil((b + pad) / step - 1e-9) * step, step


def tick_label(v: float, step: float) -> str:
    digits = max(0, -math.floor(math.log10(step) + 1e-9))
    if abs(step * 10**digits - round(step * 10**digits)) > 1e-6:
        digits += 1
    return f"{v:.{digits}f}"


def svg(t: dict, generated: str) -> str:
    W, H, L, Rm, T, B = 760, 540, 72, 32, 92, 74
    pw, ph = W - L - Rm, H - T - B
    plotted = {m: p for m, p in t["points"].items() if p["x"] is not None and p["y"] is not None}
    missing = [m for m in t["points"] if m not in plotted]
    x0, x1, xs = axis([p["x"] for p in plotted.values()] or [0.0, 1.0], 0.0, 1.0)
    y0, y1, ys = axis([p["y"] for p in plotted.values()] or [0.0, 1.0])

    def px(x: float) -> float:
        return L + (x - x0) / (x1 - x0) * pw

    def py(y: float) -> float:
        return T + (1 - (y - y0) / (y1 - y0)) * ph

    css = "".join(f".{k}{{{'stroke' if k in ('grid', 'lead') else 'fill'}:{v}}}" for k, v in DARK.items())
    css += f".ring{{stroke:{DARK['bg']}}}.hollow{{stroke:{DARK['series']};fill:{DARK['bg']}}}.refm{{fill:{DARK['ref']}}}"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif" role="img" aria-labelledby="title">',
         "<title id=\"title\">Privacy vs utility on aci_eval</title>",
         f"<style>@media (prefers-color-scheme: dark){{{css}}}</style>",
         f'<rect class="bg" width="{W}" height="{H}" fill="{LIGHT["bg"]}"/>',
         f'<text class="ink" x="{L}" y="28" font-size="15" font-weight="600" fill="{LIGHT["ink"]}">Privacy vs utility on aci_eval</text>',
         f'<text class="ink2" x="{L}" y="47" font-size="11.5" fill="{LIGHT["ink2"]}">Filled: Pareto-optimal among the real models. identity and oracle are bounds, not competitors. No combined score.</text>']  # fmt: skip
    lx = L  # legend
    for kind, label in (("pareto", "Pareto-optimal"), ("dominated", "dominated"), ("reference", "reference (identity, oracle)")):
        cy = 70
        if kind == "pareto":
            o.append(f'<circle class="series" cx="{lx + 6}" cy="{cy}" r="6" fill="{LIGHT["series"]}"/>')
        elif kind == "dominated":
            o.append(f'<circle class="hollow" cx="{lx + 6}" cy="{cy}" r="5" fill="{LIGHT["bg"]}" stroke="{LIGHT["series"]}" stroke-width="2"/>')
        else:
            o.append(f'<rect class="refm" x="{lx + 1}" y="{cy - 5}" width="10" height="10" fill="{LIGHT["ref"]}"/>')
        o.append(f'<text class="ink2" x="{lx + 17}" y="{cy + 4}" font-size="11.5" fill="{LIGHT["ink2"]}">{label}</text>')
        lx += 30 + 6.6 * len(label)
    k = 0
    while (v := x0 + k * xs) <= x1 + 1e-9:
        o.append(f'<line class="grid" x1="{px(v):.1f}" y1="{T}" x2="{px(v):.1f}" y2="{T + ph}" stroke="{LIGHT["grid"]}" stroke-width="1"/>')
        o.append(f'<text class="ink2" x="{px(v):.1f}" y="{T + ph + 17}" font-size="11" text-anchor="middle" fill="{LIGHT["ink2"]}">{tick_label(v, xs)}</text>')
        k += 1
    k = 0
    while (v := y0 + k * ys) <= y1 + 1e-9:
        o.append(f'<line class="grid" x1="{L}" y1="{py(v):.1f}" x2="{L + pw}" y2="{py(v):.1f}" stroke="{LIGHT["grid"]}" stroke-width="1"/>')
        o.append(f'<text class="ink2" x="{L - 8}" y="{py(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{LIGHT["ink2"]}">{tick_label(v, ys)}</text>')
        k += 1
    o.append(f'<text class="ink" x="{L + pw / 2:.1f}" y="{T + ph + 40}" font-size="12" text-anchor="middle" fill="{LIGHT["ink"]}">'
             "privacy, consensus gold level (share of PII occurrences removed)</text>")  # fmt: skip
    o.append(f'<text class="ink" transform="translate(18 {T + ph / 2:.1f}) rotate(-90)" font-size="12" text-anchor="middle" fill="{LIGHT["ink"]}">'
             f"{escape(t['y'])}</text>")  # fmt: skip
    groups: dict[tuple, dict] = {}  # identical points share a marker and a label
    for m, p in plotted.items():
        grp = groups.setdefault((round(p["x"], 4), round(p["y"], 4)), {"x": p["x"], "y": p["y"], "names": [], "kind": "dominated"})
        grp["names"].append(m)
        grp["kind"] = "reference" if p["reference"] else ("pareto" if p["pareto"] or grp["kind"] == "pareto" else grp["kind"])
    marks = []  # (x0, x1, y0, y1) of every marker, which labels must not cover
    for grp in groups.values():
        cx, cy = grp["cx"], grp["cy"] = px(grp["x"]), py(grp["y"])
        marks.append((cx - 7, cx + 7, cy - 7, cy + 7))
        tip = escape(f"{', '.join(grp['names'])}: privacy {grp['x']:.3f}, utility {grp['y']:.3f}")
        if grp["kind"] == "reference":
            o.append(f'<rect class="refm" x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" fill="{LIGHT["ref"]}"><title>{tip}</title></rect>')
        elif grp["kind"] == "pareto":
            o.append(f'<circle class="series ring" cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{LIGHT["series"]}" stroke="{LIGHT["bg"]}" stroke-width="2">'
                     f"<title>{tip}</title></circle>")  # fmt: skip
        else:
            o.append(f'<circle class="hollow" cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{LIGHT["bg"]}" stroke="{LIGHT["series"]}" stroke-width="2">'
                     f"<title>{tip}</title></circle>")  # fmt: skip
    placed: list[tuple[float, float, float, float]] = []

    def free(b: tuple[float, float, float, float]) -> bool:
        inside = b[0] >= L + 2 and b[1] <= W - 4 and b[2] >= T + 2 and b[3] <= T + ph - 2
        return inside and not any(b[0] < o_[1] + 2 and b[1] > o_[0] - 2 and b[2] < o_[3] + 1 and b[3] > o_[2] - 1 for o_ in placed + marks)

    def crowd(gr: dict) -> int:
        return sum(abs(o_["cx"] - gr["cx"]) < 60 and abs(o_["cy"] - gr["cy"]) < 30 for o_ in groups.values())

    for grp in sorted(groups.values(), key=lambda gr: (-crowd(gr), -gr["cx"], gr["cy"])):  # crowded points first
        cx, cy, label = grp["cx"], grp["cy"], ", ".join(grp["names"])
        w = 6.3 * len(label)
        best = None
        for r in (10, 22, 36, 52, 70, 92, 116, 144):
            d = r * 0.7
            for dx, dy, anchor in ((r, 4, "start"), (-r, 4, "end"), (0, -r + 2, "middle"), (0, r + 8, "middle"),
                                   (d, -d + 2, "start"), (d, d + 8, "start"), (-d, -d + 2, "end"), (-d, d + 8, "end")):  # fmt: skip
                tx, ty = cx + dx, cy + dy
                x_lo = tx if anchor == "start" else tx - w if anchor == "end" else tx - w / 2
                box = (x_lo, x_lo + w, ty - 10, ty + 3)
                if free(box):
                    best = (tx, ty, anchor, box, r)
                    break
            if best:
                break
        if best is None:  # nowhere free: right of the point
            best = (cx + 10, cy + 4, "start", (cx + 10, cx + 10 + w, cy - 6, cy + 7), 10)
        tx, ty, anchor, box, r = best
        placed.append(box)
        if r > 10:  # leader line from the marker to the nearest point of the label
            lx_, ly_ = min(max(cx, box[0]), box[1]), min(max(cy, box[2]), box[3])
            dist = math.hypot(lx_ - cx, ly_ - cy) or 1
            sx, sy = cx + (lx_ - cx) * 7 / dist, cy + (ly_ - cy) * 7 / dist
            o.append(f'<line class="lead" x1="{sx:.1f}" y1="{sy:.1f}" x2="{lx_:.1f}" y2="{ly_:.1f}" stroke="{LIGHT["lead"]}" stroke-width="1"/>')
        cls, fill = ("ink2", LIGHT["ink2"]) if grp["kind"] == "reference" else ("ink", LIGHT["ink"])
        o.append(f'<text class="{cls}" x="{tx:.1f}" y="{ty:.1f}" font-size="11" text-anchor="{anchor}" fill="{fill}">{escape(label)}</text>')
    o.append(f'<text class="ink2" x="{W - 8}" y="{H - 10}" font-size="10" text-anchor="end" fill="{LIGHT["ink2"]}">{escape(generated)}</text>')
    if missing:
        o.append(f'<text class="ink2" x="{L}" y="{H - 10}" font-size="11" fill="{LIGHT["ink2"]}">not plotted (no value): {escape(", ".join(missing))}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


# --- the report ----------------------------------------------------------------------------------------------------


def render(d: Data, R: Results, nc: list[dict], generated: str) -> tuple[list[str], dict, dict, list[str]]:
    models, prim = d.models, d.primary
    md = [f"# Anonymization evaluation ({generated})", ""]
    md += [f"- {name(m)}: {d.registry.get(m, {}).get('description', '')}" for m in models] + [""]
    if d.pending:
        md += ["Pending (disabled in `models.toml`, no data): " + ", ".join(f"`{m}`" for m in d.pending), ""]
    status = []
    if prim:
        without = [m for m in models if R.get(m, HEAD, "icd", "f1_retention", coder=prim) is None]
        if without:
            status.append(f"- ICD primary coder `{prim}`: no `{HEAD}` result yet for " + ", ".join(f"`{m}`" for m in without) + " (cells show –).")
    else:
        status.append("- No primary ICD coder in `coders.toml`: the ICD headline columns are empty.")
    no_score = [m for m in models if d.score(m, HEAD) is None]
    if no_score:
        status.append(f"- No `{HEAD}` score yet: " + ", ".join(f"`{m}`" for m in no_score) + ".")
    failed = sorted({f"`{m}` on `{s}`" for m in models for s in d.suites if d.failed(m, s)})
    if failed:
        status.append("- Failed runs (scored as failed): " + ", ".join(failed) + ".")
    if not d.drift:
        status.append("- No `drift.json` yet (`python3 -I eval drift`): complexity ranks are empty.")
    if status:
        md += ["Data status:", ""] + status + [""]

    heads = {m: headline(d, R, m) for m in models}
    hdr = ["model"] + [f"{c.header} ({prim})" if c.primary and prim else c.header for c in HEADLINE]
    section(md, f"Headline (`{HEAD}`)")
    md += table(hdr, [[name(m)] + [headline_cell(d, m, c, heads[m][c.key]) for c in HEADLINE] for m in models])
    md += ["Privacy, concepts and ICD columns are shares (1 = everything removed, or everything kept). ms/enc, RAM and VRAM are from the "
           f"`{HEAD}` run; explained = share of changed tokens with a logged reason; attack = the largest share an attack recovers "
           "from the output (A5 excluded). `identity` and `oracle` are bounds, not competitors.", ""]  # fmt: skip

    t = tradeoff(d, R)
    pts = t["points"]
    order_ = sorted(models, key=lambda m: (m in REFERENCE, -(pts[m]["x"] if pts[m]["x"] is not None else -1), m))
    rows = []
    for m in order_:
        p = pts[m]
        flag = "ref" if p["reference"] else ("**yes**" if p["pareto"] else (DASH if p["pareto"] is None else "no"))
        rows.append([name(m), fmt(p["x"]), fmt(R.get(m, HEAD, "privacy", "removed", level="direct")), fmt(p["y"])]
                    + ([] if t["fallback"] else [fmt(R.get(m, HEAD, "concepts", "icd_relevant"))]) + [fmt(R.get(m, HEAD, "concepts", "kept")), flag])  # fmt: skip
    ycol = f"ICD F1 retained ({prim})" if not t["fallback"] else "ICD concepts kept"
    section(md, "Privacy vs utility (tradeoff)",
            f"Sorted by consensus privacy. Pareto-optimal: no other real model is at least as good on both consensus privacy and {ycol}, and "
            "better on one. No combined score: the organizers weigh the criteria separately."
            + (f" No model has a `{prim}` result yet, so the utility axis is ICD concepts kept." if t["fallback"] else ""),
            "![privacy vs utility](tradeoff.svg)")  # fmt: skip
    md += table(["model", "privacy (consensus)", "direct", ycol] + ([] if t["fallback"] else ["ICD concepts kept"]) + ["concepts kept", "Pareto"], rows)

    # privacy levels on the headline suite
    lv_rows, src_rows = [], []
    for m in models:
        if d.score(m, HEAD) is None:
            continue
        lv_rows.append([name(m)] + [fmt(R.get(m, HEAD, "privacy", "removed", level=lv)) for lv in ("union", CONSENSUS, "direct")]
                       + [fmt(R.get(m, HEAD, "privacy", "removed_high_risk", level=CONSENSUS))]
                       + [fmt(R.get(m, HEAD, "privacy", "entity_removed", level=lv)) for lv in ("union", CONSENSUS, "direct")])  # fmt: skip
        src_rows.append([name(m)] + [fmt(R.get(m, HEAD, "privacy", f"votes:{v}", level="union")) for v in (1, 2, 3)]
                        + [fmt(R.get(m, HEAD, "privacy", f"src:{s}", level="union")) for s in ("harness", "vm", "ketai")]
                        + [fmt(R.get(m, HEAD, "privacy", "removed", level=f"gold:{s}")) for s in ("harness", "vm", "ketai")])  # fmt: skip
    gl = []
    if d.gold:
        items = [f"{lv} {fmt(R.get('', HEAD, 'privacy', 'gold_items', level=lv), None)}" for lv in ("union", CONSENSUS, "direct")]
        occ = [f"{lv} {fmt(R.get('', HEAD, 'privacy', 'gold_occurrences', level=lv), None)}" for lv in ("union", CONSENSUS, "direct")]
        votes = "/".join(fmt(R.get("", HEAD, "privacy", "gold_items", level=f"votes:{v}"), None) for v in (1, 2, 3))
        gl = [f"Merged gold: items {', '.join(items)}; occurrences {', '.join(occ)}; items with 1/2/3 votes {votes}."]
    section(md, f"Privacy levels (`{HEAD}`, merged gold)",
            "Occurrence-level removal per gold level (union: any gold; consensus: direct identifiers or 2+ votes, the headline; direct: "
            "names, dates, DOB, IDs, contacts), the high-risk share, and entity-level removal (an item counts only when every "
            "occurrence is gone).", *gl)  # fmt: skip
    md += table(["model", "union", "consensus", "direct", "high-risk", "entity union", "entity consensus", "entity direct"], lv_rows)
    md += ["Votes and sources: `votes:N` = union items listed by N golds; `src:X` = union items gold X supports; `gold X` = gold X scored "
           "alone on its own list (`gold harness` is the A+D team's cross-check value).", ""]  # fmt: skip
    md += table(["model", "votes:1", "votes:2", "votes:3", "src:harness", "src:vm", "src:ketai", "gold harness", "gold vm", "gold ketai"], src_rows)

    # ICD coders
    def icd_table(suites: list[str], with_suite: bool) -> list[str]:
        rows, errs = [], {}
        for s in suites:
            for m in models:
                block = g(d.score(m, s), "icd") or {}
                for c in d.coder_names + [c for c in block if c not in d.coder_names]:
                    res = block.get(c)
                    if not isinstance(res, dict):
                        continue
                    lead = ([f"`{s}`"] if with_suite else []) + [name(m), f"`{c}`" + (" !" if res.get("valid_for_comparison") is False else "")]
                    if res.get("error"):
                        errs.setdefault(short(res["error"]), set()).add(c)
                        rows.append(lead + ["error"] + [""] * 7)
                        continue

                    def v(metric: str, s=s, m=m, c=c):
                        return R.get(m, s, "icd", metric, coder=c)

                    dr = R.row(m, s, "icd", "f1_delta", coder=c)
                    delta = fmt(dr["value"]) + (f" [{fmt(dr['ci_low'])}, {fmt(dr['ci_high'])}]" if dr and dr["ci_low"] is not None else "") if dr else DASH
                    rows.append(lead + [fmt(v("original.code.f1")), fmt(v("anonymized.code.f1")), delta, fmt(v("f1_retention")), fmt(v("agreement.code.f1")),
                                        fmt(v("anonymized.category.f1")), fmt(v("errors"), None)])  # fmt: skip
        out = table(
            (["suite"] if with_suite else []) + ["model", "coder", "F1 orig", "F1 anon", "Δ F1 [CI95]", "retained", "agree", "cat-F1 anon", "errors"],
            rows,
            left=3 if with_suite else 2,
        )
        return out + [f"- error ({', '.join(f'`{c}`' for c in sorted(cs))}): {msg}" for msg, cs in errs.items()] + ([""] if errs else [])

    icd_suites = [s for s in d.suites if any(g(d.score(m, s), "icd") for m in models)]
    section(md, f"ICD coders (`{HEAD}`)",
            "Micro F1 of each coder's codes vs the gold codes on the original and the anonymized text (code level; cat-F1 = 3-character "
            "category), the paired-bootstrap change with its CI95, retained = anonymized F1 / original F1, agree = anonymized vs original "
            "predictions (the damage done by anonymization alone). `!` = not valid for comparison (a prediction failed: counted as "
            f"empty, see errors). The primary coder (`coders.toml`) is `{prim or DASH}`.")  # fmt: skip
    md += icd_table([HEAD], False) if HEAD in icd_suites else ["No ICD results yet.", ""]
    others = [s for s in icd_suites if s != HEAD]
    if others:
        section(md, "ICD coders on the other suites")
        md += icd_table(others, True)

    # main suites, utility and edits
    cols = [("valid", "validity", "valid_rate", ""), ("privacy", "privacy", "removed", CONSENSUS), ("high-risk", "privacy", "removed_high_risk", CONSENSUS),
            ("keep traps", "keep", "kept", ""), ("non-PII kept", "edits", "non_pii_tokens_kept", ""), ("edit prec.", "edits", "edit_precision", ""),
            ("concepts", "concepts", "kept", ""), ("ICD concepts", "concepts", "icd_relevant", ""), ("terms", "terms", "kept", ""),
            ("placeholders /1k words", "edits", "placeholders_per_1k_words", ""), ("explained", "explain", "explained", "")]  # fmt: skip
    for s in [s for s in MAIN if s in d.suites]:
        rows = []
        for m in models:
            if d.score(m, s) is None:
                continue
            if d.failed(m, s):
                rows.append([name(m), "failed"] + [""] * (len(cols) - 1))
                continue
            rows.append([name(m)] + [fmt(R.get(m, s, fam, key, level=lv), 2 if key == "placeholders_per_1k_words" else 3) for _, fam, key, lv in cols])
        if rows:
            info = d.info.get(s, {})
            section(md, f"`{s}`: {info.get('about', '')} ({fmt(info.get('records'), None)} records)")
            md += table(["model"] + [h for h, *_ in cols], rows)

    if nc:
        section(
            md,
            "Without ICD-10 codes at inference (bonus)",
            f"Each model run with `blind_codes` beside its base model on `{HEAD}` (Δ = without codes minus with codes).",
        )
        for row in nc:
            rows = []
            for k, v in row["metrics"].items():
                label = f"ICD F1 anon (`{k.partition(':')[2]}`)" if k.startswith("icd_f1:") else NOCODES_LABEL.get(k, k)
                rows.append([label, fmt(v["base"]), fmt(v["nocodes"]), signed(v["delta"])])
            md += table(["metric", f"`{row['base']}`", f"`{row['nocodes']}`", "Δ"], rows)

    # attacks
    at_suites = [s for s in d.suites if any(s in d.attacks.get(m, {}) for m in models)]
    if HEAD in at_suites:
        keys = sorted({k for m in models for k in (g(d.attacks.get(m, {}).get(HEAD), "attacks") or {})})
        names_ = {}
        for m in models:
            for k, a in (g(d.attacks.get(m, {}).get(HEAD), "attacks") or {}).items():
                if isinstance(a, dict) and a.get("name"):
                    names_.setdefault(k, a["name"])
        rows = []
        for m in models:
            at = d.attacks.get(m, {}).get(HEAD)
            if not at:
                continue
            cells = []
            for k in keys:
                a = g(at, "attacks", k)
                if not isinstance(a, dict):
                    cells.append(DASH)
                elif a.get("error") and not a.get("anonymized"):
                    cells.append("error")
                else:
                    an, co = R.get(m, HEAD, "attack", f"{k}.anonymized.recovered"), R.get(m, HEAD, "attack", f"{k}.control.recovered")
                    cells.append(f"{fmt(an)} / {fmt(co)}")
            rows.append([name(m)] + cells + [fmt(R.get(m, HEAD, "attack", "max_recovered"))])
        section(md, f"Adversarial attacks (`{HEAD}`, bonus)",
                "Share recovered from the output / from the original text (the positive control). Lower is better on the left; an "
                "attack that recovers little from the original says little. A5 links records with the originals in hand, an upper "
                "bound by design, so the max leaves it out. – = not run.")  # fmt: skip
        md += table(["model"] + [f"{k} {names_.get(k, '')}".strip() for k in keys] + ["max (A5 excluded)"], rows)
    at_more = [s for s in at_suites if any(s in d.attacks.get(m, {}) for m in models if m not in REFERENCE)]  # suites a real model was attacked on
    if [s for s in at_more if s != HEAD]:
        rows = [[name(m)] + [fmt(R.get(m, s, "attack", "max_recovered")) for s in at_more] for m in models if any(s in d.attacks.get(m, {}) for s in at_more)]
        section(md, "Adversarial attacks, max recovered per suite")
        md += table(["model"] + [f"`{s}`" for s in at_more], rows)

    ranks = d.drift.get("complexity_rank") or {}
    robust = d.suites_of("robust")
    if robust:
        rows = []
        for m in models:
            cells = [fmt(R.get(m, s, "robustness", k)) for s in robust for k in ("privacy", "high_risk", "terms")]
            if any(c != DASH for c in cells):
                rows.append([name(m)] + cells)
        hdr = ["model"] + [
            f"`{s}` {k}" + (f" (rank {ranks[s]})" if k == "privacy" and s in ranks else "") for s in robust for k in ("privacy", "high-risk", "terms")
        ]
        section(md, "Robustness to unseen formats (bonus)",
                "Privacy (consensus), high-risk privacy and ICD-10-CM terms kept on the robust suites. Rank = complexity rank from "
                f"`drift.json` (1 = closest to ACI train by unigram JSD, of {len(ranks) or DASH} suites).")  # fmt: skip
        md += table(hdr, rows)

    dsu = d.drift.get("suites") or {}
    if dsu:
        dcols = [("JSD unigram", "jsd_unigram"), ("JSD bigram", "jsd_bigram"), ("vocab Jaccard", "vocab_jaccard"), ("entropy Δ", "entropy_diff"),
                 ("MTLD Δ", "mtld_diff"), ("sentence length Δ", "sentence_length_mean_diff"), ("FK grade Δ", "flesch_kincaid_diff"),
                 ("line breaks/100 Δ", "line_breaks_per_100_diff"), ("key-value Δ", "key_value_ratio_diff")]  # fmt: skip
        srt = sorted(dsu, key=lambda s: (num(g(dsu, s, "jsd_unigram")) is None, num(g(dsu, s, "jsd_unigram")) or 0, s))
        rows = [[f"`{s}`", fmt(R.get("", s, "drift", "complexity_rank"), None), d.role(s) or DASH, fmt(R.get("", s, "drift", "records"), None),
                 fmt(R.get("", s, "drift", "tokens"), None)] + [fmt(R.get("", s, "drift", k)) for _, k in dcols] for s in srt]  # fmt: skip
        ref = d.drift.get("reference") or {}
        section(md, "Suite complexity (drift from ACI train)",
                f"Each suite vs the reference ({ref.get('name', 'ACI train')}: {fmt(R.get('', 'reference', 'drift', 'records'), None)} records, "
                f"{fmt(R.get('', 'reference', 'drift', 'tokens'), None)} tokens), ordered by JSD unigram. Δ = |suite − reference|. "
                "Definitions: `eval/docs/CORPUS-DRIFT.md`.")  # fmt: skip
        md += table(["suite", "rank", "role", "records", "tokens"] + [h for h, _ in dcols], rows, left=1)

    # Kaggle complement
    ks = d.suites_of("kaggle", "generic")
    for title, fam, key, lv in (("privacy (share of gold PII occurrences removed)", "privacy", "removed", CONSENSUS),
                                ("clinical terms kept (ICD-10-CM vocabulary)", "terms", "kept", "")):  # fmt: skip
        rows = []
        for m in models:
            cells = [("failed" if d.failed(m, s) else fmt(R.get(m, s, fam, key, level=lv))) for s in ks]
            if any(c != DASH for c in cells):
                rows.append([name(m)] + cells + [fmt(R.get(m, "kaggle_medical_mean", fam, key, level=lv))])
        if ks:
            section(md, f"Kaggle complement, {title}")
            md += table(["model"] + [f"`{s}`" for s in ks] + ["mean (medical)"], rows)
    ucols = [("NBME graded findings kept", "nbme_notes", "keep", "kept", ""), ("devp code evidence kept", "devp_discharge", "keep", "kept", ""),
             ("devp high-risk removed", "devp_discharge", "privacy", "removed_high_risk", CONSENSUS)]  # fmt: skip
    rows = [[name(m)] + [fmt(R.get(m, s, fam, key, level=lv)) for _, s, fam, key, lv in ucols] for m in models]
    rows = [r for r in rows if any(c != DASH for c in r[1:])]
    if rows:
        section(md, "Kaggle complement, utility with gold")
        md += table(["model"] + [h for h, *_ in ucols], rows)
    langs = sorted({r["metric"] for r in R.rows if r["suite"] == "imerit_multi" and r["category"] == "privacy_by_language"})
    rows = []
    for m in models:
        cells = [fmt(R.get(m, "imerit_en", "privacy", "removed", level=CONSENSUS))] + [
            fmt(R.get(m, "imerit_multi", "privacy_by_language", x, level=CONSENSUS)) for x in langs
        ]
        if any(c != DASH for c in cells):
            rows.append([name(m)] + cells)
    if rows:
        section(md, "Kaggle complement, privacy by language (iMerit scribe: patient, physician, DOB)")
        md += table(["model", "en"] + langs, rows)

    cats_found = {r["metric"] for r in R.rows if r["suite"] == HEAD and r["category"] == "privacy_by_category"}
    cats = [c for c in CATEGORY_RISK if c in cats_found] + sorted(cats_found - set(CATEGORY_RISK))
    rows = []
    for m in models:
        cells = [fmt(R.get(m, HEAD, "privacy_by_category", c, level=CONSENSUS)) for c in cats]
        cells += [fmt(R.get(m, HEAD, "privacy_by_field", f, level=CONSENSUS)) for f in ("dialogue", "note")]
        if any(c != DASH for c in cells):
            rows.append([name(m)] + cells)
    if rows:
        section(md, f"`{HEAD}` privacy by category and field (consensus level)")
        md += table(["model"] + cats + ["dialogue", "note"], rows)

    rows = []
    for m in models:
        if R.get(m, "all", "efficiency", "suites") is None:
            continue
        rows.append([name(m)] + [fmt(R.get(m, "all", "efficiency", k), dg) for k, dg in
                                 (("suites", None), ("n_records", None), ("wall_s", 1), ("ms_per_encounter", 1), ("peak_ram_mb", 1), ("peak_vram_mb", 1),
                                  ("within_vram_budget", "bool"))])  # fmt: skip
    if rows:
        section(md, "Efficiency over all suites run")
        md += table(["model", "suites", "records", "wall s", "ms/enc", "peak RAM MB", "peak VRAM MB", f"≤ {VRAM_BUDGET_MB} MB VRAM"], rows)

    md += [
        "## How to read this",
        "",
        "- **privacy**: share of gold PII occurrences no longer in the output (case-insensitive whole words; a multi-word item also leaks",
        "  when one of its distinctive tokens survives). On `aci_eval` the gold merges three golds (harness, VM, ketai) with votes:",
        "  **consensus** (direct identifiers or 2+ votes) is the headline, **direct** (names, dates, DOB, IDs, contacts) the gate,",
        "  **union** the pessimistic bound. **entity-level** counts an item only when every occurrence is gone (the paper's strict",
        "  matching). **privacy (harness gold)** scores the A+D team's gold alone, so their own number stays comparable. All three golds",
        "  are LLM-assisted, so votes are less independent than they look. **high-risk**: names, dates, ages, DOB, IDs, contacts.",
        "- **concepts / ICD concepts**: share of gold medical-concept occurrences kept (all, and those relevant to the codes).",
        "  **keep traps**: must-survive strings (eponyms, clinical numbers, code evidence). **terms**: ICD-10-CM description words kept.",
        "- **ICD F1**: the coders of `coders.toml` stand in for the organizers' hidden coder. F1 anon is micro F1 vs the gold codes on",
        "  the anonymized text; **retained** = anonymized F1 / original F1; **agree** = anonymized vs original predictions. The headline",
        f"  uses the primary coder (`{prim or DASH}`); the weak local coders (kNN, lexical) are best read through agreement.",
        "- **explained**: share of changed tokens covered by a logged decision with a reason (the explainability bonus).",
        "- **attacks**: re-identification attempts on the output, each beside the same attack on the original text (positive control).",
        "- **Pareto**: no other real model is at least as good on both consensus privacy and ICD F1 retained, and better on one. No",
        "  combined score: the organizers weigh the criteria separately.",
        "- **RAM/VRAM**: peaks of the whole process tree (VRAM from nvidia-smi, includes the CUDA context). Budget: 4 GB VRAM.",
        "- `identity` (input unchanged) and `oracle` (gold PII replaced, reads the gold) are references, not models.",
        "- Every number here is also a row of `results.csv` (with the per-category breakdowns); `results_headline.csv` is the headline.",
        "",
    ]

    report = {
        "generated": generated, "headline_suite": HEAD, "primary_coder": prim,
        "models": {m: d.registry.get(m, {}).get("description", "") for m in models}, "pending": d.pending, "suites": d.info,
        "headline": heads, "tradeoff": t,
        "icd": {s: {m: {c: {k: R.get(m, s, "icd", k, coder=c) for k in ("original.code.f1", "anonymized.code.f1", "f1_delta", "f1_retention",
                                                                          "agreement.code.f1", "anonymized.category.f1", "errors", "valid_for_comparison")}
                        for c in (g(d.score(m, s), "icd") or {}) if not g(d.score(m, s), "icd", c, "error")} for m in models if g(d.score(m, s), "icd")}
                for s in icd_suites},
        "nocodes": nc,
        "attacks": {s: {m: {"max_recovered": R.get(m, s, "attack", "max_recovered"),
                            **{k: {"anonymized": R.get(m, s, "attack", f"{k}.anonymized.recovered"), "control": R.get(m, s, "attack", f"{k}.control.recovered")}
                               for k in (g(d.attacks.get(m, {}).get(s), "attacks") or {})}} for m in models if s in d.attacks.get(m, {})} for s in at_suites},
        "robustness": {s: {"complexity_rank": ranks.get(s), "models": {m: {k: R.get(m, s, "robustness", k) for k in ("privacy", "high_risk", "terms")} for m in models}}
                       for s in robust},
        "complexity_rank": ranks,
        "efficiency": {m: {k: R.get(m, "all", "efficiency", k) for k in ("suites", "n_records", "wall_s", "ms_per_encounter", "peak_ram_mb", "peak_vram_mb",
                                                                         "within_vram_budget")} for m in models},
    }  # fmt: skip

    disc = discord(d, heads, t, generated)
    return md, report, t, disc


def discord(d: Data, heads: dict, t: dict, generated: str) -> list[str]:
    prim, models = d.primary, d.models
    cols = [("privacy", "privacy"), ("direct", "direct"), ("privacy_harness", "harness gold"), ("icd_concepts", "ICD concepts"),
            ("icd_f1_retained", f"ICD F1 ret. ({prim})" if prim else "ICD F1 ret."), ("icd_agree", "ICD agree"), ("ms_per_encounter", "ms/enc"),
            ("vram_ok", "VRAM ≤ 4 GB"), ("attack_max", "attack max")]  # fmt: skip
    digits = {c.key: c.digits for c in HEADLINE}
    rows = [[name(m)] + [fmt(heads[m][k], digits[k]) for k, _ in cols] for m in models]
    md = [f"**Anonymization eval, `{HEAD}`, same pipeline for every model** ({generated})", ""]
    md += table(["model"] + [h for _, h in cols], rows)
    real = [m for m in models if m not in REFERENCE]
    best = max((m for m in real if heads[m]["privacy"] is not None), key=lambda m: heads[m]["privacy"], default=None)
    opt = [m for m in real if t["points"][m]["pareto"]]
    ylab = "ICD concepts kept" if t["fallback"] else f"ICD F1 retained ({prim})"
    md += [f"- Pareto-optimal on consensus privacy vs {ylab}: " + (", ".join(f"`{m}`" for m in opt) or DASH) + "."]
    if best:
        md += [f"- Highest consensus privacy: `{best}` {fmt(heads[best]['privacy'])}; `oracle` and `identity` bound the scale (1 and 0)."]
    md += ["- privacy = share of gold PII occurrences removed at the consensus level of the merged ACI gold (direct = names, dates, DOB, "
           "IDs, contacts; harness gold = the A+D team's gold alone, their own cross-check)."]  # fmt: skip
    md += [f"- ICD: no `{prim}` results yet, utility shown as ICD concepts kept." if t["fallback"] else
           f"- ICD F1 ret. = anonymized / original F1 of the `{prim}` coder; agree = anonymized vs original predictions. Full tables: `report.md`."]  # fmt: skip
    return md + [""]


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel opens UTF-8 directly
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -I eval report", description=__doc__.splitlines()[0], epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.parse_args(argv)
    d = Data()
    R, nc = collect(d)
    generated = time.strftime("%Y-%m-%d %H:%M")
    md, report, t, disc = render(d, R, nc, generated)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"results_csv": OUT / "results.csv", "results_headline_csv": OUT / "results_headline.csv", "tradeoff_svg": OUT / "tradeoff.svg",
             "report_discord_md": OUT / "report_discord.md"}  # fmt: skip
    report["exports"] = {k: rel(v) for k, v in files.items()}
    write_csv(files["results_csv"], list(CSV_FIELDS), R.rows)
    wide = [{"model": m, "reference": int(m in REFERENCE), **report["headline"][m],
             "pareto": None if t["points"][m]["pareto"] is None else int(t["points"][m]["pareto"])} for m in d.models]  # fmt: skip
    for row in wide:
        row["vram_ok"] = None if row["vram_ok"] is None else int(row["vram_ok"])
    write_csv(files["results_headline_csv"], ["model", "reference"] + [c.key for c in HEADLINE] + ["pareto"], wide)
    files["tradeoff_svg"].write_text(svg(t, generated), encoding="utf-8")
    files["report_discord_md"].write_text("\n".join(disc), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUT / "report.json", report)
    print(f"wrote {OUT / 'report.md'}, report.json, results.csv ({len(R.rows)} rows), results_headline.csv, tradeoff.svg, report_discord.md "
          f"({len(d.models)} models)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
