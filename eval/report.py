"""Collect every score into one comparison: report.md, report.json, the CSV exports, tradeoff.svg, report_discord.md.

Usage: python3 -I eval report

Reads only data/eval/scores/<model>/<suite>.json, runs/*/*/run.json, attacks/<model>/<suite>.json, phi/<model>/<suite>.json,
drift.json, suites/*/suite.json, gold/aci_eval/merged_stats.json, data/converted/*/manifest.json (dataset origins),
eval/models.toml and eval/coders.toml. Data may be partial: a
missing value renders as "–", a failed run as "failed", a failed coder or attack as "error", and nothing crashes.

Writes into data/eval/ (aggregate numbers only, no leaked strings, so every file is safe to paste into a chat or a PR):

  report.md             every table; the headline is aci_eval, identity at the bottom as the reference
  report.json           the same numbers, plus "exports" (the files below)
  results.csv           long format, one number per row: model, suite, category, metric, value, n, ci_low, ci_high,
                        coder, level, notes (UTF-8 with BOM, comma). Every number of report.md is a row, plus the
                        per-category/field/language breakdowns the markdown leaves out. Full stored precision.
  results_headline.csv  wide: one row per model, one column per headline metric
  tradeoff.svg          consensus privacy vs ICD F1 on the anonymized text (primary coder of coders.toml; ICD concepts kept
                        while no model has a result from it), Pareto-optimal models filled, identity as a square
  tradeoff_adversarial.svg  the same with x = 1 − the LLM agent adversary's proportion of recovered information (attack A6)

report.md has two dataset blocks (ACI, then Kaggle), then robustness, drift and efficiency, a glossary of every
column (COLUMNS: header, definition, task criterion; every table takes its headers from it) and the sources.
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
from eval.common import CATEGORY_RISK, DATA, HERE, OUT, REPO, write_json
from eval.run import VRAM_BUDGET_MB, load_models

MODELS_TOML = HERE / "models.toml"
CODERS_TOML = HERE / "coders.toml"
REFERENCE = ("identity",)
HEAD = "aci_eval"  # the headline suite
MAIN = ("aci_eval", "aci_dev")
DASH = "–"
CSV_FIELDS = ("model", "suite", "category", "metric", "value", "n", "ci_low", "ci_high", "coder", "level", "notes")
CONSENSUS = "consensus"


class Column(NamedTuple):
    """One report column: its header everywhere, its glossary definition and the task criterion it answers."""

    header: str
    definition: str
    criterion: str = ""


TREE = "tree"  # the decision-tree coder; its own headline is the category (3-character) level
ATTACKER = "A6"  # the LLM agent adversary (bonus/adversary.py)
COLUMNS: dict[str, Column] = {
    "privacy": Column("Proportion of removed privacy-related information",
                      "Share of gold PII occurrences no longer in the output (case-insensitive whole words; a multi-word item also "
                      "leaks when one of its distinctive tokens survives). On aci_eval: the consensus level of the merged gold "
                      "(direct identifiers or items listed by 2+ of the 3 golds); elsewhere every gold item", "i.a"),
    "direct": Column("removed: direct identifiers", "The same over direct identifiers only (names, DOB, dates, IDs, contacts): a gate", "i.a"),
    "high_risk": Column("removed: high-risk", "The same over high-risk categories (names, DOB, dates, ages, IDs, contacts)", "i.a"),
    "entity": Column("removed: entity-level", "Share of gold items with no occurrence and no distinctive token left (strict entity matching)", "i.a"),
    "union": Column("removed: union", "Occurrence removal over every item any gold lists, single-vote quasi-identifiers included (pessimistic)", "i.a"),
    "privacy_harness": Column("removed: harness gold", "Occurrence removal on the A+D team's own gold alone (reproduces their 0.839 for ad_night)", "i.a"),
    "concepts": Column("Medical concept recall", "Share of gold medical-concept occurrences kept in the output", "ii.b"),
    "icd_concepts": Column("Medical concept recall (ICD-relevant)", "The same over concepts relevant to the encounter's gold codes", "ii.b"),
    "keep": Column("keep traps kept", "Share of must-survive strings kept (eponyms such as Parkinson's, clinical numbers, code evidence)", "ii.b"),
    "terms": Column("ICD-10-CM terms kept", "ICD-10-CM description words outside gold PII kept in the output: a medical vocabulary for suites "
                    "without concept gold", "ii.b"),
    "non_pii_kept": Column("non-PII tokens kept", "Share of word tokens outside gold PII that a difflib alignment keeps (exhaustive golds only)", "ii"),
    "edit_precision": Column("edit precision", "Changed tokens that were gold PII / all changed tokens (low = harmless text removed)", "ii"),
    "placeholders": Column("placeholders /1k words", "[UPPER_CASE] tags per 1,000 output words: over-redaction without gold", ""),
    "valid": Column("valid", "Share of records whose output keeps the schema, ids, order and icd10_codes byte-identical (must be 1)", ""),
    "icd_f1_anon": Column("F1-Score ICD-10 code prediction (LLM judge)",
                          "Micro F1 of the LLM judge's codes (llm_judge: Claude Sonnet 5.5 agents coding each note from a prompt) on the "
                          "anonymized text vs the gold codes, exact code level", "ii.a"),
    "icd_f1_delta": Column("Δ F1 (LLM judge)", "LLM judge: anonymized minus original micro F1 (exact codes); 0 = no loss. Its paired-bootstrap "
                           "CI95 is in the ICD table", "ii.a"),
    "icd_agree": Column("ICD agreement (LLM judge)", "LLM judge: anonymized-text codes scored against original-text codes as gold: the change "
                        "anonymization alone causes", "ii.a"),
    "tree_f1_anon": Column("F1-Score ICD-10 code prediction (decision tree)",
                           "Micro F1 of the decision-tree coder (decision-models, fit on the 140 ACI train encounters) on the "
                           "anonymized text vs the gold codes, at its own 3-character category level", "ii.a"),
    "tree_f1_delta": Column("Δ F1 (decision tree)", "Decision tree: anonymized minus original category-level micro F1; 0 = no loss", "ii.a"),
    "tree_agree": Column("ICD agreement (decision tree)", "Decision tree: anonymized-text categories scored against original-text categories", "ii.a"),
    "f1_orig": Column("F1 original", "The coder's micro F1 on the original text: its own quality, the ceiling", "ii.a"),
    "f1_delta": Column("Δ F1 [CI95]", "Anonymized minus original micro F1, paired bootstrap CI95 (1,000 resamples of the encounters)", "ii.a"),
    "cat_f1": Column("category F1 (anonymized)", "Anonymized-text micro F1 with codes cut to their 3-character category", "ii.a"),
    "cat_f1_delta": Column("category Δ F1 [CI95]", "The same change at the category level (the decision tree's own level), paired bootstrap CI95", "ii.a"),
    "icd_errors": Column("errors", "Predictions that failed after retries (scored as empty sets)", ""),
    "recovered": Column("Proportion of recovered information",
                        "Share of gold items (or name tokens, rare words) an attack recovers from the output; beside it, the same "
                        "attack on the original text (positive control)", "iv.a.ii"),
    "recovered_llm": Column("Proportion of recovered information (LLM adversary)",
                            "Attack A6: an LLM agent, prompted as a red-team auditor, reads one model's blinded output and guesses each "
                            "patient's name, age, place and job; share of gold values hit (any name token, age within 1 year, a gold place "
                            "or job inside the guess). Beside it the same agent prompt on the original text (control)", "iv.a.ii"),
    "attack_max": Column("Proportion of recovered information (max)", "Largest recovered share over the attacks run, A5 record linkage "
                         "excluded (an upper bound by design)", "iv.a.ii"),
    "explained": Column("changed tokens explained", "Share of changed tokens covered by a logged decision with a reason", "iv.b.i"),
    "ms_per_encounter": Column("Runtime (ms/encounter)", "Wall time of the model process / records, model loading included", "iv.c.i"),
    "peak_ram_mb": Column("Peak RAM (MB)", "Peak summed RSS of the model's process tree", "iv.c.ii"),
    "peak_vram_mb": Column("Peak VRAM (MB)", "Peak summed nvidia-smi memory of the process tree, CUDA context included", "iv.c.iii"),
    "vram_ok": Column(f"Peak VRAM ≤ {VRAM_BUDGET_MB} MB", "Whether peak VRAM stays within the challenge's 4 GB budget", "iv.c.iii"),
    "probe_vram": Column("Peak VRAM incl. Ollama (MB)", "Probe (`eval probe`): peak nvidia-smi memory of the model's processes plus the Ollama "
                         "server's while it anonymizes the first records of the suite, CUDA context included; counts the LLM of a model that "
                         "runs it in Ollama, which the run's own VRAM column cannot see", "iv.c.iii"),
    "probe_latency": Column("Latency (ms/encounter)", "Probe: the time one record adds once the model is loaded, (wall time on the first N "
                            "records − on the first record) / (N − 1)", "iv.c.i"),
    "probe_startup": Column("Startup (s)", "Probe: process start and model loading, wall time on one record minus one latency", "iv.c.i"),
    "pareto": Column("Pareto", "No other real model is at least as good on both axes and better on one (identity, the unchanged input, is a reference)", ""),
    "phi_before": Column("PII before (Phi)", "PII entities the fine-tuned Phi-3 PII model (ab-ai/PII-Model-Phi3-Mini) finds in the original "
                         "text, an independent detector", "i.a"),
    "phi_after": Column("PII after (Phi)", "PII entities the same detector finds in the anonymized text (placeholders ignored)", "i.a"),
    "phi_removed": Column("removed (Phi)", "1 − after / before: privacy measured by an independent detector instead of the gold", "i.a"),
}


def H(key: str) -> str:
    return COLUMNS[key].header


class Col(NamedTuple):
    key: str
    category: str
    metric: str
    level: str = ""
    coder: str = ""  # "*": the primary ICD coder of coders.toml; else a coder name
    digits: int | str = 3  # "bool": a yes/no cell (1/0 in the CSV); "signed": a change, +0.000

    @property
    def header(self) -> str:
        return H(self.key)


HEADLINE = (
    Col("privacy", "privacy", "removed", CONSENSUS),
    Col("direct", "privacy", "removed", "direct"),
    Col("entity", "privacy", "entity_removed", CONSENSUS),
    Col("privacy_harness", "privacy", "removed", "gold:harness"),
    Col("concepts", "concepts", "kept"),
    Col("icd_concepts", "concepts", "icd_relevant"),
    Col("icd_f1_anon", "icd", "anonymized.code.f1", coder="*"),
    Col("icd_f1_delta", "icd", "f1_delta", coder="*", digits="signed"),
    Col("icd_agree", "icd", "agreement.code.f1", coder="*"),
    Col("tree_f1_anon", "icd", "anonymized.category.f1", coder=TREE),
    Col("tree_f1_delta", "icd", "category.f1_delta", coder=TREE, digits="signed"),
    Col("recovered_llm", "attack", f"{ATTACKER}.anonymized.recovered"),
    Col("attack_max", "attack", "max_recovered"),
    Col("explained", "explain", "explained"),
    Col("ms_per_encounter", "efficiency", "ms_per_encounter", digits=1),
    Col("peak_ram_mb", "efficiency", "peak_ram_mb", digits=1),
    Col("vram_ok", "efficiency", "within_vram_budget", digits="bool"),
    Col("probe_latency", "efficiency", "probe.latency_ms_per_encounter", digits=1),
    Col("probe_vram", "efficiency", "probe.total_peak_vram_mb", digits=1),
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
        self.phi = load_tree(OUT / "phi")  # bonus/phi_privacy.py: PII found by an independent detector before and after
        self.probe = load_tree(OUT / "probe")  # bonus/probe.py: VRAM with the Ollama server, and latency, on a few records
        self.manifest: dict[str, dict] = {}  # the converter manifest behind each suite (its origin, for the sources)
        for s_, i in self.info.items():
            if i.get("source") and isinstance(mf := read_json(DATA / "converted" / i["source"] / "manifest.json"), dict):
                self.manifest[s_] = mf
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
    R.add(m, s, "icd", "category.f1_delta", g(res, "f1_delta_category", "value"), n=n, ci=g(res, "f1_delta_category", "ci95"), coder=coder,
          notes="anonymized minus original category F1, paired bootstrap CI95")  # fmt: skip
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
PROBE_KEYS = ("latency_ms_per_encounter", "startup_s", "peak_vram_mb", "ollama_peak_vram_mb", "total_peak_vram_mb", "within_vram_budget", "peak_ram_mb",
              "ollama_peak_ram_mb", "records")  # fmt: skip


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
            if pr := d.probe.get(m, {}).get(s):
                for k in PROBE_KEYS:
                    R.add(m, s, "efficiency", f"probe.{k}", pr.get(k), n=pr.get("records") if k == "latency_ms_per_encounter" else None,
                          notes="probe (eval probe): " + ", ".join(f"{e}={v}" for e, v in (pr.get("ollama_env") or {}).items()) if k == "total_peak_vram_mb" else "")
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
    if digits == "signed":
        return signed(x)
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


def coder_of(d: Data, c: Col) -> str:
    return (d.primary or "") if c.coder == "*" else c.coder


def value(d: Data, R: Results, m: str, c: Col, suite: str = HEAD):
    v = R.get(m, suite, c.category, c.metric, coder=coder_of(d, c), level=c.level)
    return (bool(v) if v is not None else None) if c.digits == "bool" else v


def headline(d: Data, R: Results, m: str) -> dict:
    return {c.key: value(d, R, m, c) for c in HEADLINE}


def headline_cell(d: Data, m: str, c: Col, v) -> str:
    if v is not None:
        return fmt(v, c.digits)
    if d.failed(m, HEAD):
        return "failed"
    if c.coder and g(d.score(m, HEAD), "icd", coder_of(d, c), "error"):
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


def tradeoff(d: Data, R: Results, adversarial: bool = False) -> dict:
    """x = consensus privacy (or, `adversarial`, 1 − the LLM adversary's proportion of recovered information), y = the primary
    coder's ICD F1 on the anonymized text, or ICD concepts kept when no real model has the coder."""
    real = [m for m in d.models if m not in REFERENCE]
    has = bool(d.primary) and any(R.get(m, HEAD, "icd", "anonymized.code.f1", coder=d.primary) is not None for m in real)
    if has:
        ykey, ylabel = ("icd", "anonymized.code.f1", d.primary), f"F1-Score ICD-10 code prediction (LLM judge {d.primary}, anonymized text)"
    else:
        ykey = ("concepts", "icd_relevant", "")
        ylabel = f"ICD concepts kept (fallback: no {d.primary or 'primary'} coder result yet)"
    if adversarial:
        xs = {m: None if (v := R.get(m, HEAD, "attack", f"{ATTACKER}.anonymized.recovered")) is None else round(1 - v, 6) for m in d.models}
        xname, xtitle = "1 − recovered (LLM adversary)", f"privacy against the LLM adversary (1 − proportion of recovered information, attack {ATTACKER})"
        title = f"Adversarial privacy vs ICD utility on {HEAD}"
    else:
        xs = {m: R.get(m, HEAD, "privacy", "removed", level=CONSENSUS) for m in d.models}
        xname, xtitle = "privacy (consensus)", "privacy, consensus gold level (share of PII occurrences removed)"
        title = f"Privacy vs ICD utility on {HEAD}"
    pts = {m: (xs[m], R.get(m, HEAD, ykey[0], ykey[1], coder=ykey[2])) for m in d.models}
    flags = pareto({m: p for m, p in pts.items() if m not in REFERENCE})
    return {"x": xname, "x_title": xtitle, "title": title, "y": ylabel, "y_metric": {"category": ykey[0], "metric": ykey[1], "coder": ykey[2]},
            "fallback": not has, "points": {m: {"x": x, "y": y, "reference": m in REFERENCE, "pareto": flags.get(m)} for m, (x, y) in pts.items()}}  # fmt: skip


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
         f"<title id=\"title\">{escape(t.get('title', 'Privacy vs utility'))}</title>",
         f"<style>@media (prefers-color-scheme: dark){{{css}}}</style>",
         f'<rect class="bg" width="{W}" height="{H}" fill="{LIGHT["bg"]}"/>',
         f'<text class="ink" x="{L}" y="28" font-size="15" font-weight="600" fill="{LIGHT["ink"]}">{escape(t.get("title", "Privacy vs utility"))}</text>',
         f'<text class="ink2" x="{L}" y="47" font-size="11.5" fill="{LIGHT["ink2"]}">Filled: Pareto-optimal among the real models. identity (input unchanged) is a reference, not a competitor. No combined score.</text>']  # fmt: skip
    lx = L  # legend
    for kind, label in (("pareto", "Pareto-optimal"), ("dominated", "dominated"), ("reference", "reference (identity)")):
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
             f"{escape(t.get('x_title', t['x']))}</text>")  # fmt: skip
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
        note, room = f"not plotted (no value): {', '.join(missing)}", int((W - L - 130) / 6.2)  # leave the timestamp its corner
        note = note if len(note) <= room else note[: room - 1] + "…"
        o.append(f'<text class="ink2" x="{L}" y="{H - 10}" font-size="11" fill="{LIGHT["ink2"]}">{escape(note)}<title>{escape(", ".join(missing))}</title></text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


# --- the report ----------------------------------------------------------------------------------------------------


def pareto_table(d: Data, R: Results, t: dict, xcol: str) -> list[str]:
    pts, prim = t["points"], d.primary
    order_ = sorted(d.models, key=lambda m: (m in REFERENCE, -(pts[m]["x"] if pts[m]["x"] is not None else -1), m))
    rows = []
    for m in order_:
        p = pts[m]
        flag = "ref" if p["reference"] else ("**yes**" if p["pareto"] else (DASH if p["pareto"] is None else "no"))
        rows.append([name(m), fmt(p["x"]), fmt(p["y"]), signed(R.get(m, HEAD, "icd", "f1_delta", coder=d.primary or "")),
                     fmt(R.get(m, HEAD, "icd", "anonymized.category.f1", coder=TREE)), fmt(R.get(m, HEAD, "concepts", "kept")), flag])  # fmt: skip
    ycol = H("icd_f1_anon") if not t["fallback"] else H("icd_concepts")
    return table(["model", xcol, ycol, H("icd_f1_delta"), H("tree_f1_anon"), H("concepts"), H("pareto")], rows)


def phi_rows(d: Data, suite: str) -> tuple[list[str], list[list[str]]]:
    """Header and rows of the independent Phi check on one suite: totals, then removed per category."""
    files = {m: d.phi.get(m, {}).get(suite) for m in d.models}
    cats_found = {c for f in files.values() if isinstance(f, dict) for c in (f.get("by_category") or {})}
    cats = [c for c in CATEGORY_RISK if c in cats_found] + sorted(cats_found - set(CATEGORY_RISK))
    rows = []
    for m, f in files.items():
        if not isinstance(f, dict):
            continue
        tot, bc = f.get("total") or {}, f.get("by_category") or {}
        rows.append([name(m), fmt(tot.get("before"), None), fmt(tot.get("after"), None), fmt(tot.get("removed"))]
                    + [fmt(g(bc, c, "removed")) for c in cats])  # fmt: skip
    return ["model", H("phi_before"), H("phi_after"), H("phi_removed")] + [f"{c}" for c in cats], rows


def render(d: Data, R: Results, nc: list[dict], generated: str) -> tuple[list[str], dict, dict, dict, list[str]]:
    models, prim = d.models, d.primary
    md = [f"# Anonymization evaluation ({generated})", ""]
    md += [f"- {name(m)}: {d.registry.get(m, {}).get('description', '')}" for m in models] + [""]
    if d.pending:
        md += ["Pending (disabled in `models.toml`, no data): " + ", ".join(f"`{m}`" for m in d.pending), ""]
    status = []
    if prim:
        without = [m for m in models if R.get(m, HEAD, "icd", "anonymized.code.f1", coder=prim) is None]
        if without:
            status.append(f"- LLM judge `{prim}`: no `{HEAD}` result yet for " + ", ".join(f"`{m}`" for m in without) + " (cells show –).")
    else:
        status.append("- No primary ICD coder in `coders.toml`: the LLM-judge columns are empty.")
    no_score = [m for m in models if d.score(m, HEAD) is None]
    if no_score:
        status.append(f"- No `{HEAD}` score yet: " + ", ".join(f"`{m}`" for m in no_score) + ".")
    no_adv = [m for m in models if d.score(m, HEAD) is not None and R.get(m, HEAD, "attack", f"{ATTACKER}.anonymized.recovered") is None]
    if no_adv:
        status.append(f"- No LLM-adversary ({ATTACKER}) result yet: " + ", ".join(f"`{m}`" for m in no_adv) + ".")
    failed = sorted({f"`{m}` on `{s}`" for m in models for s in d.suites if d.failed(m, s)})
    if failed:
        status.append("- Failed runs (scored as failed): " + ", ".join(failed) + ".")
    if not d.drift:
        status.append("- No `drift.json` yet (`python3 -I eval drift`): complexity ranks are empty.")
    if status:
        md += ["Data status:", ""] + status + [""]
    md += ["Two datasets: **ACI** (the challenge's own data, `aci_eval`: privacy, ICD-10 F1, attacks, ablation) first, then the **Kaggle** "
           "complement (privacy and ICD agreement; no F1, most Kaggle suites have no gold codes). Column names follow the task PDF "
           "(criteria in the [Glossary](#glossary)); data origins and licenses are under [Sources](#sources).", ""]  # fmt: skip

    # ============================================================== ACI
    md += [f"# ACI (`{HEAD}`)", ""]
    heads = {m: headline(d, R, m) for m in models}
    section(md, "Headline")
    md += table(["model"] + [c.header for c in HEADLINE], [[name(m)] + [headline_cell(d, m, c, heads[m][c.key]) for c in HEADLINE] for m in models])
    md += ["Shares: 1 = everything removed, kept or recovered. LLM judge = the primary coder `" + (prim or DASH) + "`; the decision tree is "
           "scored at its own 3-character category level. Runtime, RAM and VRAM are from the `" + HEAD + "` run (– for a delivered JSONL); latency and "
           "peak VRAM incl. Ollama are from the probe (`eval probe`, see VRAM and latency probe), which also counts the LLM a model runs in Ollama. "
           "`identity` (input unchanged) is a reference, not a competitor.", ""]  # fmt: skip

    t = tradeoff(d, R)
    section(md, "Privacy vs ICD utility (Pareto front)",
            f"x = {t['x_title']}; y = {t['y']}. Pareto-optimal: no other real model is at least as good on both axes and better on one. "
            "No combined score: the organizers weigh the criteria separately."
            + (f" No model has a `{prim}` result yet, so the utility axis is {H('icd_concepts')}." if t["fallback"] else ""),
            "![privacy vs utility](tradeoff.svg)")  # fmt: skip
    md += pareto_table(d, R, t, H("privacy"))
    ta = tradeoff(d, R, adversarial=True)
    section(md, "Adversarial privacy vs ICD utility (Pareto front)",
            f"x = {ta['x_title']}; y = {ta['y']}. The same front with privacy measured by what the LLM adversary recovers instead of "
            "the gold.", "![adversarial privacy vs utility](tradeoff_adversarial.svg)")  # fmt: skip
    md += pareto_table(d, R, ta, f"1 − {H('recovered_llm')}")

    # privacy levels and categories
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
    section(md, f"{H('privacy')} by gold level",
            "Occurrence-level removal per gold level (union: any gold; consensus: direct identifiers or 2+ votes, the headline; direct: "
            "names, dates, DOB, IDs, contacts), the high-risk share, and entity-level removal (an item counts only when every "
            "occurrence is gone).", *gl)  # fmt: skip
    md += table(["model", H("union"), f"{H('privacy')} (consensus)", H("direct"), H("high_risk"), "entity: union", "entity: consensus", "entity: direct"],
                lv_rows)  # fmt: skip
    md += ["Votes and sources: `votes:N` = union items listed by N golds; `src:X` = union items gold X supports; `gold X` = gold X scored "
           "alone on its own list (`gold harness` is the A+D team's cross-check value).", ""]  # fmt: skip
    md += table(["model", "votes:1", "votes:2", "votes:3", "src:harness", "src:vm", "src:ketai", "gold harness", "gold vm", "gold ketai"], src_rows)
    cats_found = {r["metric"] for r in R.rows if r["suite"] == HEAD and r["category"] == "privacy_by_category"}
    cats = [c for c in CATEGORY_RISK if c in cats_found] + sorted(cats_found - set(CATEGORY_RISK))
    rows = []
    for m in models:
        cells = [fmt(R.get(m, HEAD, "privacy_by_category", c, level=CONSENSUS)) for c in cats]
        cells += [fmt(R.get(m, HEAD, "privacy_by_field", f, level=CONSENSUS)) for f in ("dialogue", "note")]
        if any(c != DASH for c in cells):
            rows.append([name(m)] + cells)
    if rows:
        section(md, f"{H('privacy')} by category and field (consensus level)")
        md += table(["model"] + cats + ["dialogue", "note"], rows)
    hdr, rows = phi_rows(d, HEAD)
    if rows:
        section(md, "Independent privacy check (Phi-3 PII model), before and after anonymization",
                "PII entities the fine-tuned `ab-ai/PII-Model-Phi3-Mini` finds in the original text (before) and in the output (after), "
                "per category mapped to the gold taxonomy; removed = 1 − after / before. Needs no gold, so it also catches PII the golds "
                "missed, and counts what the detector itself misses as kept.")  # fmt: skip
        md += table(hdr, rows)

    # ICD coders, F1 on ACI only
    def icd_table(suite: str) -> list[str]:
        rows, errs = [], {}
        for m in models:
            block = g(d.score(m, suite), "icd") or {}
            for c in d.coder_names + [c for c in block if c not in d.coder_names]:
                res = block.get(c)
                if not isinstance(res, dict):
                    continue
                lead = [name(m), f"`{c}`" + (" !" if res.get("valid_for_comparison") is False else "")]
                if res.get("error"):
                    errs.setdefault(short(res["error"]), set()).add(c)
                    rows.append(lead + ["error"] + [""] * 6)
                    continue

                def v(metric: str, m=m, c=c):
                    return R.get(m, suite, "icd", metric, coder=c)

                def change(metric: str, m=m, c=c) -> str:
                    dr = R.row(m, suite, "icd", metric, coder=c)
                    return signed(dr["value"]) + (f" [{signed(dr['ci_low'])}, {signed(dr['ci_high'])}]" if dr["ci_low"] is not None else "") if dr else DASH

                rows.append(lead + [fmt(v("original.code.f1")), fmt(v("anonymized.code.f1")), change("f1_delta"), fmt(v("agreement.code.f1")),
                                    fmt(v("anonymized.category.f1")), change("category.f1_delta"), fmt(v("errors"), None)])  # fmt: skip
        out = table(["model", "coder", H("f1_orig"), "F1 anonymized", H("f1_delta"), "agreement", H("cat_f1"), H("cat_f1_delta"), H("icd_errors")], rows, left=2)
        return out + [f"- error ({', '.join(f'`{c}`' for c in sorted(cs))}): {msg}" for msg, cs in errs.items()] + ([""] if errs else [])

    section(md, "F1-Score ICD-10 code prediction, every coder",
            "Micro F1 of each coder's codes vs the gold codes on the original and the anonymized text (code level; category = 3 "
            "characters), the paired-bootstrap change Δ = anonymized minus original with its CI95, agreement = anonymized vs "
            f"original predictions (the damage done by anonymization alone). LLM judge: `{prim or DASH}`; decision tree: `{TREE}` (read "
            "its category column). `!` = not valid for comparison (a prediction failed: counted as empty, see errors). F1 is reported on "
            f"`{HEAD}` only.")  # fmt: skip
    md += icd_table(HEAD) if any(g(d.score(m, HEAD), "icd") for m in models) else ["No ICD results yet.", ""]

    # main suites, utility and edits
    cols = [("valid", "validity", "valid_rate", ""), ("privacy", "privacy", "removed", CONSENSUS), ("high_risk", "privacy", "removed_high_risk", CONSENSUS),
            ("keep", "keep", "kept", ""), ("non_pii_kept", "edits", "non_pii_tokens_kept", ""), ("edit_precision", "edits", "edit_precision", ""),
            ("concepts", "concepts", "kept", ""), ("icd_concepts", "concepts", "icd_relevant", ""), ("terms", "terms", "kept", ""),
            ("placeholders", "edits", "placeholders_per_1k_words", ""), ("explained", "explain", "explained", "")]  # fmt: skip
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
            md += table(["model"] + [H(k) for k, *_ in cols], rows)

    # attacks
    at_suites = [s for s in d.suites if any(s in d.attacks.get(m, {}) for m in models)]
    if HEAD in at_suites:
        keys = sorted({k for m in models for k in (g(d.attacks.get(m, {}).get(HEAD), "attacks") or {})})
        names_, llms = {}, set()
        for m in models:
            for k, a in (g(d.attacks.get(m, {}).get(HEAD), "attacks") or {}).items():
                if isinstance(a, dict) and a.get("name"):
                    names_.setdefault(k, a["name"])
                if k == ATTACKER and isinstance(a, dict) and a.get("llm"):
                    llms.add(a["llm"])
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
                    cells.append(f"{fmt(an)} / {fmt(co)}" + (f" ({a['llm']})" if k == ATTACKER and len(llms) > 1 and a.get("llm") else ""))
            rows.append([name(m)] + cells + [fmt(R.get(m, HEAD, "attack", "max_recovered"))])
        section(md, f"{H('recovered')} (adversarial attacks, bonus iv.a)",
                "Each cell: proportion recovered from the output / from the original text (the positive control). Lower is better on "
                f"the left; an attack that recovers little from the original says little. {ATTACKER} adversary: "
                + (", ".join(f"`{x}`" for x in sorted(llms)) or DASH) + ". A5 links records with the originals in hand, an upper bound "
                "by design, so the max leaves it out. – = not run.")  # fmt: skip
        md += table(["model"] + [f"{k} {names_.get(k, '')}".strip() for k in keys] + [H("attack_max")], rows)

    # ablation: with and without ICD-10 codes at inference
    abl = [c for c in HEADLINE if c.key in ("privacy", "direct", "concepts", "icd_concepts", "icd_f1_anon", "tree_f1_anon", "recovered_llm")]
    prs = [(b, n) for b, n in nocodes.pairs(d.registry) if d.score(b, HEAD) is not None or d.score(n, HEAD) is not None]
    if prs:
        rows = []
        for c in abl:
            cells = []
            for b, n in prs:
                vb, vn = value(d, R, b, c), value(d, R, n, c)
                cells += [fmt(vb), fmt(vn), signed(None if vb is None or vn is None else round(vn - vb, 4))]
            rows.append([c.header] + cells)
        lone = [m for m in models if m not in REFERENCE and not any(m in pr for pr in prs)]
        blind = [m for m in lone if d.registry.get(m, {}).get("reads_codes") is False]
        missing = [m for m in lone if m not in blind]
        section(md, "Without ICD-10 codes at inference (ablation, bonus iv.d)",
                f"Each model run with `blind_codes` (the codes hidden at inference, restored before scoring) beside the same model with "
                f"codes, on `{HEAD}`; Δ = without minus with. The task asks for the same metrics as the main task."
                + (" Never read the codes, so identical without them: " + ", ".join(f"`{m}`" for m in blind) + "." if blind else "")
                + (" No twin run: " + ", ".join(f"`{m}`" for m in missing) + "." if missing else ""))  # fmt: skip
        md += table(["metric"] + [h for b, n in prs for h in (f"`{b}`", f"`{n}`", "Δ")], rows)

    # ============================================================== Kaggle
    ks, medical = d.suites_of("kaggle", "generic"), d.suites_of("kaggle")
    if ks:
        md += ["# Kaggle complement", "",
               "Kaggle datasets converted to the challenge schema (`DATA_PREP.md`); their golds are partial or single-source, so they "
               "complement the ACI headline. `generic` suites are not medical; the mean is over the medical ones.", ""]  # fmt: skip
        for title, fam, key, lv in ((H("privacy"), "privacy", "removed", CONSENSUS), (f"{H('terms')} (clinical vocabulary)", "terms", "kept", "")):
            rows = []
            for m in models:
                cells = [("failed" if d.failed(m, s) else fmt(R.get(m, s, fam, key, level=lv))) for s in ks]
                if any(c != DASH for c in cells):
                    rows.append([name(m)] + cells + [fmt(R.get(m, "kaggle_medical_mean", fam, key, level=lv))])
            section(md, title)
            md += table(["model"] + [f"`{s}`" for s in ks] + ["mean (medical)"], rows)
        for c in d.coder_names:
            level = "category" if c == TREE else "code"
            cols_ = [s for s in medical if any(R.get(m, s, "icd", f"agreement.{level}.f1", coder=c) is not None for m in models)]
            rows = []
            for m in models:
                cells = [fmt(R.get(m, s, "icd", f"agreement.{level}.f1", coder=c)) for s in cols_]
                if any(x != DASH for x in cells):
                    rows.append([name(m)] + cells)
            if rows:
                section(md, f"ICD agreement, `{c}` ({level} level)",
                        "Codes predicted on the anonymized text scored against the codes predicted on the original text (no gold codes "
                        "needed): 1 = anonymization changed no prediction.")  # fmt: skip
                md += table(["model"] + [f"`{s}`" for s in cols_], rows)
        ucols = [("NBME graded findings kept", "nbme_notes", "keep", "kept", ""), ("devp code evidence kept", "devp_discharge", "keep", "kept", ""),
                 (f"devp {H('high_risk')}", "devp_discharge", "privacy", "removed_high_risk", CONSENSUS)]  # fmt: skip
        rows = [[name(m)] + [fmt(R.get(m, s, fam, key, level=lv)) for _, s, fam, key, lv in ucols] for m in models]
        rows = [r for r in rows if any(c != DASH for c in r[1:])]
        if rows:
            section(md, "Utility with gold")
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
            section(md, f"{H('privacy')} by language (iMerit scribe: patient, physician, DOB)")
            md += table(["model", "en"] + langs, rows)
        for s in ks:
            hdr, rows = phi_rows(d, s)
            if rows:
                section(md, f"Independent privacy check (Phi-3 PII model), `{s}`")
                md += table(hdr, rows)
    at_more = [s for s in at_suites if any(s in d.attacks.get(m, {}) for m in models if m not in REFERENCE)]
    if [s for s in at_more if s != HEAD]:
        rows = [[name(m)] + [fmt(R.get(m, s, "attack", "max_recovered")) for s in at_more] for m in models if any(s in d.attacks.get(m, {}) for s in at_more)]
        section(md, f"{H('attack_max')} per suite")
        md += table(["model"] + [f"`{s}`" for s in at_more], rows)

    # ============================================================== bonus and context
    md += ["# Robustness, complexity and efficiency", ""]
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
        section(md, "Robustness to unseen formats (bonus iv.e)",
                f"{H('privacy')} (consensus), {H('high_risk')} and {H('terms')} on the robust suites. Rank = complexity rank from "
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

    rows = []
    for m in models:
        if R.get(m, "all", "efficiency", "suites") is None:
            continue
        rows.append([name(m)] + [fmt(R.get(m, "all", "efficiency", k), dg) for k, dg in
                                 (("suites", None), ("n_records", None), ("wall_s", 1), ("ms_per_encounter", 1), ("peak_ram_mb", 1), ("peak_vram_mb", 1),
                                  ("within_vram_budget", "bool"))])  # fmt: skip
    if rows:
        section(md, "Efficiency over all suites run (bonus iv.c)")
        md += table(["model", "suites", "records", "wall s", H("ms_per_encounter"), H("peak_ram_mb"), H("peak_vram_mb"), H("vram_ok")], rows)

    probed = [m for m in models if d.probe.get(m, {}).get(HEAD)]
    if probed:
        pr0 = d.probe[probed[0]][HEAD]
        env = ", ".join(f"`{k}={v}`" for k, v in (pr0.get("ollama_env") or {}).items())
        section(md, "VRAM and latency probe (bonus iv.c)",
                f"`python3 -I eval probe`: every runnable model on the first {pr0.get('records')} records of `{HEAD}` and on the first record alone, "
                f"after every Ollama model is unloaded, on {pr0.get('gpu') or 'no GPU'}" + (f" (Ollama server: {env})" if env else "") + ". VRAM and RAM "
                "are sampled every 0.2 s for the model's processes and, apart, for the Ollama server's, so a model whose LLM runs in Ollama "
                "(the run's own VRAM column shows 0 for it) pays for it here. Latency = (wall time on N records − on 1) / (N − 1); startup = "
                "wall time on 1 record − one latency. A delivered JSONL has nothing to run.")  # fmt: skip
        rows = [[name(m)] + [fmt(R.get(m, HEAD, "efficiency", f"probe.{k}"), dg) for k, dg in
                             (("peak_vram_mb", 1), ("ollama_peak_vram_mb", 1), ("total_peak_vram_mb", 1), ("within_vram_budget", "bool"), ("peak_ram_mb", 1),
                              ("ollama_peak_ram_mb", 1), ("latency_ms_per_encounter", 1), ("startup_s", 2))] for m in probed]  # fmt: skip
        md += table(["model", "VRAM, model (MB)", "VRAM, Ollama (MB)", H("probe_vram"), H("vram_ok"), "RAM, model (MB)", "RAM, Ollama (MB)", H("probe_latency"),
                     H("probe_startup")], rows)  # fmt: skip

    md += glossary(d) + sources(d)

    icd_suites = [s for s in d.suites if any(g(d.score(m, s), "icd") for m in models)]
    report = {
        "generated": generated, "headline_suite": HEAD, "primary_coder": prim,
        "models": {m: d.registry.get(m, {}).get("description", "") for m in models}, "pending": d.pending, "suites": d.info,
        "columns": {k: c._asdict() for k, c in COLUMNS.items()},
        "headline": heads, "tradeoff": t, "tradeoff_adversarial": ta,
        "icd": {s: {m: {c: {k: R.get(m, s, "icd", k, coder=c) for k in ("original.code.f1", "anonymized.code.f1", "f1_delta", "category.f1_delta",
                                                                          "agreement.code.f1", "anonymized.category.f1", "agreement.category.f1",
                                                                          "original.category.f1", "errors", "valid_for_comparison")}
                        for c in (g(d.score(m, s), "icd") or {}) if not g(d.score(m, s), "icd", c, "error")} for m in models if g(d.score(m, s), "icd")}
                for s in icd_suites},
        "nocodes": nc,
        "attacks": {s: {m: {"max_recovered": R.get(m, s, "attack", "max_recovered"),
                            **{k: {"anonymized": R.get(m, s, "attack", f"{k}.anonymized.recovered"), "control": R.get(m, s, "attack", f"{k}.control.recovered")}
                               for k in (g(d.attacks.get(m, {}).get(s), "attacks") or {})}} for m in models if s in d.attacks.get(m, {})} for s in at_suites},
        "phi": {s: {m: d.phi[m][s].get("total") for m in models if s in d.phi.get(m, {})} for s in d.suites if any(s in d.phi.get(m, {}) for m in models)},
        "probe": {m: {k: R.get(m, HEAD, "efficiency", f"probe.{k}") for k in PROBE_KEYS} for m in models if d.probe.get(m, {}).get(HEAD)},
        "robustness": {s: {"complexity_rank": ranks.get(s), "models": {m: {k: R.get(m, s, "robustness", k) for k in ("privacy", "high_risk", "terms")} for m in models}}
                       for s in robust},
        "complexity_rank": ranks,
        "efficiency": {m: {k: R.get(m, "all", "efficiency", k) for k in ("suites", "n_records", "wall_s", "ms_per_encounter", "peak_ram_mb", "peak_vram_mb",
                                                                         "within_vram_budget")} for m in models},
    }  # fmt: skip

    disc = discord(d, heads, t, generated)
    return md, report, t, ta, disc


ACI_CITATION = ("Yuan, Moy, et al. \"Toward reliable clinical coding with language models: verification and lightweight adaptation.\" "
                "EMNLP 2025 Industry Track (the ACI-Bench encounters with ICD-10 codes; entries unmodified)")  # fmt: skip


def glossary(d: Data) -> list[str]:
    rows = [[c.header, c.definition, c.criterion or DASH] for c in COLUMNS.values()]
    md = ["# Glossary", "",
          "Every column of this report: what it measures and the task criterion it answers (task PDF, *Evaluation*: i privacy, "
          "ii medical information, iv bonus tasks). Full computation rules: `eval/docs/METRICS.md`. Every number is also a row of "
          "`results.csv`; `results_headline.csv` is the headline.", ""]  # fmt: skip
    md += table(["column", "definition", "criterion"], rows, left=3)
    md += ["- `identity` (input unchanged: privacy 0, utility 1) is a reference that calibrates the scale, not a model.",
           "- All three ACI golds are LLM-assisted, so their votes are less independent than they look.", ""]  # fmt: skip
    return md


def sources(d: Data) -> list[str]:
    rows = []
    for s in d.suites:
        info, mf = d.info.get(s) or {}, d.manifest.get(s) or {}
        origin = str(mf.get("source") or info.get("source") or DASH)
        if origin.startswith("kaggle:"):
            slug = origin.split()[0].removeprefix("kaggle:")
            origin = f"[{slug}](https://www.kaggle.com/datasets/{slug})" + origin[len("kaggle:") + len(slug):]
        elif info.get("source") == "aci_bench":
            origin = ACI_CITATION
        rows.append([f"`{s}`", d.role(s) or DASH, fmt(info.get("records"), None), origin.replace("|", "/"), str(info.get("license") or DASH),
                     ", ".join(info.get("gold_sources") or []) or DASH])  # fmt: skip
    md = ["# Sources", "", "## Datasets", ""]
    md += table(["suite", "role", "records", "origin", "license", "gold"], rows, left=6)
    used = {c for per in d.scores.values() for sc in per.values() for c in (g(sc, "icd") or {})}
    tools = [[f"`{c}`", (d.coders.get(c) or {}).get("description", "")] for c in d.coder_names if c in d.coders and c in used]
    llms = sorted({a["llm"] for per in d.attacks.values() for at in per.values() for a in [g(at, "attacks", ATTACKER)] if isinstance(a, dict) and a.get("llm")})
    tools += [[f"`{x}` ({ATTACKER})", "LLM agent adversary: prompted to guess name, age, place and job from one blinded model output"] for x in llms]
    phi = sorted({f.get("phi_model") for per in d.phi.values() for f in per.values() if isinstance(f, dict) and f.get("phi_model")})
    tools += [[f"`{x}`", "Independent PII detector of the before/after privacy check (evaluation only)"] for x in phi]
    md += ["## Coders, adversary and detectors", ""] + table(["name", "what it is"], tools, left=2)
    md += ["- Gold: `harness` = the A+D team's annotation, `vm` = the anonymous VM's, `ketai` = Ketai's; merged by vote on `aci_eval` "
           "(`eval/docs/GOLD.md`).", "- Kaggle catalog with measured contents and licenses: `data/kaggle/DATASETS.md`.", ""]  # fmt: skip
    return md


def discord(d: Data, heads: dict, t: dict, generated: str) -> list[str]:
    prim, models = d.primary, d.models
    cols = [("privacy", "privacy"), ("direct", "direct"), ("privacy_harness", "harness gold"), ("icd_concepts", "ICD concepts"),
            ("icd_f1_anon", "F1 (LLM judge)"), ("icd_f1_delta", "Δ F1 (LLM judge)"), ("tree_f1_anon", "F1 (tree)"), ("recovered_llm", "recovered (LLM adv.)"),
            ("ms_per_encounter", "ms/enc"), ("probe_vram", "VRAM MB"), ("vram_ok", "VRAM ≤ 4 GB")]  # fmt: skip
    digits = {c.key: c.digits for c in HEADLINE}
    rows = [[name(m)] + [fmt(heads[m][k], digits[k]) for k, _ in cols] for m in models]
    md = [f"**Anonymization eval, `{HEAD}`, same pipeline for every model** ({generated})", ""]
    md += table(["model"] + [h for _, h in cols], rows)
    real = [m for m in models if m not in REFERENCE]
    best = max((m for m in real if heads[m]["privacy"] is not None), key=lambda m: heads[m]["privacy"], default=None)
    opt = [m for m in real if t["points"][m]["pareto"]]
    ylab = "ICD concepts kept" if t["fallback"] else f"ICD F1 on the anonymized text ({prim})"
    md += [f"- Pareto-optimal on consensus privacy vs {ylab}: " + (", ".join(f"`{m}`" for m in opt) or DASH) + "."]
    if best:
        md += [f"- Highest consensus privacy: `{best}` {fmt(heads[best]['privacy'])}; `identity` (input unchanged) scores 0."]
    md += ["- privacy = share of gold PII occurrences removed at the consensus level of the merged ACI gold (direct = names, dates, DOB, "
           "IDs, contacts; harness gold = the A+D team's gold alone, their own cross-check)."]  # fmt: skip
    md += [f"- ICD: no `{prim}` results yet, utility shown as ICD concepts kept." if t["fallback"] else
           f"- F1 = ICD-10 code prediction on the anonymized text (LLM judge `{prim}`, decision tree at category level); Δ F1 = anonymized minus "
           "original (0 = no loss); recovered = share the LLM adversary "
           "re-identifies. Full tables: `report.md`."]  # fmt: skip
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
    md, report, t, ta, disc = render(d, R, nc, generated)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"results_csv": OUT / "results.csv", "results_headline_csv": OUT / "results_headline.csv", "tradeoff_svg": OUT / "tradeoff.svg",
             "tradeoff_adversarial_svg": OUT / "tradeoff_adversarial.svg", "report_discord_md": OUT / "report_discord.md"}  # fmt: skip
    report["exports"] = {k: rel(v) for k, v in files.items()}
    write_csv(files["results_csv"], list(CSV_FIELDS), R.rows)
    wide = [{"model": m, "reference": int(m in REFERENCE), **report["headline"][m],
             "pareto": None if t["points"][m]["pareto"] is None else int(t["points"][m]["pareto"])} for m in d.models]  # fmt: skip
    for row in wide:
        row["vram_ok"] = None if row["vram_ok"] is None else int(row["vram_ok"])
    write_csv(files["results_headline_csv"], ["model", "reference"] + [c.key for c in HEADLINE] + ["pareto"], wide)
    files["tradeoff_svg"].write_text(svg(t, generated), encoding="utf-8")
    files["tradeoff_adversarial_svg"].write_text(svg(ta, generated), encoding="utf-8")
    files["report_discord_md"].write_text("\n".join(disc), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUT / "report.json", report)
    print(f"wrote {OUT / 'report.md'}, report.json, results.csv ({len(R.rows)} rows), results_headline.csv, tradeoff.svg, tradeoff_adversarial.svg, "
          "report_discord.md "
          f"({len(d.models)} models)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
