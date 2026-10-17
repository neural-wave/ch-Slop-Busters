"""ICD-10 utility: run the coders of eval/coders.toml on the original and the anonymized text, cache, and score.

The organizers code the anonymized notes with their own fixed coder; we stand in with local coders and score each one
on the original and on the anonymized text of the same records, against the gold codes:

    original     coder quality on the untouched text (the ceiling for that coder)
    anonymized   the same on the anonymized text: the organizer-style number
    agreement    anonymized predictions against original predictions taken as gold: the damage done by
                 anonymization alone, whatever the coder's own errors
    f1_delta     anonymized minus original code-level micro F1, with a paired bootstrap CI over encounters
    f1_delta_category  the same at the category level (the decision tree's own level)

Each arm is scored at the code level (exact canonical codes, eval/coders/codes.py) and at the category level (first
3 characters), as micro precision/recall/F1, the mean per-note F1 and a bootstrap CI of the micro F1 (1000 resamples,
seed 0). Gold codes are normalized like predictions. A ranked coder (knn, lexical, command) is cut at k = the
record's number of gold codes, so it needs no threshold; an unranked one (the LLM coders) is scored on its whole set.
Only records with gold codes are scored. A coder never sees the gold codes. On a suite without any gold code an
unranked coder is still scored on the agreement arm alone (`agreement_only`: the Kaggle suites); a ranked one has no k
to cut at and is skipped.

A failed prediction counts as an empty set, is counted in `errors` (then `valid_for_comparison` is false) and is
never cached. A failure of the whole coder (Ollama unreachable, three consecutive failed predictions) gives
{"error": message} for that coder, and scoring goes on.

Predictions are cached in data/eval/cache/icd/<coder>/<sha1(config, encounter_id, the text fields it reads)>.json
as {"codes", "attempts", "seconds", "usage", "raw_invalid", "backend"} (backend: where an LLM ran, gpu, partial or cpu;
recorded per prediction only, not in the score); the plain code lists that scripts/eval wrote for knn and
lexical are read as {"codes": list}. The original text is coded once for every model, and an anonymized text that a
model left unchanged reuses that prediction.

Usage (from eval/score.py):
    name, result = icd.evaluate("llm_note", "aci_eval", original, anonymized)
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from eval.coders.base import Coder, CoderError, Prediction
from eval.coders.bootstrap import counts, paired, rates
from eval.coders.codes import category, count_invalid, normalize, normalize_set
from eval.coders.command import Command
from eval.coders.knn import Knn
from eval.coders.lexical import Lexical, label_space
from eval.coders.text import aci
from eval.common import CACHE_DIR, HERE, sha1_text

REGISTRY = HERE / "coders.toml"
KINDS = ("ollama", "knn", "lexical", "command", "agent")
ARMS = ("original", "anonymized", "agreement")
LEVELS = ("code", "category")


def load_registry(path: Path = REGISTRY) -> dict[str, dict]:
    with Path(path).open("rb") as fh:
        reg = tomllib.load(fh)
    for name, spec in reg.items():
        if spec.get("kind") not in KINDS:
            raise ValueError(f"coders.toml [{name}]: kind must be one of {', '.join(KINDS)}")
    return reg


def default_names(registry: dict | None = None) -> list[str]:
    """The default --icd set: every enabled coder with default = true."""
    reg = load_registry() if registry is None else registry
    return [n for n, s in reg.items() if s.get("enabled", True) and s.get("default")]


def primary(registry: dict | None = None) -> str | None:
    reg = load_registry() if registry is None else registry
    return next((n for n, s in reg.items() if s.get("primary")), None)


def applies(name: str, suite: str, registry: dict | None = None) -> bool:
    """Whether coder `name` scores `suite` (its `suites` key; an ad-hoc cmd: coder scores every suite)."""
    if name.startswith("cmd:"):
        return True
    spec = (load_registry() if registry is None else registry)[name]
    return "suites" not in spec or suite in spec["suites"]


def make(name: str, suite_records: list[dict], registry: dict | None = None) -> Coder:
    """The coder `name` of the registry, or an ad-hoc `cmd:<template>`. Raises CoderError when it cannot work."""
    if name.startswith("cmd:"):
        return Command(name[4:])
    spec = (load_registry() if registry is None else registry)[name]
    kind = spec["kind"]
    if kind == "knn":
        return Knn(k=spec.get("k", 7), name=name)
    if kind == "lexical":
        return Lexical(label_space(suite_records, aci()[1]), name=name)
    if kind == "agent":
        from eval.coders.agent import AgentCoder

        return AgentCoder(name, spec["model"], spec.get("inputs", "note"))
    if kind == "command":
        return Command(spec["cmd"], name=name, timeout=spec.get("timeout", 3600), ranked=spec.get("ranked", True), inputs=spec.get("inputs", "both"))
    from eval.coders.ollama import DEFAULT_MODEL, LlmCoder

    keys = ("inputs", "context", "max_output", "seed", "device", "timeout", "retries")
    return LlmCoder(name, model=spec.get("model", DEFAULT_MODEL), **{k: spec[k] for k in keys if k in spec})


def _cache_path(coder: Coder, rec: dict) -> Path:
    key = sha1_text(coder.config, rec["encounter_id"], *(rec.get(f) or "" for f in coder.fields()))
    return CACHE_DIR / "icd" / coder.name / f"{key}.json"


def _load(path: Path) -> Prediction | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, list):  # scripts/eval cache file: the ranked code list
        return Prediction(data)
    if not isinstance(data, dict) or not isinstance(data.get("codes"), list):
        return None
    return Prediction(data["codes"], None, data.get("attempts", 1), data.get("seconds", 0.0), data.get("raw_invalid", 0), data.get("usage") or {},
                      data.get("backend"))  # fmt: skip


def _save(path: Path, p: Prediction) -> None:
    if p.error is not None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    row = {"codes": p.codes, "attempts": p.attempts, "seconds": p.seconds, "usage": p.usage, "raw_invalid": p.raw_invalid}
    if p.backend:
        row["backend"] = p.backend
    tmp.write_text(json.dumps(row), encoding="utf-8")
    tmp.replace(path)  # atomic: a concurrent scorer never reads half a file


def cached_predict(coder: Coder, recs: list[dict]) -> dict[str, Prediction]:
    """Predictions keyed by encounter id; cached per coder config and the text it reads. Errors are not cached."""
    out, todo = {}, []
    for r in recs:
        path = _cache_path(coder, r)
        if (p := _load(path)) is not None:
            out[r["encounter_id"]] = p
        else:
            todo.append(r)
    if todo:
        new = coder.predict(todo, on_result=lambda r, p: _save(_cache_path(coder, r), p))
        for r in todo:
            p = new.get(r["encounter_id"]) or Prediction([], "missing_output")
            path = _cache_path(coder, r)
            if not path.exists():
                _save(path, p)
            out[r["encounter_id"]] = p
    return out


def gold_codes(rec: dict) -> list[str]:
    """Canonical gold codes; an invalid one is kept upper-cased, so it still counts as a miss."""
    out: dict[str, None] = {}
    for c in rec.get("icd10_codes") or []:
        code = normalize(c)
        out.setdefault(code or str(c.get("code") if isinstance(c, dict) else c).strip().upper())
    return list(out)


def _level(rows: list[tuple[int, int, int]], ci: list[float]) -> dict:
    tp, fp, fn = (sum(r[i] for r in rows) for i in range(3))
    micro = rates(tp, fp, fn)
    mean_f1 = sum(rates(*r)["f1"] for r in rows) / len(rows)
    return {**{k: round(v, 4) for k, v in micro.items()}, "mean_f1": round(mean_f1, 4), "ci95": [round(x, 4) for x in ci]}


def score(coder: Coder, original: list[dict], anonymized: list[dict], resamples: int = 1000, seed: int = 0) -> dict | None:
    """The coder's result on the records that have gold codes (None when there are none). See the module doc."""
    by_id = {r["encounter_id"]: r for r in anonymized}
    recs = [r for r in original if r.get("icd10_codes") and r["encounter_id"] in by_id]
    if not recs:
        return None if coder.ranked else agreement_only(coder, original, anonymized, resamples, seed)
    try:
        preds = {"original": cached_predict(coder, recs), "anonymized": cached_predict(coder, [by_id[r["encounter_id"]] for r in recs])}
    except CoderError as e:
        return {"error": str(e)}
    rows: dict[str, list] = {f"{a}.{lv}": [] for a in ARMS for lv in LEVELS}
    changed, n_codes = [], 0
    for r in recs:
        gold = gold_codes(r)
        n_codes += len(gold)
        sets = {}
        for arm in ("original", "anonymized"):
            codes = normalize_set(preds[arm][r["encounter_id"]].codes)
            sets[arm] = set(codes[: len(gold)] if coder.ranked else codes)
        o, a, g = sets["original"], sets["anonymized"], set(gold)
        for arm, pred, ref in (("original", o, g), ("anonymized", a, g), ("agreement", a, o)):
            rows[f"{arm}.code"].append(counts(pred, ref))
            rows[f"{arm}.category"].append(counts({category(c) for c in pred}, {category(c) for c in ref}))
        if o != a:
            changed.append({"encounter_id": r["encounter_id"], "gold": sorted(g), "original": sorted(o), "anonymized": sorted(a)})
    ci, delta = paired(rows, [("anonymized.code", "original.code"), ("anonymized.category", "original.category")], resamples, seed)
    res = {"coder": coder.name, "config": coder.config, "ranked": coder.ranked, "inputs": coder.inputs}
    if coder.endpoint:
        res["endpoint"] = coder.endpoint
    res |= {"n_records": len(recs), "n_codes": n_codes}
    res |= {arm: {lv: _level(rows[f"{arm}.{lv}"], ci[f"{arm}.{lv}"]) for lv in LEVELS} for arm in ARMS}
    def f1(arm: str, level: str) -> float:
        return rates(*(sum(x[i] for x in rows[f"{arm}.{level}"]) for i in range(3)))["f1"]

    def change(level: str) -> dict:
        return {"value": round(f1("anonymized", level) - f1("original", level), 4),
                "ci95": [round(x, 4) for x in delta[(f"anonymized.{level}", f"original.{level}")]]}  # fmt: skip

    errors = {arm: sum(p.error is not None for p in preds[arm].values()) for arm in preds}
    res |= {
        "f1_delta": change("code"),
        "f1_delta_category": change("category"),
        "errors": errors,
        "invalid_codes": {arm: sum(p.raw_invalid + count_invalid(p.codes) for p in preds[arm].values()) for arm in preds},
        "valid_for_comparison": not any(errors.values()),
        "changed_predictions": changed,
    }
    return res


def agreement_only(coder: Coder, original: list[dict], anonymized: list[dict], resamples: int = 1000, seed: int = 0) -> dict | None:
    """Without gold codes: the agreement arm alone (anonymized-text predictions scored against original-text
    predictions), over every record, for an unranked coder. None when no record has an output."""
    by_id = {r["encounter_id"]: r for r in anonymized}
    recs = [r for r in original if r["encounter_id"] in by_id]
    if not recs:
        return None
    try:
        preds = {"original": cached_predict(coder, recs), "anonymized": cached_predict(coder, [by_id[r["encounter_id"]] for r in recs])}
    except CoderError as e:
        return {"error": str(e)}
    rows: dict[str, list] = {f"agreement.{lv}": [] for lv in LEVELS}
    changed = []
    for r in recs:
        o, a = (set(normalize_set(preds[arm][r["encounter_id"]].codes)) for arm in ("original", "anonymized"))
        rows["agreement.code"].append(counts(a, o))
        rows["agreement.category"].append(counts({category(c) for c in a}, {category(c) for c in o}))
        if o != a:
            changed.append({"encounter_id": r["encounter_id"], "original": sorted(o), "anonymized": sorted(a)})
    ci, _ = paired(rows, [], resamples, seed)
    res = {"coder": coder.name, "config": coder.config, "ranked": coder.ranked, "inputs": coder.inputs}
    if coder.endpoint:
        res["endpoint"] = coder.endpoint
    errors = {arm: sum(p.error is not None for p in preds[arm].values()) for arm in preds}
    res |= {"n_records": len(recs), "n_codes": 0, "agreement": {lv: _level(rows[f"agreement.{lv}"], ci[f"agreement.{lv}"]) for lv in LEVELS},
            "errors": errors, "invalid_codes": {arm: sum(p.raw_invalid + count_invalid(p.codes) for p in preds[arm].values()) for arm in preds},
            "valid_for_comparison": not any(errors.values()), "changed_predictions": changed}  # fmt: skip
    return res


def evaluate(name: str, suite: str, original: list[dict], anonymized: list[dict], registry: dict | None = None) -> tuple[str, dict] | None:
    """(result key, result) of coder `name` on one run, None when it does not apply or no record has gold codes.
    A coder that cannot even start (Ollama unreachable) gives {"error": message}."""
    reg = load_registry() if registry is None else registry
    if not applies(name, suite, reg):
        return None
    key = Command(name[4:]).name if name.startswith("cmd:") else name
    try:
        coder = make(name, original, reg)
    except CoderError as e:
        return (key, {"error": str(e)}) if any(r.get("icd10_codes") for r in original) else None
    res = score(coder, original, anonymized)
    return None if res is None else (key, res)
