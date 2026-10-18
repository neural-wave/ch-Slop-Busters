"""Evaluation orchestration and the one-table-per-run report."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import paths
from ..io import read_jsonl, validate_output
from . import privacy, utility

COLUMNS = [
    ("run", "name"),
    ("privacy", "privacy_removed"),
    ("high", "privacy_high"),
    ("high+med", "privacy_high_medium"),
    ("low", "privacy_low"),
    ("direct", "privacy_direct"),
    ("quasi", "privacy_quasi"),
    ("name tok", "name_token_removed"),
    ("keep", "keep_survival"),
    ("prec", "edit_precision"),
    ("concepts", "concept_recall"),
    ("lab concepts", "labeled_concept_recall"),
    ("evidence", "evidence_retention"),
    ("ICD F1 orig", "icd_f1_original"),
    ("ICD F1 anon", "icd_f1_anonymized"),
    ("ICD agree", "icd_agreement_f1"),
    ("ms/enc", "ms_per_encounter"),
    ("LLM", "llm_model"),
    ("LLM calls", "llm_calls"),
    ("LLM VRAM MB", "llm_vram_mb"),
    ("RAM MB", "peak_ram_mb"),
    ("VRAM MB", "peak_vram_mb"),
]


def label_space(*record_lists: list[dict]) -> dict[str, str]:
    space = {}
    for recs in record_lists:
        for r in recs:
            for c in r.get("icd10_codes") or []:
                space[c["code"]] = c["description"]
    for f in (paths.split_file("train"), paths.split_file("eval")):
        for r in read_jsonl(f):
            for c in r["icd10_codes"]:
                space.setdefault(c["code"], c["description"])
    return space


def evaluate(
    original: list[dict],
    anonymized: list[dict],
    gold_dir: Path | None,
    decisions: list[dict] | None,
    summary: dict | None,
    icd: str = "embed",
    name: str = "run",
    llm_model: str | None = None,
    status=None,
) -> dict:
    say = status or (lambda message: None)
    t0 = time.perf_counter()
    res: dict = {"name": name, "n_records": len(original)}
    res["schema_errors"] = validate_output(original, anonymized)
    if gold_dir and Path(gold_dir).exists():
        gold = privacy.load_gold(gold_dir)
        say(f"scoring privacy against the labels in {gold_dir} ({len(gold)} encounters)...")
        p = privacy.score(original, anonymized, gold, decisions)
        res["privacy"] = p
        res["privacy_removed"] = p["removed"].get("ALL")
        res["privacy_high"] = p["removed"].get("risk:high")
        res["privacy_high_medium"] = p["removed"].get("risk:high+medium")
        res["privacy_low"] = p["removed"].get("risk:low")
        res["privacy_direct"] = p["removed"].get("tier:direct")
        res["privacy_quasi"] = p["removed"].get("tier:quasi")
        res["name_token_removed"] = p["name_token_removed"]
        res["labeled_concept_recall"] = p["labeled_concept_recall"].get("ALL")
        res["keep_survival"] = p["keep_survival"].get("ALL")
        res["edit_precision"] = p.get("edit_precision")
    say("measuring medical concept recall and ICD evidence retention...")
    cr = utility.concept_recall(original, anonymized)
    res["concepts"] = cr
    res["concept_recall"] = cr["concept_recall"]
    ev = utility.evidence_retention(original, anonymized)
    res.update(ev)
    if icd != "none":
        try:
            space = label_space(original)
            if icd == "llm":
                from . import icd_llm

                say("ICD proxy: a local LLM selects codes for every text (this takes a while)...")
                ip = icd_llm.icd_llm_proxy(
                    original, anonymized, space, paths.cache_dir() / "emb.sqlite", model=llm_model
                )
            else:
                say(
                    f"ICD proxy: embedding {len(space)} code descriptions and {2 * len(original)} texts with bge-m3 "
                    "in Ollama (cached after the first run; use --icd none to skip)..."
                )
                ip = utility.icd_proxy(original, anonymized, space, paths.cache_dir() / "emb.sqlite")
            res["icd"] = ip
            for k in ("icd_f1_original", "icd_f1_anonymized", "icd_agreement_f1"):
                res[k] = ip.get(k)
        except Exception as e:  # noqa: BLE001  (Ollama not running: utility proxy skipped, reported)
            res["icd_error"] = repr(e)
    if summary:
        res["summary"] = summary
        for k in ("ms_per_encounter", "peak_ram_mb", "peak_vram_mb", "runtime_s"):
            res[k] = summary.get(k)
        llm = (summary.get("detectors") or {}).get("llm") or {}
        res["llm_model"], res["llm_calls"], res["llm_vram_mb"] = llm.get("model"), llm.get("calls"), llm.get("vram_mb")
    res["eval_seconds"] = round(time.perf_counter() - t0, 1)
    return res


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if v <= 1.0 else f"{v:,.1f}"
    return str(v)


def table(rows: list[dict]) -> str:
    head = "| " + " | ".join(c for c, _ in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for _, k in COLUMNS) + " |")
    return "\n".join(lines)


def save(res: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{res['name']}.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log = out_dir / "results.md"
    new = not log.exists()
    with log.open("a", encoding="utf-8") as fh:
        if new:
            fh.write(
                "# Local evaluation log\n\nOne row per run (`unified-harness run` / `evaluate`). "
                "Privacy = share of silver-label PII occurrences removed.\n\n"
            )
            fh.write(table([]).split("\n", 2)[0] + "\n" + table([]).split("\n", 2)[1] + "\n")
        fh.write(table([res]).split("\n")[2] + "\n")
    return p
