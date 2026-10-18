"""Evaluation orchestration and the one-table-per-run report."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..io import read_jsonl, validate_output
from . import privacy, utility

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "resources/data"
SPLIT_FILES = {
    "eval": DATA / "eval_patient_doctor_encounters.jsonl",
    "dev": DATA / "patient_doctor_encounters.jsonl",
    "train": DATA / "patient_doctor_encounters.jsonl",
}

COLUMNS = [
    ("run", "name"),
    ("privacy", "privacy_removed"),
    ("high", "privacy_high"),
    ("high+med", "privacy_high_medium"),
    ("keep", "keep_survival"),
    ("prec", "edit_precision"),
    ("concepts", "concept_recall"),
    ("evidence", "evidence_retention"),
    ("ICD F1 orig", "icd_f1_original"),
    ("ICD F1 anon", "icd_f1_anonymized"),
    ("ICD agree", "icd_agreement_f1"),
    ("ms/enc", "ms_per_encounter"),
    ("RAM MB", "peak_ram_mb"),
    ("VRAM MB", "peak_vram_mb"),
]


def label_space(*record_lists: list[dict]) -> dict[str, str]:
    space = {}
    for recs in record_lists:
        for r in recs:
            for c in r.get("icd10_codes") or []:
                space[c["code"]] = c["description"]
    for f in SPLIT_FILES.values():
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
) -> dict:
    t0 = time.perf_counter()
    res: dict = {"name": name, "n_records": len(original)}
    res["schema_errors"] = validate_output(original, anonymized)
    if gold_dir and Path(gold_dir).exists():
        gold = privacy.load_gold(gold_dir)
        p = privacy.score(original, anonymized, gold, decisions)
        res["privacy"] = p
        res["privacy_removed"] = p["removed"].get("ALL")
        res["privacy_high"] = p["removed"].get("risk:high")
        res["privacy_high_medium"] = p["removed"].get("risk:high+medium")
        res["keep_survival"] = p["keep_survival"].get("ALL")
        res["edit_precision"] = p.get("edit_precision")
    cr = utility.concept_recall(original, anonymized, gold=privacy.load_gold(gold_dir) if gold_dir and Path(gold_dir).exists() else None)
    res["concepts"] = cr
    res["concept_recall"] = cr["concept_recall"]
    ev = utility.evidence_retention(original, anonymized)
    res.update(ev)
    if icd != "none":
        try:
            space = label_space(original)
            if icd == "llm":
                from . import icd_llm

                ip = icd_llm.icd_llm_proxy(original, anonymized, space, ROOT / "outputs/cache/emb.sqlite", model=llm_model)
            else:
                ip = utility.icd_proxy(original, anonymized, space, ROOT / "outputs/cache/emb.sqlite")
            res["icd"] = ip
            for k in ("icd_f1_original", "icd_f1_anonymized", "icd_agreement_f1"):
                res[k] = ip.get(k)
        except Exception as e:  # noqa: BLE001  (Ollama not running: utility proxy skipped, reported)
            res["icd_error"] = repr(e)
    if summary:
        res["summary"] = summary
        for k in ("ms_per_encounter", "peak_ram_mb", "peak_vram_mb", "runtime_s"):
            res[k] = summary.get(k)
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
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log = out_dir / "results.md"
    new = not log.exists()
    with log.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("# Local evaluation log\n\nOne row per run (`neural-wave run` / `evaluate`). Privacy = share of gold PII occurrences removed.\n\n")
            fh.write(table([]).split("\n", 2)[0] + "\n" + table([]).split("\n", 2)[1] + "\n")
        fh.write(table([res]).split("\n")[2] + "\n")
    return p
