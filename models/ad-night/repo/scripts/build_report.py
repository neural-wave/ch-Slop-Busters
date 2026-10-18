"""Assemble outputs/eval/REPORT.md (and the README results section) from the saved result JSONs.

Run after scripts/final_runs.sh:  uv run python scripts/build_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from neural_wave.eval.report import table

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "outputs/eval"


def load(name: str) -> dict | None:
    p = EV / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def rows(names: list[str], rename: dict[str, str] | None = None) -> list[dict]:
    out = []
    for n in names:
        r = load(n)
        if r is None:
            print(f"warning: missing result outputs/eval/{n}.json (row left out of the report)")
        if r:
            r = dict(r)
            r["name"] = (rename or {}).get(n, n)
            out.append(r)
    return out


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def privacy_breakdown(r: dict) -> str:
    rem, cnt = r["privacy"]["removed"], r["privacy"]["counts"]
    keys = [k for k in rem if k.startswith(("risk:", "cat:", "field:", "dataset:"))]
    lines = ["| slice | removed | occurrences |", "|---|---|---|"]
    for k in sorted(keys, key=lambda k: (k.split(":")[0], -cnt[k][1])):
        lines.append(f"| {k} | {pct(rem[k])} | {cnt[k][0]}/{cnt[k][1]} |")
    return "\n".join(lines)


def attack_table(a: dict) -> str:
    lines = ["| attack | original text | anonymized text |", "|---|---|---|"]
    if "A1_ner" in a:
        lines.append(f"| A1 independent NER (deid_roberta + GLiNER + rules) recovers gold PII (high+medium) | {pct(a['A1_ner']['original']['rate'])} | {pct(a['A1_ner']['anonymized']['rate'])} |")
    if "A2_roster" in a:
        lines.append(f"| A2 roster dictionary attack ({a['A2_roster']['roster_size']} names) finds person names | {pct(a['A2_roster']['original']['rate'])} | {pct(a['A2_roster']['anonymized']['rate'])} |")
    if "A4_llm" in a:
        for k, label in (("name", "A4 local LLM guesses the patient's name"), ("age_exact", "A4 local LLM guesses the exact age (+-1 y)"),
                         ("place", "A4 local LLM guesses the place"), ("employer_or_job", "A4 local LLM guesses employer/job")):
            o, an = a["A4_llm"]["original"].get(k), a["A4_llm"]["anonymized"].get(k)
            if o and an:
                lines.append(f"| {label} | {pct(o['rate'])} ({o['recovered']}/{o['of']}) | {pct(an['rate'])} ({an['recovered']}/{an['of']}) |")
    r3 = a.get("A3_rare_words", {})
    if r3:
        lines.append(f"| A3 rare identifier-like words (corpus-unique names/places/employers) still present | {r3['rare_tokens_original']} | {r3['rare_tokens_surviving']} |")
    r5 = a.get("A5_linkage", {})
    if r5:
        lines.append(f"| A5 TF-IDF re-link to the original record (attacker holds originals) | - | {pct(r5['rate'])} |")
    return "\n".join(lines)


def main() -> None:
    sub = load("submission_eval")
    llm = load("submission_eval_icdllm")
    parts = ["# Results report\n", "All numbers are produced by `scripts/final_runs.sh` on the current code; JSON details are in `outputs/eval/`.\n"]
    if sub:
        s = sub["summary"]
        parts.append("## Submission: `outputs/anonymized_eval_patient_doctor_encounters.jsonl`\n")
        parts.append("Balanced preset, rules + `obi/deid_roberta_i2b2` + GLiNER-PII, `icd10_codes` used for the shield.\n")
        parts.append(table([sub]) + "\n")
        parts.append(f"- Privacy (share of the 738 gold PII occurrences removed): **{pct(sub['privacy_removed'])}**; high-risk direct identifiers **{pct(sub['privacy_high'])}**; high+medium **{pct(sub['privacy_high_medium'])}**.")
        parts.append(f"- Over-removal: keep-trap survival **{pct(sub['keep_survival'])}**, edit precision {pct(sub['edit_precision'])}, medical concept recall **{pct(sub['concept_recall'])}**, code-evidence retention {pct(sub['evidence_retention'])}.")
        parts.append(f"- ICD-10 proxy F1 (bge-m3 retrieval, oracle k): original {sub.get('icd_f1_original')} vs anonymized {sub.get('icd_f1_anonymized')}, agreement {sub.get('icd_agreement_f1')}.")
        if llm:
            parts.append(f"- ICD-10 proxy F1 (local Qwen 3.5 2B coder over retrieved candidates): original {llm.get('icd_f1_original')} vs anonymized {llm.get('icd_f1_anonymized')}, agreement {llm.get('icd_agreement_f1')}.")
        parts.append(f"- Efficiency: {s['ms_per_encounter']} ms/encounter, peak RAM {s['peak_ram_mb']} MB, peak VRAM {s['peak_vram_mb']} MB (budget 4096 MB), placement {s.get('device_placement')}.")
        parts.append(f"- Integrity: schema errors {len(s['schema_errors'])}, shield/edit overlaps {s['shield_edit_overlaps']}, residual cross-field leaks {s['residual_leaks']}, explained decisions {pct(s['explained_decisions'])} of {s['n_decisions']}.\n")
        parts.append("### Privacy by slice\n\n" + privacy_breakdown(sub) + "\n")
    for split in ("eval", "dev"):
        names = [f"sweep_{split}_{p}" for p in ("conservative", "balanced", "aggressive", "balanced_nocodes")]
        names += [n + "_rules+deid+gliner" for n in names]
        rs = rows(names)
        if rs:
            parts.append(f"## Privacy / utility curve on `{split}` (presets x codes, rules only vs with models)\n\n" + table(rs) + "\n")
    for split in ("eval", "dev"):
        names = [f"sweep_{split}_balanced", f"ablation_{split}_rules+deid", f"ablation_{split}_rules+gliner", f"sweep_{split}_balanced_rules+deid+gliner",
                 f"ablation_{split}_rules+llm", f"ablation_{split}_rules+deid+gliner+llm"]
        rs = rows(names)
        if rs:
            parts.append(f"## Detector ablation on `{split}` (balanced)\n\n" + table(rs) + "\n")
    a = load("attack_submission")
    if a:
        parts.append("## Adversarial attacks on the submission (eval split)\n\nEvery attack also runs on the original text as a positive control.\n\n" + attack_table(a) + "\n")
    b = load("attack_balanced_rules")
    if b:
        parts.append("### Same attacks on the rules-only balanced output\n\n" + attack_table(b) + "\n")
    fp = rows(["robust_synthetic_firstpass", "robust_synthetic_firstpass_full"], {
        "robust_synthetic_firstpass": "synthetic FIRST PASS (before hospital-format rules), rules",
        "robust_synthetic_firstpass_full": "synthetic FIRST PASS, rules+deid+gliner"})
    if fp:
        parts.append("## Robustness: untuned first pass on the synthetic hospital documents\n\n"
                     "Measured once, before any rule was written for these formats (and scored with the earlier whole-string metric, "
                     "which is more lenient than the strict token-level scoring used everywhere else). The `synthetic` set was then used to develop "
                     "`formats.py`, so its later numbers are development numbers; the `holdout` set (24 documents in 8 other hospital "
                     "formats, written by an agent that never saw the code) is the untouched test.\n\n" + table(fp) + "\n")
    for det in ("", "_rules+deid+gliner"):
        rs = rows([f"robust_{n}_balanced{nc}{det}" for n in ("holdout2", "holdout", "discharge", "synthetic", "medconv") for nc in ("", "_nocodes") if not (n == "medconv" and nc == "")])
        if rs:
            parts.append(f"## Robustness to unseen formats ({'rules + models' if det else 'rules only'})\n\n"
                         "`holdout` = 24 untouched hospital documents (ICU, operative, psychiatry intake, paediatrics, ambulance handover, "
                         "medication reconciliation, home health, portal chat); `discharge` = 500 Kaggle discharge summaries rendered in 3 layouts; "
                         "`synthetic` = 24 hospital documents used to develop the format rules; `medconv` = 80 web medical conversations with injected PII.\n\n" + table(rs) + "\n")
    report = "\n".join(parts)
    (EV / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"-> {EV / 'REPORT.md'}")
    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    if sub:
        summary = ["### Results (eval split, 67 encounters; full report: `outputs/eval/REPORT.md`)\n", table(rows([
            "sweep_eval_conservative", "sweep_eval_balanced", "sweep_eval_aggressive", "sweep_eval_balanced_nocodes",
            "submission_eval", "ablation_eval_rules+deid+gliner+llm"], {"submission_eval": "SUBMISSION balanced+deid+gliner"})), ""]
        rob = rows(["robust_holdout2_balanced", "robust_holdout2_balanced_rules+deid+gliner", "robust_holdout_balanced_rules+deid+gliner",
                    "robust_discharge_balanced_rules+deid+gliner", "robust_synthetic_balanced_rules+deid+gliner",
                    "robust_medconv_balanced_nocodes_rules+deid+gliner"])
        if rob:
            summary += ["Robustness to unseen formats (`holdout2` = untouched 16-document hospital set, 8 formats nobody tuned on):\n", table(rob), ""]
        if a:
            summary += ["Adversarial attacks on the submission:\n", attack_table(a), ""]
        block = "<!-- RESULTS -->\n" + "\n".join(summary) + "\n<!-- /RESULTS -->"
        if "RESULTS_PLACEHOLDER" in txt:
            txt = txt.replace("RESULTS_PLACEHOLDER", block)
        else:
            txt = re.sub(r"<!-- RESULTS -->.*?<!-- /RESULTS -->", lambda _: block, txt, flags=re.S)
        readme.write_text(txt, encoding="utf-8")
        print("-> README.md results section updated")


if __name__ == "__main__":
    main()
