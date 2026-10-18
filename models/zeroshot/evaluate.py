"""Evaluate an anonymized JSONL against the original data, mirroring the hackathon criteria.

Two kinds of metrics:
  data-only   computed from data/ alone (the customer's records and their icd10_codes): ICD-10 F1,
              ICD description term retention, integrity. These need no annotation.
  reference   computed from harness/gold/eval_gold.jsonl, a reference annotation that Claude agents
              produced from data/ at development time. It is NOT organizer ground truth: treat these
              numbers as an approximation and check them with harness/gold/audit_gold.py.

Privacy   (reference; needs gold PII, see harness/gold/build_gold.py)
  - mention-level removal rate : 1 - sum(min(#anon, #orig)) / sum(#orig) over gold PII strings
  - entity-level removal rate  : share of gold PII strings with zero remaining mentions
  - name-token removal rate    : same as mention-level but per name token, so "[PERSON] Collins"
                                 counts "Collins" as leaked even though "Martha Collins" is gone
  broken down by category and by tier (direct identifiers vs quasi-identifiers)

Utility
  - medical concept recall     : (reference) share of gold medical concepts still present (all / ICD-relevant)
  - ICD-10 F1                  : (data-only) a fixed zero-shot LLM coder predicts codes from original and
                                 from anonymized text; micro P/R/F1 vs the icd10_codes in data/ (full code
                                 and 3-character category), plus agreement F1 between the two predictions
  - ICD term retention         : (data-only) mention-level share of the words of each encounter's ICD-10
                                 descriptions ("diabetes", "knee") that survive anonymization
  - integrity                  : (data-only) icd10_codes unchanged, missing records, length ratio

Example
  python3 harness/evaluate.py --anon runs/qwen3.5-2b_extract/anonymized.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVAL_DATA, GOLD_EVAL, JsonlAppender, count_mentions, norm, chat,  # noqa: E402
                    parse_json_loose, read_jsonl, salvage_objects, sha)
from anonymize import icd_terms  # noqa: E402

FIELDS = ("dialogue", "note")
NAME_CATEGORIES = {"PATIENT_NAME", "CLINICIAN_NAME", "OTHER_PERSON_NAME"}
TITLES = {"dr", "mr", "mrs", "ms", "miss", "doctor", "nurse", "md", "pa", "np", "jr", "sr"}
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def safe_div(a, b):
    return a / b if b else None


def prf(tp, fp, fn):
    p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f = 2 * p * r / (p + r) if p and r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------- privacy

def name_tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z][a-z']+", norm(text))
    return [t for t in toks if len(t) >= 3 and t not in TITLES]


def eval_privacy(orig: dict, anon: dict, gold: dict) -> dict:
    """Per-encounter privacy counts. Gold strings that do not occur in the original are skipped."""
    rows, token_rows, skipped = [], [], 0
    seen_tokens = set()
    for e in gold.get("pii", []):
        f = e["field"]
        o = count_mentions(e["text"], orig[f + "_n"])
        if o == 0:
            skipped += 1
            continue
        a = min(count_mentions(e["text"], anon[f + "_n"]), o)
        rows.append({"field": f, "text": e["text"], "category": e["category"], "tier": e["tier"],
                     "orig": o, "left": a})
        if e["category"] in NAME_CATEGORIES:
            for t in name_tokens(e["text"]):
                if (f, t) in seen_tokens:
                    continue
                seen_tokens.add((f, t))
                ot = count_mentions(t, orig[f + "_n"])
                if ot:
                    token_rows.append({"field": f, "token": t, "orig": ot,
                                       "left": min(count_mentions(t, anon[f + "_n"]), ot)})
    return {"rows": rows, "token_rows": token_rows, "skipped": skipped}


def aggregate_privacy(per_enc: list[dict]) -> dict:
    def agg(rows):
        m_o = sum(r["orig"] for r in rows)
        m_l = sum(r["left"] for r in rows)
        return {"n_entities": len(rows), "n_mentions": m_o,
                "mention_removal": safe_div(m_o - m_l, m_o),
                "entity_removal": safe_div(sum(r["left"] == 0 for r in rows), len(rows))}

    rows = [r for p in per_enc for r in p["rows"]]
    toks = [r for p in per_enc for r in p["token_rows"]]
    by_cat, by_tier, by_field = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        by_tier[r["tier"]].append(r)
        by_field[r["field"]].append(r)
    t_o, t_l = sum(r["orig"] for r in toks), sum(r["left"] for r in toks)
    return {
        "overall": agg(rows),
        "name_token_removal": safe_div(t_o - t_l, t_o),
        "by_tier": {k: agg(v) for k, v in sorted(by_tier.items())},
        "by_category": {k: agg(v) for k, v in sorted(by_cat.items())},
        "by_field": {k: agg(v) for k, v in sorted(by_field.items())},
        "gold_strings_not_found_in_original": sum(p["skipped"] for p in per_enc),
    }


# ---------------------------------------------------------------- medical concepts

def eval_concepts(orig: dict, anon: dict, gold: dict) -> dict:
    rows = []
    for c in gold.get("medical_concepts", []):
        f = c["field"]
        o = count_mentions(c["text"], orig[f + "_n"])
        if o == 0:
            continue
        a = min(count_mentions(c["text"], anon[f + "_n"]), o)
        rows.append({"field": f, "text": c["text"], "type": c.get("type"), "icd_relevant": bool(c.get("icd_relevant")),
                     "orig": o, "left": a})
    # Only non-PII ambiguous terms that the annotation also lists as medical concepts count as eponyms
    # ("Cushing's", "Lisfranc"); the rest are product names and small talk ("dax", "dragon", "tennis").
    med = {(c["field"], norm(c["text"])) for c in gold.get("medical_concepts", [])}
    amb = []
    for t in gold.get("ambiguous_terms", []):
        f = t["field"]
        o = count_mentions(t["text"], orig[f + "_n"])
        is_pii = bool(t.get("is_pii"))
        if o and (is_pii or any(norm(t["text"]) in mt or mt in norm(t["text"]) for fm, mt in med if fm == f)):
            amb.append({"text": t["text"], "is_pii": is_pii, "kept": count_mentions(t["text"], anon[f + "_n"]) > 0})
    return {"rows": rows, "ambiguous": amb}


def aggregate_concepts(per_enc: list[dict]) -> dict:
    rows = [r for p in per_enc for r in p["rows"]]

    def agg(rs):
        return {"n": len(rs), "concept_recall": safe_div(sum(r["left"] > 0 for r in rs), len(rs)),
                "mention_recall": safe_div(sum(r["left"] for r in rs), sum(r["orig"] for r in rs))}

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    amb = [a for p in per_enc for a in p["ambiguous"]]
    med_amb = [a for a in amb if not a["is_pii"]]
    pii_amb = [a for a in amb if a["is_pii"]]
    return {
        "overall": agg(rows),
        "icd_relevant": agg([r for r in rows if r["icd_relevant"]]),
        "by_type": {k: agg(v) for k, v in sorted(by_type.items(), key=lambda kv: str(kv[0]))},
        "ambiguous_medical_terms_kept": safe_div(sum(a["kept"] for a in med_amb), len(med_amb)),
        "ambiguous_person_terms_removed": safe_div(sum(not a["kept"] for a in pii_amb), len(pii_amb)),
        "n_ambiguous": {"medical": len(med_amb), "person": len(pii_amb)},
    }


# ---------------------------------------------------------------- ICD description terms (data-only)

def eval_icd_terms(orig: dict, anon: dict) -> list[dict]:
    """Mentions of the words of the encounter's own ICD-10 descriptions, before and after anonymization."""
    rows = []
    for t in sorted(icd_terms(orig)):
        for f in FIELDS:
            o = count_mentions(t, orig[f + "_n"])
            if o:
                rows.append({"field": f, "term": t, "orig": o, "left": min(count_mentions(t, anon[f + "_n"]), o)})
    return rows


def aggregate_icd_terms(rows: list[dict]) -> dict:
    return {"n_terms": len(rows), "n_mentions": sum(r["orig"] for r in rows),
            "mention_retention": safe_div(sum(r["left"] for r in rows), sum(r["orig"] for r in rows)),
            "term_retention": safe_div(sum(r["left"] > 0 for r in rows), len(rows))}


# ---------------------------------------------------------------- ICD-10 coder

CODER_PROMPT_VERSION = "v1"
CODER_SYSTEM = """You are an expert clinical coder.
Read the doctor-patient encounter (dialogue transcript and physician note) and assign the ICD-10-CM diagnosis codes for the conditions that were diagnosed or addressed in this encounter.
Use the most specific valid ICD-10-CM codes supported by the text.
Return JSON of the form {"codes": [{"code": "<ICD-10-CM code>", "description": "<short description>"}]}."""

CODER_SCHEMA = {
    "type": "object",
    "properties": {"codes": {"type": "array", "items": {
        "type": "object", "properties": {"code": {"type": "string"}, "description": {"type": "string"}},
        "required": ["code", "description"]}}},
    "required": ["codes"],
}


def norm_code(c: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(c).upper())


class IcdCoder:
    """Zero-shot ICD-10 coder with a persistent cache keyed on (model, prompt version, text)."""

    def __init__(self, model: str, num_ctx: int, think: bool):
        self.model, self.num_ctx, self.think = model, num_ctx, think
        CACHE_DIR.mkdir(exist_ok=True)
        tag = re.sub(r"[^A-Za-z0-9._-]", "_", model) + ("_think" if think else "")
        self.cache_path = CACHE_DIR / f"icd_{tag}.jsonl"
        # rows that failed to parse before salvaging existed are recomputed (same deterministic call)
        self.cache = {r["key"]: r for r in (read_jsonl(self.cache_path) if self.cache_path.exists() else [])
                      if r.get("parse_ok") or "salvaged" in r}
        self.writer = JsonlAppender(self.cache_path)

    def predict(self, dialogue: str, note: str) -> dict:
        user = f"DIALOGUE:\n{dialogue}\n\nPHYSICIAN NOTE:\n{note}"
        key = sha(self.model, self.think, CODER_PROMPT_VERSION, user)
        if key in self.cache:
            return self.cache[key]
        r = chat(self.model, [{"role": "system", "content": CODER_SYSTEM}, {"role": "user", "content": user}],
                 fmt=CODER_SCHEMA, num_ctx=self.num_ctx, think=self.think, num_predict=1024, timeout=1200)
        parsed = parse_json_loose(r["content"])
        # a reply cut off by the token limit (one code repeated) keeps its finished items
        codes = parsed.get("codes", []) if isinstance(parsed, dict) else salvage_objects(r["content"])
        codes = sorted({norm_code(c.get("code", "")) for c in codes if isinstance(c, dict)} - {""})
        row = {"key": key, "codes": codes, "parse_ok": parsed is not None, "salvaged": parsed is None and bool(codes),
               "seconds": round(r["seconds"], 2)}
        self.cache[key] = row
        self.writer.append(row)
        return row


def set_prf(pred_sets, gold_sets, level: str):
    tp = fp = fn = 0
    for p, g in zip(pred_sets, gold_sets):
        if level == "category":
            p, g = {c[:3] for c in p}, {c[:3] for c in g}
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return prf(tp, fp, fn)


# ---------------------------------------------------------------- main

def fmt(x, pct=True):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def render_report(m: dict) -> str:
    L = [f"# Evaluation: {m['run']}", "",
         f"- anonymized file: `{m['anon_file']}`", f"- records: {m['integrity']['n_evaluated']}"
         f" (missing {m['integrity']['n_missing']})",
         f"- icd10_codes preserved: {fmt(m['integrity']['icd_codes_preserved'])}",
         f"- length ratio anon/orig: dialogue {fmt(m['integrity']['len_ratio']['dialogue'], False)}, "
         f"note {fmt(m['integrity']['len_ratio']['note'], False)}", ""]
    t = m["icd_terms"]
    L += ["## ICD-10 description terms (data-only, higher = better)", "",
          f"- **mention retention: {fmt(t['mention_retention'])}** ({t['n_mentions']} mentions of"
          f" {t['n_terms']} (term, field) pairs from the encounters' own icd10_codes descriptions)",
          f"- term retention (at least one mention left): {fmt(t['term_retention'])}", ""]
    if m.get("privacy") or m.get("concepts"):
        L += [f"> Privacy and concept metrics below use the reference annotation `{m['gold_file']}`,"
              " which was generated by LLM agents from data/, not provided by the organizers."
              " Treat them as approximate (see harness/gold/audit_gold.py).", ""]
    if m.get("privacy"):
        p = m["privacy"]
        L += ["## Privacy (reference annotation, higher = better)", "",
              f"- **mention-level removal: {fmt(p['overall']['mention_removal'])}**"
              f" ({p['overall']['n_mentions']} mentions of {p['overall']['n_entities']} gold PII strings)",
              f"- entity-level removal: {fmt(p['overall']['entity_removal'])}",
              f"- name-token removal: {fmt(p['name_token_removal'])}", "",
              "| group | strings | mentions | mention removal | entity removal |", "|---|---:|---:|---:|---:|"]
        for grp in ("by_tier", "by_category", "by_field"):
            for k, v in p[grp].items():
                L.append(f"| {grp[3:]}: {k} | {v['n_entities']} | {v['n_mentions']} | "
                         f"{fmt(v['mention_removal'])} | {fmt(v['entity_removal'])} |")
        L.append("")
    if m.get("concepts"):
        c = m["concepts"]
        L += ["## Medical information (reference annotation, higher = better)", "",
              f"- **medical concept recall: {fmt(c['overall']['concept_recall'])}** (n={c['overall']['n']})",
              f"- ICD-relevant concept recall: {fmt(c['icd_relevant']['concept_recall'])}"
              f" (n={c['icd_relevant']['n']})",
              f"- ambiguous eponyms kept: {fmt(c['ambiguous_medical_terms_kept'])}"
              f" (n={c['n_ambiguous']['medical']}); ambiguous person terms removed:"
              f" {fmt(c['ambiguous_person_terms_removed'])} (n={c['n_ambiguous']['person']})", "",
              "| concept type | n | concept recall |", "|---|---:|---:|"]
        for k, v in c["by_type"].items():
            L.append(f"| {k} | {v['n']} | {fmt(v['concept_recall'])} |")
        L.append("")
    if m.get("icd"):
        i = m["icd"]
        L += [f"## ICD-10 prediction (data-only; coder: `{i['coder_model']}`, zero-shot)", "",
              "| input | level | precision | recall | F1 |", "|---|---|---:|---:|---:|"]
        for src in ("original", "anonymized"):
            for lvl in ("code", "category"):
                s = i[src][lvl]
                L.append(f"| {src} | {lvl} | {fmt(s['precision'])} | {fmt(s['recall'])} | {fmt(s['f1'])} |")
        L += ["", f"- F1 retained (anon/orig, full code): {fmt(i['f1_retention_code'])}",
              f"- agreement F1 between predictions on anonymized vs original: {fmt(i['agreement_f1_code'])}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anon", required=True, help="anonymized JSONL")
    ap.add_argument("--orig", default=str(EVAL_DATA))
    ap.add_argument("--gold", default=str(GOLD_EVAL))
    ap.add_argument("--out", default=None, help="output dir (default: directory of --anon)")
    ap.add_argument("--coder-model", default="qwen3.5:2b",
                    help="model used as the fixed ICD-10 coder: Ollama tag, or the vLLM served name with "
                         "LLM_BACKEND=vllm (e.g. qwen3.5-2b)")
    ap.add_argument("--coder-think", action="store_true", help="let the coder use thinking mode")
    ap.add_argument("--coder-ctx", type=int, default=16384)
    ap.add_argument("--coder-workers", type=int, default=2)
    ap.add_argument("--skip-icd", action="store_true")
    args = ap.parse_args()

    anon_path = Path(args.anon)
    out_dir = Path(args.out) if args.out else anon_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = {r["encounter_id"]: r for r in read_jsonl(args.orig)}
    anon = {r["encounter_id"]: r for r in read_jsonl(anon_path)}
    ids = [i for i in orig if i in anon]
    gold = {r["encounter_id"]: r for r in read_jsonl(args.gold)} if Path(args.gold).exists() else {}
    if not gold:
        print(f"[evaluate] WARNING: no gold file at {args.gold}; privacy and concept metrics skipped")

    def prep(r):
        return {**r, **{f + "_n": norm(r.get(f) or "") for f in FIELDS}}

    integ = {
        "n_evaluated": len(ids), "n_missing": len(orig) - len(ids),
        "icd_codes_preserved": safe_div(sum(anon[i]["icd10_codes"] == orig[i]["icd10_codes"] for i in ids), len(ids)),
        "len_ratio": {f: safe_div(sum(len(anon[i].get(f) or "") for i in ids),
                                  sum(len(orig[i].get(f) or "") for i in ids)) for f in FIELDS},
    }
    metrics = {"run": out_dir.name, "anon_file": str(anon_path), "integrity": integ}
    details = {i: {"encounter_id": i} for i in ids}

    term_rows = []
    for i in ids:
        rows = eval_icd_terms(prep(orig[i]), prep(anon[i]))
        term_rows += rows
        details[i]["lost_icd_terms"] = [{k: r[k] for k in ("field", "term")} for r in rows if not r["left"]]
    metrics["icd_terms"] = aggregate_icd_terms(term_rows)

    if gold:
        gids = [i for i in ids if i in gold]
        priv, conc = [], []
        for i in gids:
            o, a = prep(orig[i]), prep(anon[i])
            p, c = eval_privacy(o, a, gold[i]), eval_concepts(o, a, gold[i])
            priv.append(p)
            conc.append(c)
            details[i]["leaked_pii"] = [{k: r[k] for k in ("field", "text", "category", "left")}
                                        for r in p["rows"] if r["left"]]
            details[i]["lost_concepts"] = [{k: r[k] for k in ("field", "text", "type")}
                                           for r in c["rows"] if not r["left"]]
        metrics["privacy"] = aggregate_privacy(priv)
        metrics["concepts"] = aggregate_concepts(conc)
        metrics["privacy"]["n_encounters_with_gold"] = len(gids)
        metrics["gold_file"] = args.gold

    if not args.skip_icd:
        coder = IcdCoder(args.coder_model, args.coder_ctx, args.coder_think)
        t0 = time.time()

        def run(i):
            return (coder.predict(orig[i]["dialogue"], orig[i]["note"]),
                    coder.predict(anon[i].get("dialogue") or "", anon[i].get("note") or ""))

        print(f"[evaluate] ICD coder {args.coder_model} on {len(ids)} encounters x2 (cached calls are free)", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.coder_workers)) as ex:
            preds = list(ex.map(run, ids))
        gold_sets = [{norm_code(c["code"]) for c in orig[i]["icd10_codes"]} for i in ids]
        po, pa = [set(p[0]["codes"]) for p in preds], [set(p[1]["codes"]) for p in preds]
        icd = {"coder_model": args.coder_model, "coder_think": args.coder_think,
               "original": {lvl: set_prf(po, gold_sets, lvl) for lvl in ("code", "category")},
               "anonymized": {lvl: set_prf(pa, gold_sets, lvl) for lvl in ("code", "category")},
               "agreement_f1_code": set_prf(pa, po, "code")["f1"],
               "parse_failures": sum(not p[k]["parse_ok"] for p in preds for k in (0, 1)),
               "salvaged": sum(bool(p[k].get("salvaged")) for p in preds for k in (0, 1)),
               "seconds": round(time.time() - t0, 1)}
        icd["f1_retention_code"] = safe_div(icd["anonymized"]["code"]["f1"], icd["original"]["code"]["f1"])
        metrics["icd"] = icd
        for i, g, o_, a_ in zip(ids, gold_sets, po, pa):
            details[i].update({"gold_codes": sorted(g), "pred_orig": sorted(o_), "pred_anon": sorted(a_)})

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "per_encounter.jsonl", "w") as f:
        for i in ids:
            f.write(json.dumps(details[i], ensure_ascii=False) + "\n")
    report = render_report(metrics)
    (out_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
