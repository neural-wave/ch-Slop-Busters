"""Anonymize records with the trained policy on a vLLM server, and score policy outputs against reference labels.

Deployment path (TRAINING_PLAN.md §0): the exported checkpoint is served by ../term-generalizer/serve_vllm.sh (VRAM_GB set: fp8,
<= 4 GB), this client builds the prompt, prefills `dx` from icd10_codes as an assistant prefix (codes never enter the
user turn), and renders the JSON decisions with rl/render.py. Standard library only on this path; the server applies
the chat template (continue_final_message for the prefix).

- One pass up to --max-prompt tokens (measured with the server's /tokenize). Longer records are cut on line
  boundaries into windows of --window-tokens content tokens with --overlap tokens of overlap; each window carries the
  identifiers found so far (<= 64), and every string found is applied to the whole record.
- A reply cut off by the token limit keeps its finished items (salvage), then the input is split in half and re-run.

  python3 rl/infer.py --model nw-policy --out runs/pilot_ppo --split test
  python3 rl/infer.py --model nw-policy --out runs/pilot_ppo_noicd --split test --no-icd
  python3 rl/infer.py --check-template --model nw-policy     server prompt == training prompt (needs transformers)

Writes <out>/anonymized.jsonl (same schema, icd10_codes unchanged), anonymization_log.jsonl (one explanation per
decision) and anonymize_meta.json; then run ../term-generalizer/evaluate.py --anon <out>/anonymized.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
from common import (EVAL_DATA, GOLD_EVAL, VLLM_URL, LLMError, norm, parse_json_loose, read_jsonl,  # noqa: E402
                    salvage_objects, write_jsonl)

import render as R  # noqa: E402
from spec import SYSTEM_PROMPT, dx_prefix, user_message  # noqa: E402

SECTION_TITLE = {"dialogue": "DIALOGUE", "note": "PHYSICIAN NOTE"}
SPLITS = Path(__file__).resolve().parent / "splits.json"


def load_eval() -> list[dict]:
    return read_jsonl(EVAL_DATA)


def messages(sections: list[tuple[str, str]], known: list[dict] | None = None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message(sections, known)}]


# ---------------------------------------------------------------- output parsing

def parse_output(text: str) -> tuple[dict, bool]:
    """(decisions, parse_ok). An unparsable or cut-off reply keeps its finished flat items."""
    obj = parse_json_loose(text or "")
    if isinstance(obj, dict):
        return obj, True
    out = {"dx": [], "evidence": [], "entities": [], "keep": []}
    for o in salvage_objects(text):
        if "type" in o and "text" in o:
            out["entities"].append(o)
        elif "text" in o and "reason" in o:
            out["keep"].append(o)
        elif "code" in o and "desc" in o:
            out["dx"].append(o)
    return out, False


def merge_outputs(outs: list[dict]) -> dict:
    m = {"dx": [], "evidence": [], "entities": [], "keep": []}
    for o in outs:
        for k in m:
            m[k] += [x for x in (o.get(k) or []) if isinstance(x, dict)]
    return m


# ---------------------------------------------------------------- server client

def _post(path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(f"{VLLM_URL}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def server_tokens(model: str, msgs: list[dict], prefix: str) -> list[int]:
    body = {"model": model, "messages": msgs + ([{"role": "assistant", "content": prefix}] if prefix else []),
            "add_generation_prompt": not prefix, "continue_final_message": bool(prefix),
            "chat_template_kwargs": {"enable_thinking": False}}
    return _post("/tokenize", body)["tokens"]


def complete(model: str, msgs: list[dict], prefix: str, max_tokens: int, retries: int = 3) -> dict:
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "presence_penalty": 0.0, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": msgs + ([{"role": "assistant", "content": prefix}] if prefix else [])}
    if prefix:
        body.update(add_generation_prompt=False, continue_final_message=True)
    err = None
    for a in range(retries):
        try:
            t0 = time.time()
            out = _post("/v1/chat/completions", body)
            c = out["choices"][0]
            return {"text": prefix + (c["message"].get("content") or ""), "finish": c.get("finish_reason"),
                    "prompt_tokens": out["usage"]["prompt_tokens"], "gen_tokens": out["usage"]["completion_tokens"],
                    "seconds": time.time() - t0}
        except Exception as e:   # noqa: BLE001 - retried, then raised
            err = e
            time.sleep(2 * (a + 1))
    raise LLMError(f"policy call failed: {err}")


# ---------------------------------------------------------------- one record

def sections_of(rec: dict) -> list[tuple[str, str]]:
    return [(SECTION_TITLE[f], rec.get(f) or "") for f in ("dialogue", "note") if (rec.get(f) or "").strip()]


def _windows(rec: dict, budget_chars: int, overlap_chars: int) -> list[list[tuple[str, str]]]:
    """Line-boundary windows over the sections, each <= budget_chars, with overlap."""
    lines = [(t, l) for t, txt in sections_of(rec) for l in txt.split("\n")]
    wins, i = [], 0
    while i < len(lines):
        j, size = i, 0
        while j < len(lines) and (size + len(lines[j][1]) + 1 <= budget_chars or j == i):
            size += len(lines[j][1]) + 1
            j += 1
        secs = []
        for t, l in lines[i:j]:
            if secs and secs[-1][0] == t:
                secs[-1] = (t, secs[-1][1] + "\n" + l)
            else:
                secs.append((t, l))
        wins.append(secs)
        if j >= len(lines):
            break
        back, k = 0, j
        while k > i + 1 and back + len(lines[k - 1][1]) + 1 <= overlap_chars:
            k -= 1
            back += len(lines[k][1]) + 1
        i = k
    return wins


def run_sections(model: str, secs, prefix: str, known: list, args, log: dict, depth: int = 0) -> dict:
    r = complete(model, messages(secs, known), prefix, args.max_resp)
    log["calls"].append({k: r[k] for k in ("finish", "prompt_tokens", "gen_tokens", "seconds")})
    out, ok = parse_output(r["text"])
    if r["finish"] == "length" and depth < 2:   # cut off: keep the finished items, then re-run both halves
        lines = [(t, l) for t, txt in secs for l in txt.split("\n")]
        if len(lines) > 1:
            half = len(lines) // 2
            parts = []
            for chunk in (lines[:half], lines[half:]):
                s2 = []
                for t, l in chunk:
                    s2 = s2[:-1] + [(t, s2[-1][1] + "\n" + l)] if s2 and s2[-1][0] == t else s2 + [(t, l)]
                parts.append(run_sections(model, s2, prefix, known, args, log, depth + 1))
            return merge_outputs([out] + parts)
    log["parse_ok"] = log.get("parse_ok", True) and ok
    return out


def anonymize(rec: dict, args) -> tuple[dict, dict]:
    t0 = time.time()
    prefix = dx_prefix(rec["icd10_codes"]) if args.icd and rec.get("icd10_codes") else ""
    log = {"encounter_id": rec["encounter_id"], "calls": []}
    secs = sections_of(rec)
    n_tok = len(server_tokens(args.model, messages(secs), prefix))
    if n_tok <= args.max_prompt:
        out = run_sections(args.model, secs, prefix, [], args, log)
    else:   # windows: chars per token measured on this record
        ratio = sum(len(t) for _, t in secs) / max(1, n_tok)
        wins = _windows(rec, int(args.window_tokens * ratio), int(args.overlap * ratio))
        outs, known = [], []
        for w in wins:
            o = run_sections(args.model, w, prefix, known[:64], args, log)
            outs.append(o)
            for e in o.get("entities") or []:
                if isinstance(e, dict) and e.get("text") and not any(k["text"].lower() == str(e["text"]).lower()
                                                                     for k in known):
                    known.append({"text": str(e["text"]), "type": str(e.get("type", ""))})
        out = merge_outputs(outs)
        log["windows"] = len(wins)
    descs = ([c.get("description", "") for c in rec["icd10_codes"]] if prefix else
             [d.get("desc", "") for d in out.get("dx") or [] if isinstance(d, dict)])
    anon, expl = R.render_record(rec, out, protect_terms=R.terms_of(descs))
    log.update(prompt_tokens=n_tok, seconds=round(time.time() - t0, 2), decisions=expl, raw=out)
    return anon, log


# ---------------------------------------------------------------- scoring (dev validation in rl/ppo.py)

def score_outputs(recs: list[dict], outputs: list[str], gold_path=GOLD_EVAL, prefilled: bool = True) -> dict:
    """Privacy / concept metrics of raw policy outputs (prefix + reply) against reference labels."""
    import evaluate as E
    gold = {g["encounter_id"]: g for g in read_jsonl(gold_path)}
    priv, conc, ok, leaked_direct = [], [], 0, 0
    for rec, text in zip(recs, outputs):
        out, parse_ok = parse_output(text)
        ok += parse_ok
        descs = [c.get("description", "") for c in rec["icd10_codes"]] if prefilled else \
            [d.get("desc", "") for d in out.get("dx") or [] if isinstance(d, dict)]
        anon, _ = R.render_record(rec, out, protect_terms=R.terms_of(descs))
        g = gold.get(rec["encounter_id"])
        if g is None:
            continue
        o = {**rec, **{f + "_n": norm(rec.get(f) or "") for f in E.FIELDS}}
        a = {**anon, **{f + "_n": norm(anon.get(f) or "") for f in E.FIELDS}}
        p = E.eval_privacy(o, a, g)
        priv.append(p)
        conc.append(E.eval_concepts(o, a, g))
        leaked_direct += any(r["left"] and r["tier"] == "direct" for r in p["rows"])
    P, C = E.aggregate_privacy(priv), E.aggregate_concepts(conc)
    m = {"n": len(recs), "json_valid": ok / max(1, len(recs)),
         "pii_removed": P["overall"]["mention_removal"], "name_tokens": P["name_token_removal"],
         "direct": P["by_tier"].get("direct", {}).get("mention_removal"),
         "quasi": P["by_tier"].get("quasi", {}).get("mention_removal"),
         "concept_recall": C["overall"]["concept_recall"], "icd_concept_recall": C["icd_relevant"]["concept_recall"],
         "enc_with_direct_leak": leaked_direct / max(1, len(priv))}
    m.update({f"cat_{k}": v["mention_removal"] for k, v in P["by_category"].items()})
    return m


def check_template(model: str) -> None:
    """The server's prompt tokens equal the training prompt (rl/data.py) for a prefilled and a free record."""
    import data as D
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(D.MODEL_DIR))
    rec = load_eval()[0]
    for prefill in (True, False):
        s = D.Sample(rec["encounter_id"], rec, {}, prefill, ["dialogue", "note"])
        local = tok(s.prompt_text(tok), add_special_tokens=False)["input_ids"]
        remote = server_tokens(model, s.messages(), s.prefix())
        status = "OK" if local == remote else f"MISMATCH (local {len(local)} vs server {len(remote)} tokens)"
        print(f"prefill={prefill}: {status}")
        if local != remote:
            k = next((i for i, (a, b) in enumerate(zip(local, remote)) if a != b), min(len(local), len(remote)))
            print("  local :", repr(tok.decode(local[max(0, k - 20):k + 20])))
            print("  server:", repr(tok.decode(remote[max(0, k - 20):k + 20])))
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nw-policy", help="served model name")
    ap.add_argument("--input", default=str(EVAL_DATA))
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all", help="eval subset (rl/splits.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-icd", dest="icd", action="store_false", help="no dx prefill (bonus: codes unknown)")
    ap.add_argument("--max-prompt", type=int, default=16384)
    ap.add_argument("--max-resp", type=int, default=2560)
    ap.add_argument("--window-tokens", type=int, default=15000)
    ap.add_argument("--overlap", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-template", action="store_true")
    args = ap.parse_args()
    if args.check_template:
        check_template(args.model)
        return
    if not args.out:
        ap.error("--out is required")
    recs = read_jsonl(args.input)
    if args.split != "all":
        keep = set(json.loads(SPLITS.read_text())[args.split])
        recs = [r for r in recs if r["encounter_id"] in keep]
    if args.limit:
        recs = recs[:args.limit]
    out_dir = Path(args.out)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda r: anonymize(r, args), recs))
    write_jsonl(out_dir / "anonymized.jsonl", [a for a, _ in res])
    write_jsonl(out_dir / "anonymization_log.jsonl", [l for _, l in res])
    secs = [l["seconds"] for _, l in res]
    meta = {"mode": "policy", "model": args.model, "backend": "vllm", "input": args.input, "split": args.split,
            "icd_prefill": args.icd, "n_records": len(res), "max_prompt": args.max_prompt, "max_resp": args.max_resp,
            "wall_seconds": round(time.time() - t0, 1), "mean_seconds_per_record": round(sum(secs) / len(secs), 2),
            "parse_ok": sum(bool(l.get("parse_ok", True)) for _, l in res) / len(res),
            "truncated_calls": sum(c["finish"] == "length" for _, l in res for c in l["calls"])}
    (out_dir / "anonymize_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
