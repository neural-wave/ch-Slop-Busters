"""Zero-shot LLM anonymization baseline (Ollama or vLLM via LLM_BACKEND, on-prem).

Modes
  extract   LLM returns the PII spans of each chunk as JSON (structured output); the spans are
            replaced deterministically with [TYPE] placeholders. Text outside spans is untouched.
  generalize  like extract (same prompt and spans), plus a deterministic regex pass for ages and dates
            (generalize.py); AGE and DATE become vague ranges instead of tags ("44-year-old" ->
            "forty-something", "4/24/2021" -> "early 2020s", "july 8th" -> "summer"). Other types keep [TYPE].
            Spans made only of words from the encounter's own icd10_codes descriptions ("diabetes" tagged
            ORGANIZATION, "knee" tagged LOCATION) are kept as diagnosis information (--no-icd-hints turns
            this off, for input without codes). Every decision is logged with an explanation.
  rewrite   LLM rewrites each chunk with placeholders inlined. Closest to "just ask the LLM",
            but the model may also drop, paraphrase or hallucinate text.
  identity  No anonymization (control run: upper bound for utility, zero privacy).

Output: <out_dir>/anonymized.jsonl in the same schema as the input (icd10_codes copied as-is),
plus <out_dir>/anonymization_log.jsonl with per-chunk model output, tokens and latency
(generalize mode also logs every removed, generalized or retained span with its reason).

--reuse-entities LOG replays the per-chunk entities of an earlier extract/generalize run (same model and
--chunk-chars) instead of calling the model; a chunk missing from LOG is an error, never a model call.

Example
  python3 harness/anonymize.py --mode extract --model qwen3.5:2b --out runs/qwen3.5-2b_extract
  python3 harness/anonymize.py --mode generalize --model qwen3.5:2b --out runs/qwen3.5-2b_generalize
      --reuse-entities runs/qwen3.5-2b_extract/anonymization_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVAL_DATA, LLM_BACKEND, ROOT, JsonlAppender, chunk_lines, chat,  # noqa: E402
                    parse_json_loose, read_jsonl, salvage_objects, write_jsonl)
import generalize  # noqa: E402

PII_TYPES = ["PERSON", "DATE", "AGE", "LOCATION", "ORGANIZATION", "OCCUPATION", "CONTACT", "ID"]
# Dialogue turn markers ([doctor], [patient], ...) and our own placeholders are format, not PII.
SPEAKER_TAG = re.compile(r"\[\s*[a-z_ ]+\s*\]", re.IGNORECASE)

# Zero-shot prompts: task definition + label set, no demonstrations.
EXTRACT_SYSTEM = """You are a de-identification system for clinical text.
Find every piece of personally identifying information (PII) in the text the user gives you.
PII types:
- PERSON: names of any person (patient, doctor, nurse, family member, friend, other)
- DATE: specific dates, dates of birth, specific years
- AGE: a person's age
- LOCATION: cities, states, countries, addresses, landmarks, named places
- ORGANIZATION: employers, companies, schools, hospitals, clinics
- OCCUPATION: a person's job or profession
- CONTACT: phone numbers, email addresses, URLs
- ID: medical record numbers, social security numbers, insurance or other ID numbers
Do NOT mark medical information (diseases, symptoms, medications, procedures, test results, body parts), even if a disease is named after a person (e.g. Parkinson's disease).
Return JSON of the form {"entities": [{"text": "<exact text as it appears>", "type": "<PII type>"}]}. Return {"entities": []} if there is no PII."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": PII_TYPES}},
                "required": ["text", "type"],
            },
        }
    },
    "required": ["entities"],
}

REWRITE_SYSTEM = """You are a de-identification system for clinical text.
Rewrite the text the user gives you so that all personally identifying information is replaced by a placeholder:
[PERSON] for names of any person, [DATE] for specific dates, [AGE] for ages, [LOCATION] for places and addresses,
[ORGANIZATION] for employers, schools, hospitals and companies, [OCCUPATION] for jobs, [CONTACT] for phone numbers,
emails and URLs, [ID] for ID numbers.
Keep everything else exactly as it is, including all medical information, wording, punctuation and line breaks.
Output only the rewritten text, nothing else."""


def user_msg(chunk: str) -> str:
    return f"Text:\n<<<\n{chunk}\n>>>"


# ---------------------------------------------------------------- extract mode

def default_render(text: str, typ: str) -> str:
    return f"[{typ}]"


def apply_entities(chunk: str, entities: list[dict], render=default_render,
                   protect: frozenset | None = None) -> tuple[str, list[dict]]:
    """Replace every whole-word, case-insensitive occurrence of each entity with render(occurrence, TYPE).

    One combined regex (longest first) so we never rewrite inside an earlier replacement.
    Entities not present in the chunk (hallucinated or paraphrased) are dropped; for a text listed
    twice the first entry wins. The default render gives the [TYPE] placeholder. Entities for which
    is_icd_term(text, protect) holds are dropped too (diagnosis words, see icd_terms).
    """
    kept, seen = [], set()
    for e in entities:
        if not isinstance(e, dict):
            continue
        t = str(e.get("text", "")).strip()
        typ = str(e.get("type", "")).upper()
        if typ not in PII_TYPES:
            typ = "PERSON" if typ in ("NAME", "PATIENT", "DOCTOR") else "ID" if not typ else typ
            if typ not in PII_TYPES:
                continue
        if len(t) < 2 or t.lower() in seen or SPEAKER_TAG.fullmatch(t):
            continue
        if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", chunk, flags=re.IGNORECASE) is None:
            continue
        if is_protected(t, typ, protect):
            continue
        seen.add(t.lower())
        kept.append({"text": t, "type": typ})
    if not kept:
        return chunk, kept
    kept.sort(key=lambda e: len(e["text"]), reverse=True)
    type_of = {e["text"].lower(): e["type"] for e in kept}
    pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(e["text"]) for e in kept) + r")(?!\w)", re.IGNORECASE)
    return pat.sub(lambda m: render(m.group(1), type_of[m.group(1).lower()]), chunk), kept


# Generic words of ICD-10 descriptions that say nothing about the diagnosis.
ICD_STOPWORDS = {"and", "or", "of", "the", "with", "without", "due", "to", "in", "on", "for", "by", "not", "other",
                 "unspecified", "specified", "type", "site", "part", "encounter", "initial", "subsequent",
                 "sequela", "elsewhere", "classified", "person", "history"}


def _words(text: str) -> list[str]:
    return [re.sub(r"'s?$", "", w) for w in re.findall(r"[a-z][a-z']*", text.lower().replace("\u2019", "'"))]


def icd_terms(rec: dict) -> frozenset:
    """Content words of the encounter's own icd10_codes descriptions ("Type 2 diabetes mellitus without
    complications" -> diabetes, mellitus, complications). The task makes every field available at inference."""
    return frozenset(w for c in rec.get("icd10_codes") or [] for w in _words(str(c.get("description", "")))
                     if len(w) >= 3 and w not in ICD_STOPWORDS)


def is_icd_term(text: str, terms: frozenset) -> bool:
    """True if every word of a span occurs in the ICD-10 descriptions, i.e. it is diagnosis information
    ("diabetes", "right knee"), never a name unless a description contains that name."""
    words = [w for w in _words(text) if w not in ICD_STOPWORDS]
    return bool(words) and all(w in terms or w in ("left", "right") for w in words)


def is_protected(text: str, typ: str, terms: frozenset | None) -> bool:
    """An LLM span that generalize mode keeps: diagnosis words, never an age, date, contact or ID."""
    return bool(terms) and typ not in ("AGE", "DATE", "CONTACT", "ID") and is_icd_term(text, terms)


def parse_entities(raw: str) -> tuple[list, bool, int]:
    """(entities, parse_ok, n_salvaged). A reply cut off by the token limit (the model repeating one
    entity) keeps its finished entities instead of losing the whole chunk's spans."""
    parsed = parse_json_loose(raw or "")
    if parsed is None:
        ents = [o for o in salvage_objects(raw) if "text" in o]
        return ents, False, len(ents)
    ents = parsed.get("entities", []) if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
    return ents, True, 0


def llm_entities(chunk: str, model: str, num_ctx: int, cached: dict | None) -> tuple[list, dict]:
    """Entities for one chunk, from the model or replayed from a previous run's chunk log."""
    if cached is not None:
        ents = cached["entities"]
        if not cached.get("parse_ok") and cached.get("raw"):  # logs from before salvaging: re-read the raw reply
            ents = parse_entities(cached["raw"])[0]
        return ents, {"raw": cached.get("raw"), "parse_ok": cached.get("parse_ok"), "reused": True,
                      "salvaged": 0 if cached.get("parse_ok") else len(ents),
                      "prompt_tokens": 0, "gen_tokens": 0, "seconds": 0.0,
                      "reused_seconds": cached.get("seconds"), "done_reason": cached.get("done_reason")}
    r = chat(model, [{"role": "system", "content": EXTRACT_SYSTEM},
                     {"role": "user", "content": user_msg(chunk)}],
             fmt=EXTRACT_SCHEMA, num_ctx=num_ctx, num_predict=2048)
    ents, ok, n_salvaged = parse_entities(r["content"])
    return ents, {"raw": r["content"], "parse_ok": ok, "salvaged": n_salvaged,
                  "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"], "seconds": r["seconds"],
                  "done_reason": r["done_reason"]}


def anonymize_chunk_extract(chunk: str, model: str, num_ctx: int, cached: dict | None = None) -> tuple[str, dict]:
    ents, info = llm_entities(chunk, model, num_ctx, cached)
    out, kept = apply_entities(chunk, ents)
    return out, {"raw": info.pop("raw"), "parse_ok": info.pop("parse_ok"), "entities": kept, **info}


# ---------------------------------------------------------------- generalize mode

def anonymize_chunk_generalize(chunk: str, model: str, num_ctx: int, cached: dict | None = None,
                               protect: frozenset | None = None) -> tuple[str, dict]:
    """Extract-mode LLM spans + regex age/date spans; AGE/DATE rendered as ranges (generalize.py).

    `protect` is icd_terms() of the record (None with --no-icd-hints): LLM spans made only of those
    words stay in the text and are logged under "retained". Every applied span carries the action
    (removed / generalized), its replacement and the reason (generalize.explain).
    """
    ents, info = llm_entities(chunk, model, num_ctx, cached)
    _, llm_kept = apply_entities(chunk, ents)  # same filtering as extract; logged for --reuse-entities
    regex = generalize.find_spans(chunk)
    out, applied = apply_entities(chunk, regex + llm_kept, render=generalize.render, protect=protect)
    regex_texts, llm_texts = {e["text"].lower() for e in regex}, {e["text"].lower() for e in llm_kept}
    retained = [{**e, "action": "retained", "reason": "every word occurs in this encounter's ICD-10 code "
                 "descriptions, so it is diagnosis information, not an identifier"}
                for e in llm_kept if is_protected(e["text"], e["type"], protect)]
    for e in applied:
        t = e["text"].lower()
        e["source"] = "both" if t in regex_texts and t in llm_texts else "regex" if t in regex_texts else "llm"
        e["replacement"] = generalize.render(e["text"], e["type"])
        e["action"] = ("removed" if e["replacement"] == f"[{e['type']}]" else
                       "retained" if e["replacement"] == e["text"] else "generalized")
        e["reason"] = generalize.explain(e["type"], e["action"])
    retained += [e for e in applied if e["action"] == "retained"]
    applied = [e for e in applied if e["action"] != "retained"]
    return out, {"raw": info.pop("raw"), "parse_ok": info.pop("parse_ok"), "entities": llm_kept,
                 "regex_spans": regex, "applied": applied, "retained": retained, **info}


# ---------------------------------------------------------------- rewrite mode

def clean_rewrite(text: str) -> str:
    text = re.sub(r"^\s*```[a-z]*\n?|\n?```\s*$", "", text)
    text = re.sub(r"^\s*<<<\n?|\n?>>>\s*$", "", text)
    text = re.sub(r"^\s*(?:Text|Rewritten text)\s*:\s*\n", "", text, flags=re.IGNORECASE)
    return text.strip("\n")


def anonymize_chunk_rewrite(chunk: str, model: str, num_ctx: int) -> tuple[str, dict]:
    budget = len(chunk) // 2 + 512  # ~2x the input's token count; stops runaway generations
    r = chat(model, [{"role": "system", "content": REWRITE_SYSTEM},
                     {"role": "user", "content": user_msg(chunk)}],
             num_ctx=num_ctx, num_predict=budget)
    out = clean_rewrite(r["content"])
    return out, {"raw_len": len(r["content"]), "len_ratio": round(len(out) / max(1, len(chunk)), 3),
                 "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"], "seconds": r["seconds"],
                 "done_reason": r["done_reason"]}


# ---------------------------------------------------------------- driver

CHUNK_FN = {"extract": anonymize_chunk_extract, "generalize": anonymize_chunk_generalize,
            "rewrite": anonymize_chunk_rewrite}


def llm_chunks(rec: dict, field: str, chunk_chars: int) -> list[str]:
    """The non-blank chunks of a field, i.e. exactly the ones anonymize_record sends to the model."""
    text = rec.get(field) or ""
    return [c for c in chunk_lines(text, chunk_chars) if c.strip()] if text.strip() else []


def load_reuse(path: str, model: str) -> dict:
    """{(encounter_id, field): [chunk log, ...]} from an extract/generalize anonymization_log.jsonl."""
    reuse = {}
    for row in read_jsonl(path):
        if row.get("mode") not in ("extract", "generalize"):
            sys.exit(f"[anonymize] --reuse-entities: {path} has mode={row.get('mode')!r}; need extract or generalize")
        if row.get("model") != model:
            sys.exit(f"[anonymize] --reuse-entities: {path} was made with model={row.get('model')!r}, not {model!r}")
        for field, chunk_logs in row["fields"].items():
            reuse[(row["encounter_id"], field)] = chunk_logs
    return reuse


def check_reuse(records: list[dict], reuse: dict, chunk_chars: int) -> None:
    """Fail before any work if a chunk has no logged entities (so --reuse-entities never calls the model)."""
    bad = []
    for r in records:
        for field in ("dialogue", "note"):
            n = len(llm_chunks(r, field, chunk_chars))
            logged = reuse.get((r["encounter_id"], field), [])
            if n != len(logged) or any("entities" not in c for c in logged):
                bad.append(f"{r['encounter_id']}/{field}: {n} chunks, {len(logged)} logged")
    if bad:
        sys.exit(f"[anonymize] --reuse-entities does not cover {len(bad)} field(s): " + "; ".join(bad[:10]))


def anonymize_record(rec: dict, args) -> tuple[dict, dict]:
    out = {k: v for k, v in rec.items()}
    log = {"encounter_id": rec["encounter_id"], "mode": args.mode, "model": args.model, "fields": {}}
    reuse = getattr(args, "reuse", None)
    # generalize mode keeps diagnosis words of the record's own ICD-10 descriptions (all fields are
    # available at inference); --no-icd-hints simulates input without codes
    kw = {"protect": icd_terms(rec)} if args.mode == "generalize" and getattr(args, "icd_hints", True) else {}
    t0 = time.time()
    for field in ("dialogue", "note"):
        text = rec.get(field) or ""
        if args.mode == "identity" or not text.strip():
            continue
        fn = CHUNK_FN[args.mode]
        cached = reuse.get((rec["encounter_id"], field), []) if reuse is not None else None
        pieces, chunk_logs = [], []
        for chunk in chunk_lines(text, args.chunk_chars):
            if not chunk.strip():
                pieces.append(chunk)
                continue
            if cached is None:
                new, info = fn(chunk, args.model, args.num_ctx, **kw)
            elif len(chunk_logs) < len(cached):
                new, info = fn(chunk, args.model, args.num_ctx, cached=cached[len(chunk_logs)], **kw)
            else:
                raise RuntimeError(f"--reuse-entities has no entry for {rec['encounter_id']}/{field} "
                                   f"chunk {len(chunk_logs)}")
            pieces.append(new)
            chunk_logs.append(info)
        out[field] = "\n".join(pieces)
        log["fields"][field] = chunk_logs
    out["icd10_codes"] = rec["icd10_codes"]  # must be preserved verbatim
    log["seconds"] = round(time.time() - t0, 2)
    if reuse is not None:
        log["reused_seconds"] = round(sum(c.get("reused_seconds") or 0 for cs in log["fields"].values() for c in cs), 2)
    return out, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(EVAL_DATA))
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--mode", choices=["extract", "generalize", "rewrite", "identity"], default="extract")
    ap.add_argument("--model", default="qwen3.5:2b")
    ap.add_argument("--chunk-chars", type=int, default=2000, help="max characters per LLM call")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4, help="concurrent encounters (Ollama queues extras; vLLM batches them, use ~32)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N encounters (0 = all)")
    ap.add_argument("--fresh", action="store_true", help="ignore previous partial output")
    ap.add_argument("--reuse-entities", default=None, metavar="LOG",
                    help="extract/generalize anonymization_log.jsonl (same model and --chunk-chars) whose "
                         "per-chunk entities are replayed instead of calling the model")
    ap.add_argument("--no-icd-hints", dest="icd_hints", action="store_false",
                    help="generalize mode: do not use icd10_codes to keep diagnosis words (bonus task: codes unknown)")
    args = ap.parse_args()
    args.reuse = None
    if args.reuse_entities:
        if args.mode not in ("extract", "generalize"):
            sys.exit("[anonymize] --reuse-entities only applies to --mode extract or generalize")
        src_meta = Path(args.reuse_entities).parent / "anonymize_meta.json"
        args.src_meta = json.loads(src_meta.read_text()) if src_meta.exists() else {}
        if args.src_meta and args.src_meta.get("chunk_chars") != args.chunk_chars:
            sys.exit(f"[anonymize] --reuse-entities: {src_meta} has a different chunk_chars than {args.chunk_chars}")
        args.reuse = load_reuse(args.reuse_entities, args.model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "anonymized.partial.jsonl"
    log_path = out_dir / "anonymization_log.jsonl"
    if args.fresh:
        for p in (partial, log_path):
            p.unlink(missing_ok=True)

    records = read_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    done = {r["encounter_id"]: r for r in read_jsonl(partial)} if partial.exists() else {}
    todo = [r for r in records if r["encounter_id"] not in done]
    if args.reuse is not None:
        check_reuse(todo, args.reuse, args.chunk_chars)
    print(f"[anonymize] mode={args.mode} model={args.model} records={len(records)} "
          f"cached={len(records) - len(todo)} todo={len(todo)}"
          + (f" reusing entities from {args.reuse_entities}" if args.reuse is not None else ""), flush=True)

    out_w, log_w = JsonlAppender(partial), JsonlAppender(log_path)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(anonymize_record, r, args): r["encounter_id"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            eid = futs[fut]
            try:
                rec, log = fut.result()
            except Exception as e:  # keep going; a failed record is retried on the next run
                print(f"  ! {eid}: {e}", flush=True)
                continue
            out_w.append(rec)
            log_w.append(log)
            done[eid] = rec
            print(f"  [{i}/{len(todo)}] {eid} {log['seconds']}s", flush=True)

    ordered = [done[r["encounter_id"]] for r in records if r["encounter_id"] in done]
    write_jsonl(out_dir / "anonymized.jsonl", ordered)
    inp = Path(args.input).resolve()
    backend = LLM_BACKEND
    if args.mode == "identity":
        backend = None
    elif args.reuse is not None:  # the backend that made the replayed entities; old logs are Ollama-only
        backend = args.src_meta.get("backend", "ollama") if args.src_meta else None
    meta = {"mode": args.mode, "model": args.model, "backend": backend,
            "chunk_chars": args.chunk_chars, "num_ctx": args.num_ctx,
            "input": str(inp.relative_to(ROOT)) if inp.is_relative_to(ROOT) else str(inp),
            "n_records": len(ordered), "n_missing": len(records) - len(ordered),
            "wall_seconds_this_run": round(time.time() - t0, 1),
            "entities_reused": args.reuse is not None, "entities_reused_from": args.reuse_entities}
    if args.mode == "generalize":
        meta["generalize"] = "regex age/date pass + AGE/DATE rendered as ranges (harness/generalize.py)"
        meta["icd_hints"] = args.icd_hints
    (out_dir / "anonymize_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[anonymize] wrote {len(ordered)} records -> {out_dir / 'anonymized.jsonl'}"
          + (f" ({meta['n_missing']} failed; rerun to retry)" if meta["n_missing"] else ""))


if __name__ == "__main__":
    main()
