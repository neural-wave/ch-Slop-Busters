"""Adversarial re-identification attacks on every model's output (bonus task a), with the original text as positive control.

Usage: python3 -I eval attack [--models ...] [--suites aci_eval ...] [--attacks A1 A2 A3 A4 A5 | all] [--limit N]
                              [--device auto|cpu|cuda] [--llm MODEL]

Each attack runs twice per (model, suite): on the model's output (`anonymized`) and on the suite's original text
(`control`). The control shows what the attack gets when nothing was removed: an attack that recovers little from the
original says nothing about the anonymization. Writes `data/eval/attacks/<model>/<suite>.json`:

    {"model", "suite", "n_records", "gold_level",
     "attacks": {"A1".."A5": {"name", "anonymized": ARM, "control": ARM, "notes", "error"?, ...}},
     "max_recovered", "sources"}
    ARM = {"recovered": hits / total, "hits", "total", "by_category": {category: share}}

`max_recovered` is the largest anonymized `recovered` over the attacks that ran, A5 left out (an upper bound by design).

A1  NER residual scan: independent detectors (deid_roberta i2b2, GLiNER multi-PII and the regex layer of the VM code,
    run by bonus/_ner_worker.py with the anonymous venv) scan each field; recovered = gold items (not low risk) covered by a
    detected span: the span contains the item, or is whole words of it (3+ characters, not only titles and cue words
    such as `Ms.`). Spans with a placeholder (`[NAME]`) don't count.
A2  roster dictionary: the attacker knows every person name of the suite's gold and a public first-name list (Kaggle
    krist0phersmith/pii-real-names-dataset, nam_dict.txt) and greps for them; recovered = name tokens of gold names
    still present (whole word) in their encounter's field.
A3  rare words: tokens of 4+ letters that occur in one encounter only, are not clinical (ICD-10-CM words, RxNorm drug
    names), not ordinary English words, not months or weekdays, and look like identifiers (a proper noun of the English
    word list or a known first name); recovered = such tokens still present in their encounter. It needs no gold:
    by_category is the category of the gold item that holds the token, or UNLABELLED.
A4  LLM inference: an LLM reads the record and guesses the patient's name, age, place and employer or job; scored
    against the gold values (any name token, age within 1 year, a gold place or job inside the guess). Default
    attacker: the local Ollama qwen3.5:2b-q4_K_M. `--llm claude-*` runs it through bonus/_claude_worker.py
    (`uv run --with anthropic`, 8 requests at a time, ANTHROPIC_API_KEY) and sends the record text to the API.
A6  LLM agent adversary: the same guesses made by LLM agents from a prompt, one agent per blinded model output, and
    scored by the same rule (bonus/adversary.py: `eval adversary export`, the agents, `eval adversary import`). This
    command never runs it and keeps its results.
A5  record linkage: TF-IDF nearest neighbour from each anonymized record to the original records (placeholders
    removed). Clinical content is kept by design, so text similarity re-links nearly every record: this bounds the
    threat model (anonymization hides who the patient is, not which record it was from someone who holds the originals).

Gold: the suite's `pii` items; when they carry vote metadata (the merged ACI gold), only the consensus level (direct
identifiers or 2+ votes), recorded as `gold_level`. Defaults: every model with a run, suite aci_eval, attacks A2 A3 A5
(standard library, about 2.5 s per model and suite). A1 and A4 run only when asked (`--attacks all`): A1 loads two
transformer models and scans about one text (a dialogue or a note) per second on the RTX 4050, ~10 min for 4 models on
aci_eval (`--device auto` takes the GPU when 2.5 GB are free, else the CPU with 4 threads, an order of magnitude
slower); A4 makes one LLM call per record and version (the control is shared by every model; about 3,500 input
tokens per call with Sonnet, so 67 records cost roughly $0.6 per model; an Ollama model takes 1.4 s per call on an idle
GPU). Both cache by text under data/eval/cache/attacks/ (atomic writes, so a killed run
resumes), the control and unchanged outputs are computed once, and a later run without them keeps their results while
the model output, the suite and `--limit` are unchanged. A suite without a run or without PII gold is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path

from eval import metrics as M
from eval import suites as S
from eval.common import (CACHE_DIR, REPO, DATA, FIELDS, OUT, RUNS_DIR, SUITES_DIR, count, load_jsonl, pattern, ratio, sha1_file, sha1_text,
                         words, write_json, write_jsonl)  # fmt: skip

ATTACKS_DIR = OUT / "attacks"
ACACHE = CACHE_DIR / "attacks"
NAMES = {"A1": "NER residual scan", "A2": "roster dictionary", "A3": "rare words", "A4": "LLM inference", "A5": "record linkage",
         "A6": "LLM agent adversary"}  # fmt: skip
AUTOMATED = ["A1", "A2", "A3", "A4", "A5"]  # A6 comes from `eval adversary import`, never from this command, which keeps it
DEFAULT_ATTACKS = ["A2", "A3", "A5"]
ANONYMOUS_PYTHON = REPO / "models" / "anonymous" / ".venv" / "bin" / "python"
VM_ROOT = REPO / "models" / "anonymous"
WORKER = Path(__file__).with_name("_ner_worker.py")
CLAUDE_WORKER = Path(__file__).with_name("_claude_worker.py")
DEFAULT_LLM = "qwen3.5:2b-q4_K_M"  # the coders' Ollama model; `--llm claude-sonnet-5-5` sends the records to the API instead
NER_LAYERS = "regex,deid,gliner"
GPU_MIN_FREE_MB = 2500
WORKER_TIMEOUT_S = 7200
NAM_DICT = DATA / "kaggle" / "krist0phersmith__pii-real-names-dataset" / "0717-182" / "0717-182" / "nam_dict.txt"
RXNORM = DATA / "kaggle" / "nguyenchuhunganh__medical-code" / "rxnorm_final_dictionary.csv"
PLACEHOLDER = re.compile(r"\[[A-Z_0-9+<> ]+\]")
TITLES = frozenset("dr mr mrs ms miss mister doctor prof md".split())
CALENDAR = frozenset("""january february march april june july august september october november december monday tuesday
    wednesday thursday friday saturday sunday mondays tuesdays wednesdays thursdays fridays saturdays sundays""".split())
NUMBER_WORDS = {w: i for i, w in enumerate("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
    seventeen eighteen nineteen""".split())} | {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}  # fmt: skip


# ---------------------------------------------------------------- shared pieces


def cache_read(path: Path):
    """A cached JSON value, or None when absent or unreadable (a file cut by a kill is recomputed)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cache_write(path: Path, obj) -> None:
    """Write through a temporary file and rename, so a run killed mid-write leaves no half-written cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def arm(hits: Counter, total: Counter) -> dict:
    h, t = sum(hits.values()), sum(total.values())
    return {"recovered": ratio(h, t), "hits": h, "total": t, "by_category": {c: ratio(hits[c], total[c]) for c in sorted(total)}}


def result(key: str, anonymized: dict | None, control: dict | None, notes: str, **extra) -> dict:
    return {"name": NAMES[key], "anonymized": anonymized, "control": control, "notes": notes, **extra}


def failed(key: str, error: str, notes: str = "") -> dict:
    return {"name": NAMES[key], "anonymized": None, "control": None, "notes": notes, "error": error}


def gold_level(labels: dict[str, dict]) -> str:
    """`consensus` when the gold items carry vote metadata (merged ACI gold), else `all`."""
    return "consensus" if any("votes" in it or "direct" in it for g in labels.values() for it in g.get("pii", [])) else "all"


def gold_items(g: dict, level: str) -> list[dict]:
    items = g.get("pii") or []
    return [it for it in items if it.get("direct") or (it.get("votes") or 0) >= 2] if level == "consensus" else items


def name_tokens(text: str) -> list[str]:
    return [t for t, _, _ in words(text) if len(t) >= 2 and t.replace("'", "").replace("’", "").isalpha() and t not in TITLES]


def record_text(r: dict) -> str:
    return (r.get("dialogue") or "") + "\n" + (r.get("note") or "")


@cache
def first_names() -> frozenset[str]:
    """Lowercase first names of nam_dict.txt (Latin letters; `Jun+Wei` gives jun and wei). Empty when not downloaded."""
    out: set[str] = set()
    if NAM_DICT.exists():
        for line in NAM_DICT.open(encoding="latin-1"):
            if line[:1] in "#=" or len(line) < 30 or line[29] == "+":  # comments, equivalences, duplicate umlaut sorting
                continue
            out.update(t for t in re.split(r"[\s+\-]+", line[3:29].strip().lower()) if re.fullmatch(r"[a-z]{2,}", t))
    return frozenset(out)


@cache
def proper_nouns() -> frozenset[str]:
    """Capitalized entries of the English word list that are no lowercase entry (`Boston`, `Martha`, `Walmart`)."""
    for p in M.WORDLISTS:
        if p.exists():
            lines = [w.strip() for w in p.open(encoding="utf-8", errors="ignore") if "'" not in w]
            return frozenset({w.lower() for w in lines if w[:1].isupper()} - {w for w in lines if w[:1].islower()})
    return frozenset()


@cache
def drug_words() -> frozenset[str]:
    """Single-word RxNorm names of 4+ letters (`lisinopril`). Empty when not downloaded."""
    out: set[str] = set()
    if RXNORM.exists():
        with RXNORM.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                out.update(s for s in (x.strip().lower() for x in (row.get("Synonyms") or "").split("||")) if re.fullmatch(r"[a-z]{4,}", s))
    return frozenset(out)


# ---------------------------------------------------------------- A1 NER residual scan


def ner_config() -> str:
    """sha1 of what determines the detections: layers, worker and the VM detector code."""
    parts = [NER_LAYERS] + [p.read_text(encoding="utf-8") if p.exists() else "" for p in (WORKER, VM_ROOT / "detect" / "ml_layers.py", VM_ROOT / "detect" / "regex_layer.py")]
    return sha1_text(*parts)[:16]


def _ner_cache(config: str, text: str) -> Path:
    return ACACHE / "ner" / config / f"{sha1_text(text)}.json"


def gpu_free_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    vals = [float(x) for x in out.split() if re.fullmatch(r"\d+(\.\d+)?", x)]
    return max(vals) if vals else None


def run_worker(texts: list[str], device: str) -> dict[str, list]:
    """Spans per text from the NER worker (the anonymous venv); raises RuntimeError when it cannot run or fails."""
    if not ANONYMOUS_PYTHON.exists():
        raise RuntimeError(f"anonymous venv not found: {ANONYMOUS_PYTHON}")
    if device == "auto":
        device = "cuda" if (gpu_free_mb() or 0) >= GPU_MIN_FREE_MB else "cpu"
    with tempfile.TemporaryDirectory(prefix="nw-attack-") as tmp:
        inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        inp.write_text("".join(json.dumps({"key": i, "text": t}, ensure_ascii=False) + "\n" for i, t in enumerate(texts)), encoding="utf-8")
        print(f"A1: scanning {len(texts)} texts on {device}", flush=True)
        try:
            proc = subprocess.run([str(ANONYMOUS_PYTHON), "-I", "-B", str(WORKER), str(inp), str(outp), "--layers", NER_LAYERS, "--device", device,
                                   "--root", str(VM_ROOT)], cwd=tmp, env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
                                  capture_output=True, text=True, timeout=WORKER_TIMEOUT_S)  # fmt: skip
            rc, err = proc.returncode, proc.stderr or ""
        except subprocess.TimeoutExpired as e:  # what it finished is kept (and cached), the rest is scanned next time
            rc, err = 124, e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr or ""
        out = {}
        for line in outp.open(encoding="utf-8") if outp.exists() else []:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut by the kill
                continue
            out[texts[row["key"]]] = row["spans"]
        if rc or len(out) < len(set(texts)):
            tail = " | ".join(err.strip().splitlines()[-3:])
            if not out:
                raise RuntimeError(f"NER worker failed (exit {rc}): {tail}")
            print(f"warning: NER worker stopped after {len(out)} of {len(texts)} texts (exit {rc}): {tail}", file=sys.stderr)
        return out


def ner_spans(texts: set[str], device: str, worker: Callable[[list[str], str], dict[str, list]] | None = None) -> dict[str, list]:
    """Detections for every text, from the cache or the worker (new ones are cached; texts the worker missed are absent)."""
    config = ner_config()
    out, todo = {}, []
    for t in sorted(texts):
        spans = cache_read(_ner_cache(config, t))
        if spans is not None:
            out[t] = spans
        else:
            todo.append(t)
    if todo:
        for t, spans in (worker or run_worker)(todo, device).items():
            cache_write(_ner_cache(config, t), spans)
            out[t] = spans
    return out


def informative(span: str) -> bool:
    """A detected span that can stand for part of a gold item: 3+ characters and a word other than a title or cue word
    (`Ms.` alone says nothing about `Ms. Hill`)."""
    return len(span) >= 3 and any(t not in M.CUE_WORDS and t not in TITLES for t, _, _ in words(span))


def a1_arm(original: list[dict], recs: list[dict], gold: dict[str, list[dict]], spans: dict[str, list]) -> tuple[dict, list[dict]]:
    hits, total, found = Counter(), Counter(), []
    for o, r in zip(original, recs):
        for it in gold.get(o["encounter_id"], []):
            f = it["field"]
            if it.get("risk") == "low" or not count(o[f], it["text"]):
                continue
            total[it["category"]] += 1
            det = [d for d in (s[2].strip() for s in spans.get(r.get(f) or "", [])) if d and not PLACEHOLDER.search(d)]
            if any(pattern(it["text"]).search(d) or (informative(d) and pattern(d).search(it["text"])) for d in det):
                hits[it["category"]] += 1
                found.append({"encounter_id": o["encounter_id"], "field": f, "text": it["text"], "category": it["category"]})
    return arm(hits, total), found


def a1(original: list[dict], anonymized: list[dict], gold: dict[str, list[dict]], spans: dict[str, list]) -> dict:
    missing = sum(1 for r in original + anonymized for f in FIELDS if r.get(f) and r[f] not in spans)
    if missing:
        return failed("A1", f"{missing} texts were not scanned (NER worker incomplete)")
    anon, found = a1_arm(original, anonymized, gold, spans)
    notes = f"detectors: {NER_LAYERS} (VM code, anonymous venv); low-risk items excluded. anonymous uses the same detectors, so for it A1 is not independent"
    return result("A1", anon, a1_arm(original, original, gold, spans)[0], notes, found=found[:50])


# ---------------------------------------------------------------- A2 roster dictionary


def a2(original: list[dict], anonymized: list[dict], gold: dict[str, list[dict]]) -> dict:
    corpus = {t for items in gold.values() for it in items if it["category"] in M.NAME_CATS for t in name_tokens(it["text"]) if len(t) >= 3}
    public = first_names()
    roster = corpus | public

    def run(recs: list[dict]) -> tuple[dict, list[dict]]:
        hits, total, found = Counter(), Counter(), []
        for o, r in zip(original, recs):
            for it in gold.get(o["encounter_id"], []):
                f = it["field"]
                if it["category"] not in M.NAME_CATS or not count(o[f], it["text"]):
                    continue
                n0 = count(o[f], it["text"])
                for t in name_tokens(it["text"]):
                    if t in roster:
                        total[it["category"]] += 1
                        if count(r.get(f) or "", t) > max(0, count(o[f], t) - n0 * count(it["text"], t)):  # more than its uses outside the name
                            hits[it["category"]] += 1
                            found.append({"encounter_id": o["encounter_id"], "field": f, "token": t, "category": it["category"]})
        return arm(hits, total), found

    anon, found = run(anonymized)
    return result("A2", anon, run(original)[0], "name tokens of gold person names still in their field (whole word, more often than the token "
                  "occurs outside the name in the original); roster = the suite's gold names + nam_dict.txt first names", roster_size=len(roster), public_names=len(public), found=found[:50])  # fmt: skip


# ---------------------------------------------------------------- A3 rare words


def rare_tokens(recs: list[dict]) -> dict[str, set[str]]:
    """Per encounter: identifier-like tokens that occur in no other encounter of `recs`."""
    clinical, common, drugs = M.clinical_vocabulary(), M.common_words()[0], drug_words()
    known = proper_nouns() | first_names()
    per, df = {}, Counter()
    for r in recs:
        toks = {t.removesuffix("'s").removesuffix("’s") for f in FIELDS for t, _, _ in words(r.get(f) or "")}
        toks = {t for t in toks if len(t) >= 4 and t.isalpha() and t not in clinical and t not in common and t not in drugs and t not in CALENDAR}
        per[r["encounter_id"]] = toks
        df.update(toks)
    return {eid: {t for t in toks if df[t] == 1 and t in known} for eid, toks in per.items()}


def a3(original: list[dict], anonymized: list[dict], gold: dict[str, list[dict]]) -> dict:
    rare = rare_tokens(original)

    def category(eid: str, tok: str) -> str:
        return next((it["category"] for it in gold.get(eid, []) if tok in {t for t, _, _ in words(it["text"])}), "UNLABELLED")

    def run(recs: list[dict]) -> tuple[dict, list[dict]]:
        hits, total, found = Counter(), Counter(), []
        for o, r in zip(original, recs):
            text = record_text(r)
            for tok in sorted(rare.get(o["encounter_id"], ())):
                cat = category(o["encounter_id"], tok)
                total[cat] += 1
                if count(text, tok):
                    hits[cat] += 1
                    found.append({"encounter_id": o["encounter_id"], "token": tok, "category": cat})
        return arm(hits, total), found

    anon, found = run(anonymized)
    return result("A3", anon, run(original)[0], "encounter-unique identifier-like tokens (not clinical, drug, dictionary or calendar words; a proper noun "
                  "or first name) still present; needs no gold, the control is 1 by construction", found=found[:50])  # fmt: skip


# ---------------------------------------------------------------- A4 LLM inference

ATTACK_PROMPT = """You are a privacy auditor testing a de-identified medical record. Using every clue in the text,
guess the patient's identity. If the information was removed, still give your best guess.
Answer JSON only: {"first_name": "", "last_name": "", "age": 0, "city_or_state": "", "employer_or_job": ""}

RECORD:
"""
GUESS_SCHEMA = {"type": "object", "properties": {"first_name": {"type": "string"}, "last_name": {"type": "string"}, "age": {"type": "integer"},
                "city_or_state": {"type": "string"}, "employer_or_job": {"type": "string"}},
                "required": ["first_name", "last_name", "age", "city_or_state", "employer_or_job"]}  # fmt: skip
LLM_CONTEXT, LLM_OUTPUT, LLM_RESERVE = 16384, 256, 1024


def prompt(r: dict) -> str:
    """The attack prompt: the note (up to 4,500 characters) then the dialogue, cut to fit the context (UTF-8 bytes)."""
    text = ((r.get("note") or "")[:4500] + "\n\nDIALOGUE (excerpt):\n" + (r.get("dialogue") or ""))[:9000]
    budget = LLM_CONTEXT - LLM_OUTPUT - LLM_RESERVE - len(ATTACK_PROMPT.encode("utf-8")) - 64
    while len(text.encode("utf-8")) > budget:
        text = text[: int(len(text) * 0.9)]
    return ATTACK_PROMPT + text


def age_of(text: str) -> int | None:
    """Age in years of a gold AGE string (`52-year-old`, `fifty two`), None for months, weeks or no number."""
    t = text.lower()
    if re.search(r"month|week", t):
        return None
    m = re.search(r"\d{1,3}", t)
    if m:
        return int(m.group())
    vals = [NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", t) if w in NUMBER_WORDS]
    return sum(vals) or None


def guess_hits(guess: dict, items: list[dict]) -> tuple[Counter, Counter]:
    """(hits, totals) by guessed attribute for one record."""
    hits, total = Counter(), Counter()
    names = {t for it in items if it["category"] == "PATIENT_NAME" for t in name_tokens(it["text"])}
    if names:
        total["name"] += 1
        hits["name"] += bool(set(name_tokens(f"{guess.get('first_name', '')} {guess.get('last_name', '')}")) & names)
    age = next((a for a in (age_of(it["text"]) for it in items if it["category"] == "AGE") if a), None)
    if age:
        total["age"] += 1
        try:
            hits["age"] += abs(int(float(guess.get("age") or 0)) - age) <= 1
        except (TypeError, ValueError):
            pass
    for key, cats, field in (("place", {"LOCATION"}, "city_or_state"), ("employer_or_job", {"OCCUPATION", "ORGANIZATION"}, "employer_or_job")):
        vals = {it["text"] for it in items if it["category"] in cats}
        if vals:
            total[key] += 1
            g = str(guess.get(field) or "")
            hits[key] += bool(g.strip()) and any(pattern(v).search(g) for v in vals)
    return hits, total


def claude_many(model: str, prompts: list[str]) -> list[dict | str]:
    """One guess (dict) or error code (str) per prompt, from bonus/_claude_worker.py."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "prompts.jsonl", Path(tmp) / "guesses.jsonl"
        write_jsonl(inp, ({"key": i, "prompt": p} for i, p in enumerate(prompts)))
        cmd = ["uv", "run", "--quiet", "--no-project", "--with", "anthropic", "python3", "-I", str(CLAUDE_WORKER), "--model", model,
               "--schema", json.dumps(GUESS_SCHEMA), str(inp), str(outp)]  # fmt: skip
        try:
            rc = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL, timeout=WORKER_TIMEOUT_S).returncode
        except (OSError, subprocess.TimeoutExpired) as e:
            return [f"worker:{type(e).__name__}"] * len(prompts)
        rows = {r["key"]: r for r in load_jsonl(outp)} if outp.exists() else {}
    return [rows[i].get("guess") or rows[i].get("error", "invalid_output") if i in rows else f"worker:exit_{rc}" for i in range(len(prompts))]


def a4(original: list[dict], anonymized: list[dict], gold: dict[str, list[dict]], model: str, generate: Callable | None = None,
       generate_many: Callable | None = None) -> dict:  # fmt: skip
    """`generate(model, prompt, schema, ...)`: one Ollama call; `generate_many(model, prompts)`: a Claude batch (one guess or
    error code per prompt). A claude-* model uses claude_many unless a test passes generate_many."""
    from eval.coders import ollama

    if model.startswith("claude-"):
        digest = "api"  # Claude model ids name a fixed snapshot
        if generate_many is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                return failed("A4", "ANTHROPIC_API_KEY not set")
            generate_many = claude_many
    elif generate is None:
        if not ollama.available():
            return failed("A4", f"Ollama not reachable at {ollama.URL}")
        try:
            digest = ollama.identity(model)["digest"]
        except ollama.PredictionError as e:
            return failed("A4", f"{model}: {e.code}")
        generate = ollama.generate_json
    else:
        digest = "test"
    errors, streak, calls = Counter(), 0, Counter()

    def path_of(p: str) -> Path:
        return ACACHE / "llm" / f"{sha1_text(model, digest, json.dumps(GUESS_SCHEMA, sort_keys=True), p)}.json"

    batch_errors: dict[str, str] = {}
    if generate_many is not None:  # every prompt this job needs, in one batch; the cache makes reruns free
        needed = {path_of(p): p for o, r in [*zip(original, anonymized), *zip(original, original)] if gold.get(o["encounter_id"]) for p in [prompt(r)]}
        todo = {path: p for path, p in needed.items() if not isinstance(cache_read(path), dict)}
        if todo:
            print(f"A4: {len(todo)} new {model} calls", flush=True)
            for (path, p), g in zip(todo.items(), generate_many(model, list(todo.values()))):
                if isinstance(g, dict):
                    cache_write(path, g)
                    calls["new"] += 1
                else:
                    batch_errors[p] = g
            if batch_errors and len(batch_errors) == len(needed):
                return failed("A4", f"every {model} call failed ({', '.join(sorted(set(batch_errors.values())))})")

    def guess(r: dict) -> dict:
        nonlocal streak
        p = prompt(r)
        path = path_of(p)
        cached = cache_read(path)
        if isinstance(cached, dict):
            return cached
        if generate_many is not None:
            errors[batch_errors.get(p, "missing")] += 1
            return {}
        try:
            g = generate(model, p, GUESS_SCHEMA, seed=0, context=LLM_CONTEXT, max_output=LLM_OUTPUT)
        except ValueError as e:  # ollama.PredictionError
            errors[getattr(e, "code", "error")] += 1
            streak += 1
            if streak >= 3:
                raise RuntimeError(f"3 consecutive failures ({', '.join(errors)})") from None
            return {}
        streak = 0
        cache_write(path, g)
        calls["new"] += 1
        if calls["new"] % 20 == 0:
            print(f"A4: {calls['new']} new LLM calls", flush=True)
        return g

    def run(recs: list[dict]) -> tuple[dict, list[dict]]:
        hits, total, rows = Counter(), Counter(), []
        for o, r in zip(original, recs):
            items = gold.get(o["encounter_id"], [])
            if not items:
                continue
            g = guess(r)
            h, t = guess_hits(g, items)
            hits.update(h)
            total.update(t)
            rows.append({"encounter_id": o["encounter_id"], "guess": g, "hits": sorted(+h)})
        return arm(hits, total), rows

    try:
        anon, rows = run(anonymized)
        control = run(original)[0]
    except RuntimeError as e:
        return failed("A4", str(e))
    return result("A4", anon, control, f"{model}: guesses of name (any token), age (within 1 year), place and employer or job (gold value inside the "
                  "guess) against the gold of the record", llm=model, digest=digest, errors=dict(errors), new_calls=calls["new"], guesses=rows)  # fmt: skip


# ---------------------------------------------------------------- A5 record linkage

TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tfidf(docs: list[str]) -> list[dict[str, float]]:
    """L2-normalized TF-IDF vectors (sublinear tf, smoothed idf: scikit-learn's TfidfVectorizer(sublinear_tf=True))."""
    tfs = [Counter(TOKEN.findall(d.lower())) for d in docs]
    df = Counter(t for c in tfs for t in c)
    n = len(docs)
    out = []
    for c in tfs:
        v = {t: (1 + math.log(k)) * (math.log((1 + n) / (1 + df[t])) + 1) for t, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / norm for t, x in v.items()})
    return out


def linkage(original: list[dict], recs: list[dict]) -> tuple[int, int]:
    """(records whose nearest original is their own, records): TF-IDF fitted on both sets, ties to the first original."""
    docs_o = [record_text(r) for r in original]
    docs_a = [PLACEHOLDER.sub(" ", record_text(r)) for r in recs]
    vecs = tfidf(docs_o + docs_a)
    index: dict[str, list[tuple[int, float]]] = {}
    for j, v in enumerate(vecs[: len(docs_o)]):
        for t, w in v.items():
            index.setdefault(t, []).append((j, w))
    top1 = 0
    for i, v in enumerate(vecs[len(docs_o) :]):
        sim = [0.0] * len(docs_o)
        for t, x in v.items():
            for j, w in index.get(t, ()):
                sim[j] += x * w
        top1 += max(range(len(sim)), key=lambda j: (sim[j], -j)) == i
    return top1, len(docs_a)


def a5(original: list[dict], anonymized: list[dict]) -> dict:
    def run(recs: list[dict]) -> dict:
        hits, n = linkage(original, recs)
        return {"recovered": ratio(hits, n), "hits": hits, "total": n, "by_category": {}}

    return result("A5", run(anonymized), run(original), "TF-IDF nearest original of each anonymized record (placeholders removed). The attacker already "
                  "holds the originals and clinical content is kept by design, so this is an upper bound: anonymization protects who the "
                  "patient is, not which record it was. Excluded from max_recovered")  # fmt: skip


# ---------------------------------------------------------------- driver


def load(model: str, suite: str, limit: int = 0) -> dict | str:
    """Everything one (model, suite) needs, or why it is skipped."""
    sdir, rdir = SUITES_DIR / suite, RUNS_DIR / model / suite
    if not (sdir / "encounters.jsonl").exists():
        return "suite not built"
    if not (rdir / "run.json").exists():
        return "no run"
    run = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
    if run.get("exit_code") != 0 or not (rdir / "output.jsonl").exists():
        return f"run failed (exit {run.get('exit_code')})"
    labels = {g["encounter_id"]: g for g in load_jsonl(sdir / "labels.jsonl")}
    if not any(g.get("pii") for g in labels.values()):
        return "no PII gold"
    try:
        out = {r.get("encounter_id"): r for r in load_jsonl(rdir / "output.jsonl")}
    except json.JSONDecodeError:
        return "output is not valid JSONL"
    pairs = [(o, out[o["encounter_id"]]) for o in load_jsonl(sdir / "encounters.jsonl") if isinstance(out.get(o["encounter_id"]), dict)]
    pairs = [(o, {f: a.get(f) if isinstance(a.get(f), str) else "" for f in ("encounter_id", *FIELDS)}) for o, a in pairs][: limit or None]
    level = gold_level(labels)
    info = json.loads((sdir / "suite.json").read_text(encoding="utf-8")) if (sdir / "suite.json").exists() else {}
    return {"model": model, "suite": suite, "original": [o for o, _ in pairs], "anonymized": [a for _, a in pairs], "level": level,
            "gold": {o["encounter_id"]: gold_items(labels.get(o["encounter_id"], {}), level) for o, _ in pairs},
            "sources": {"input_sha1": info.get("input_sha1"), "labels_sha1": info.get("labels_sha1"), "output_sha1": sha1_file(rdir / "output.jsonl"),
                        "limit": limit}}  # fmt: skip


def attack(job: dict, attacks: list[str], spans: dict[str, list] | str | None = None, llm: str | None = None) -> dict:
    """Run `attacks` on one loaded job. `spans`: A1 detections per text, or the error that prevented them."""
    o, a, g = job["original"], job["anonymized"], job["gold"]
    res = {}
    for key in attacks:
        try:
            if key == "A1":
                res[key] = failed("A1", spans) if isinstance(spans, str) else a1(o, a, g, spans or {})
            elif key == "A2":
                res[key] = a2(o, a, g)
            elif key == "A3":
                res[key] = a3(o, a, g)
            elif key == "A4":
                res[key] = a4(o, a, g, llm or DEFAULT_LLM)
            elif key == "A5":
                res[key] = a5(o, a)
        except Exception as e:  # one broken attack must not lose the others
            res[key] = failed(key, f"{type(e).__name__}: {e}")
    return res


def max_recovered(attacks: dict[str, dict]) -> float | None:
    vals = [r["anonymized"]["recovered"] for k, r in attacks.items() if k != "A5" and r.get("anonymized") and r["anonymized"].get("recovered") is not None]
    return max(vals) if vals else None


def write(job: dict, results: dict[str, dict]) -> dict:
    """Write the attack file, keeping earlier results of attacks not rerun while their inputs are unchanged."""
    path = ATTACKS_DIR / job["model"] / f"{job['suite']}.json"
    merged = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
        if prev.get("sources") == job["sources"] and prev.get("gold_level") == job["level"]:
            merged = {k: v for k, v in prev.get("attacks", {}).items() if k in NAMES}
    merged.update(results)
    attacks = {k: merged[k] for k in NAMES if k in merged}
    out = {"model": job["model"], "suite": job["suite"], "n_records": len(job["original"]), "gold_level": job["level"], "attacks": attacks,
           "max_recovered": max_recovered(attacks), "sources": job["sources"]}  # fmt: skip
    write_json(path, out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval attack", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--models", nargs="+", help="default: every model with a run")
    p.add_argument("--suites", nargs="+", default=["aci_eval"], choices=list(S.SUITES))
    p.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS, choices=[*AUTOMATED, "all"], help="default: A2 A3 A5 (all = A1..A5)")
    p.add_argument("--limit", type=int, default=0, help="only the first N records of each suite (development)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="A1 detectors (auto: the GPU when 2.5 GB are free)")
    p.add_argument("--llm", default=DEFAULT_LLM, help=f"A4 attacker: an Ollama model (default {DEFAULT_LLM}) or a claude-* model (API)")
    a = p.parse_args(argv)
    attacks = list(AUTOMATED) if "all" in a.attacks else [k for k in AUTOMATED if k in a.attacks]
    models = a.models or (sorted(d.name for d in RUNS_DIR.iterdir() if d.is_dir()) if RUNS_DIR.is_dir() else [])
    jobs = []
    for s in a.suites:
        for m in models:
            job = load(m, s, a.limit)
            if isinstance(job, str):
                print(f"{m:18s} {s:16s} skipped: {job}")
            else:
                jobs.append(job)
    spans: dict[str, list] | str | None = None
    if "A1" in attacks and jobs:
        texts = {r[f] for j in jobs for r in j["original"] + j["anonymized"] for f in FIELDS if r.get(f)}
        try:
            spans = ner_spans(texts, a.device)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            spans = str(e)
            print(f"A1: {spans}", file=sys.stderr)
    for job in jobs:
        out = write(job, attack(job, attacks, spans, a.llm))
        cells = []
        for k, r in out["attacks"].items():
            an, co = r.get("anonymized") or {}, r.get("control") or {}
            cells.append(f"{k} " + ("error" if r.get("error") else f"{an.get('recovered')}/{co.get('recovered')}"))
        print(f"{job['model']:18s} {job['suite']:16s} n={out['n_records']:<4d} max {out['max_recovered']}  " + "  ".join(cells) + "  (anonymized/control)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
