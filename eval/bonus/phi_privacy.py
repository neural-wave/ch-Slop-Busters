"""Independent privacy check: a fine-tuned PII detector scans the original text (before) and each model's output (after).

Usage: python3 -I eval phi [--models ...] [--suites aci_eval ...] [--model OLLAMA_MODEL] [--limit N]

The gold-based privacy metric counts what the golds list; this one needs no gold. The detector is
`ab-ai/PII-Model-Phi3-Mini` (Phi-3-mini-4k-instruct fine-tuned for PII extraction, MIT), as the Q4_K_M GGUF of
mradermacher/PII-Model-Phi3-Mini-GGUF served by Ollama (`ollama pull hf.co/mradermacher/PII-Model-Phi3-Mini-GGUF:Q4_K_M`,
2.4 GB). It is evaluation only, never a model row. Each field is cut at line breaks into chunks of at most 2,000
characters (its context is 4k tokens) and sent raw with the model card's prompt; the answer is a dict {label: [values]}.
Its labels map to the gold taxonomy (LABELS). Labels that are not identifying in clinical text (sex, gender, height,
eye colour, amounts, currencies, title prefixes) are dropped, and so are values that hold a placeholder (`[NAME]`).

Writes data/eval/phi/<model>/<suite>.json:

    {"model", "suite", "n_records", "phi_model", "digest",
     "total": {"before", "after", "removed"}, "by_category": {category: {"before", "after", "removed"}},
     "errors", "new_calls"}

before / after = distinct (category, value) mentions the detector lists per chunk of the original / output text;
removed = 1 − after / before, clipped at 0 when it finds more after than before; None without any mention before. A
generalized value (`fifty-something`) still counts after when the detector reads it as an age. The detector is noisy
(it reads `Chief Complaint` as a job, an age as a DOB): a false positive in text no model touches counts both before
and after, so `removed` is a lower bound and best compared between models. A chunk whose answer cannot be parsed
counts in `errors` and as no mention.

Run it LAST and alone: it unloads every other Ollama model first, keeps only the detector loaded, and unloads it at
the end. Answers are cached by chunk text under data/eval/cache/phi/ (atomic writes, so a killed run resumes): the
original text is scanned once for every model and an unchanged chunk is never scanned twice. About 6 chunks per ACI
record, so roughly 400 calls for the originals and up to 400 per model.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import suites as S
from eval.common import CACHE_DIR, FIELDS, OUT, RUNS_DIR, SUITES_DIR, load_jsonl, ratio, sha1_text, write_json

PHI_DIR = OUT / "phi"
PCACHE = CACHE_DIR / "phi"
DEFAULT_MODEL = "hf.co/mradermacher/PII-Model-Phi3-Mini-GGUF:Q4_K_M"
CHUNK_CHARS = 2000
CONTEXT, MAX_OUTPUT = 4096, 512

# the model card's label list and prompt, verbatim (the fine-tune was trained on it)
CARD_LABELS = ("companyname, pin, currencyname, email, phoneimei, litecoinaddress, currency, eyecolor, street, mac, state, time, vehiclevin, "
               "jobarea, date, bic, currencysymbol, currencycode, age, nearbygpscoordinate, amount, ssn, ethereumaddress, zipcode, "
               "buildingnumber, dob, firstname, middlename, ordinaldirection, jobtitle, bitcoinaddress, jobtype, phonenumber, height, password, "
               "ip, useragent, accountname, city, gender, secondaryaddress, iban, sex, prefix, ipv4, maskednumber, url, username, lastname, "
               "creditcardcvv, county, vehiclevrm, ipv6, creditcardissuer, accountnumber, creditcardnumber")  # fmt: skip
PROMPT = ("### Instruction:\n    Identify and extract the following PII entities from the text, if present: " + CARD_LABELS
          + ". Return the output in JSON format.\n    ### Input:\n    {text}\n    ### Output: ")  # fmt: skip

LABELS = {  # detector label -> gold category (common.CATEGORY_RISK); None = not counted
    **dict.fromkeys(("firstname", "middlename", "lastname", "username", "accountname"), "PERSON_NAME"),
    "dob": "DOB", "date": "DATE", "time": "DATE", "age": "AGE",
    **dict.fromkeys(("email", "phonenumber", "url", "ip", "ipv4", "ipv6", "mac", "useragent"), "CONTACT"),
    **dict.fromkeys(("street", "city", "state", "county", "zipcode", "country", "secondaryaddress", "buildingnumber", "nearbygpscoordinate",
                     "ordinaldirection"), "LOCATION"),
    "companyname": "ORGANIZATION",
    **dict.fromkeys(("jobtitle", "jobarea", "jobtype"), "OCCUPATION"),
    **dict.fromkeys(("ssn", "pin", "accountnumber", "creditcardnumber", "creditcardcvv", "iban", "bic", "phoneimei", "imei", "vehiclevin",
                     "vehiclevrm", "maskednumber", "password", "bitcoinaddress", "litecoinaddress", "ethereumaddress"), "ID"),
    **dict.fromkeys(("sex", "gender", "height", "eyecolor", "amount", "currency", "currencyname", "currencysymbol", "currencycode",
                     "creditcardissuer", "prefix"), None),
}  # fmt: skip
PLACEHOLDER = re.compile(r"\[[A-Za-z_ ]+\]")


def chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Consecutive pieces of at most `limit` characters, cut at line breaks (a longer line at its last space)."""
    out, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit) + 1 or limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:cut])
            line = line[cut:]
        if cur and len(cur) + 1 + len(line) > limit:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        out.append(cur)
    return [c for c in out if c.strip()]


def parse(answer: str) -> dict[str, list[str]]:
    """The first {...} of the answer as {label: [values]}; JSON or a Python literal (the card's examples use quotes ')."""
    start = answer.find("{")
    if start < 0:
        if answer.strip() in ("", "{}", "None"):
            return {}
        raise ValueError("no object")
    depth = 0
    for i, ch in enumerate(answer[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            body = answer[start : i + 1]
            break
    else:
        raise ValueError("unclosed object")
    try:
        obj = json.loads(body)
    except ValueError:
        obj = ast.literal_eval(body)  # a ValueError or SyntaxError when it is neither
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    out: dict[str, list[str]] = {}
    for k, v in obj.items():
        vals = v if isinstance(v, list) else [v]
        out[str(k).lower()] = [str(x) for x in vals if isinstance(x, (str, int, float)) and str(x).strip()]
    return out


def mentions(entities: dict[str, list[str]]) -> Counter:
    """Distinct (category, value) mentions of one chunk per gold category: `Martha` listed as both first and last name
    counts once. Unmapped labels, dropped labels and placeholder values are not counted."""
    seen = {(cat, v.strip().lower()) for label, vals in entities.items() if (cat := LABELS.get(label)) for v in vals if not PLACEHOLDER.search(v)}
    return Counter(cat for cat, _ in seen)


def ollama_generate(model: str, prompt: str) -> str:
    from eval.coders import ollama

    body = {"model": model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": ollama.KEEP_ALIVE,
            "options": {"temperature": 0, "seed": 0, "num_ctx": CONTEXT, "num_predict": MAX_OUTPUT}}  # fmt: skip
    resp = ollama.request("/api/generate", body, timeout=300)
    if not isinstance(resp, dict) or not isinstance(resp.get("response"), str):
        raise ValueError("missing response")
    return resp["response"]


class Scanner:
    """Detector answers per chunk, cached on disk; counts calls and failures."""

    def __init__(self, model: str, digest: str, generate: Callable[[str, str], str] | None = None):
        self.model, self.digest, self.generate = model, digest, generate or ollama_generate
        self.calls, self.errors = 0, 0

    def chunk(self, text: str) -> dict[str, list[str]] | None:
        path = PCACHE / f"{sha1_text(self.model, self.digest, PROMPT, text)}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        try:
            ents = parse(self.generate(self.model, PROMPT.format(text=text)))
        except (OSError, ValueError, SyntaxError):
            self.errors += 1
            return None
        self.calls += 1
        if self.calls % 25 == 0:
            print(f"phi: {self.calls} new calls", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(ents, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return ents

    def count(self, records: list[dict]) -> Counter:
        total = Counter()
        for r in records:
            for f in FIELDS:
                for ch in chunks(r.get(f) or ""):
                    if (ents := self.chunk(ch)) is not None:
                        total += mentions(ents)
        return total


def summary(before: Counter, after: Counter) -> dict:
    def one(b: int, a: int) -> dict:
        return {"before": b, "after": a, "removed": None if not b else max(0.0, ratio(b - a, b))}

    cats = sorted(set(before) | set(after))
    return {"total": one(sum(before.values()), sum(after.values())), "by_category": {c: one(before[c], after[c]) for c in cats}}


def evaluate(scanner: Scanner, model: str, suite: str, limit: int = 0) -> dict | str:
    """The before/after counts of one (model, suite), or why it is skipped."""
    rdir = RUNS_DIR / model / suite
    if not (rdir / "output.jsonl").exists():
        return "no output"
    run = json.loads((rdir / "run.json").read_text(encoding="utf-8")) if (rdir / "run.json").exists() else {}
    if run.get("exit_code") != 0:
        return f"run failed (exit {run.get('exit_code')})"
    original = load_jsonl(SUITES_DIR / suite / "encounters.jsonl")[: limit or None]
    out = {r.get("encounter_id"): r for r in load_jsonl(rdir / "output.jsonl")}
    pairs = [(o, out[o["encounter_id"]]) for o in original if isinstance(out.get(o["encounter_id"]), dict)]
    errors0, calls0 = scanner.errors, scanner.calls
    before, after = scanner.count([o for o, _ in pairs]), scanner.count([a for _, a in pairs])
    return {"model": model, "suite": suite, "n_records": len(pairs), "phi_model": scanner.model, "digest": scanner.digest, **summary(before, after),
            "errors": scanner.errors - errors0, "new_calls": scanner.calls - calls0, "limit": limit}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    from eval.coders import ollama

    p = argparse.ArgumentParser(prog="python3 -I eval phi", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--models", nargs="+", help="default: every model with a run on the suite")
    p.add_argument("--suites", nargs="+", default=["aci_eval"], choices=list(S.SUITES))
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"the detector, an installed Ollama model (default {DEFAULT_MODEL})")
    p.add_argument("--limit", type=int, default=0, help="only the first N records of each suite (development)")
    a = p.parse_args(argv)
    if not ollama.available():
        print(f"phi: Ollama not reachable at {ollama.URL}", file=sys.stderr)
        return 1
    try:
        digest = ollama.identity(a.model)["digest"]
    except ollama.PredictionError as e:
        print(f"phi: {a.model}: {e.code} (ollama pull {a.model})", file=sys.stderr)
        return 1
    ollama.unload()  # alone in memory: no coder or attacker model stays loaded beside the detector
    scanner = Scanner(a.model, digest)
    failed = 0
    try:
        for s in a.suites:
            models = a.models or (sorted(d.name for d in RUNS_DIR.iterdir() if (d / s / "output.jsonl").exists()) if RUNS_DIR.is_dir() else [])
            for m in models:
                res = evaluate(scanner, m, s, a.limit)
                if isinstance(res, str):
                    print(f"{m:24s} {s:16s} skipped: {res}")
                    continue
                write_json(PHI_DIR / m / f"{s}.json", res)
                failed += res["errors"] > 0
                t = res["total"]
                print(f"{m:24s} {s:16s} before {t['before']:5d}  after {t['after']:5d}  removed {t['removed']}  errors {res['errors']}  "
                      f"new calls {res['new_calls']}", flush=True)  # fmt: skip
    finally:
        ollama.unload(a.model)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
