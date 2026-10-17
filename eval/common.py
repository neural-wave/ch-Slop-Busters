"""Shared helpers for the evaluation pipeline: paths, JSONL, text matching and tokenization.

Standard library only. Everything the pipeline writes goes under `data/eval/` (gitignored: anonymized copies of
datasets whose licenses forbid re-sharing, and leak lists that quote PII).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent  # the eval package
REPO = HERE.parent
CONTAINER = REPO.parent  # the folder that also holds the model checkouts (models/, vm-backup/, ...)
ADAPTERS = HERE / "adapters"
DATA = REPO / "data"
CONVERTED = DATA / "converted"
OUT = DATA / "eval"
SUITES_DIR = OUT / "suites"
RUNS_DIR = OUT / "runs"
SCORES_DIR = OUT / "scores"
CACHE_DIR = OUT / "cache"

FIELDS = ("dialogue", "note")
ENCOUNTER_KEYS = ("encounter_id", "dataset", "dialogue", "note", "icd10_codes")


def ollama_url(host: str | None) -> str:
    """OLLAMA_HOST as ollama reads it (envconfig.Host): `host`, `host:port`, `:port`, an IPv6 address bare or in
    brackets, or a URL. An empty host is 127.0.0.1; a missing or invalid port is 11434 (80/443 in an http/https URL)."""
    scheme, sep, rest = (host or "").strip().partition("://")
    scheme, rest, port = (scheme, rest, {"http": "80", "https": "443"}.get(scheme, "11434")) if sep else ("http", scheme, "11434")
    hostport, _, path = rest.partition("/")
    m = re.fullmatch(r"\[([^][]*)\]:([^][:]*)|([^][:]*):([^][:]*)", hostport)  # net.SplitHostPort
    name, p = [g for g in m.groups() if g is not None] if m else (hostport.strip("[]"), port)
    port = p if p.isascii() and p.isdigit() and int(p) <= 65535 else port
    name = name or "127.0.0.1"
    path = path.strip("/")
    return f"{scheme}://{f'[{name}]' if ':' in name else name}:{port}" + (f"/{path}" if path else "")


OLLAMA_URL = ollama_url(os.environ.get("OLLAMA_HOST"))

# Same categories and default risk as scripts/convert/common.py, so every gold source scores together, plus CONTACT
# (phones, e-mails, URLs) for the merged ACI gold (eval/gold/taxonomy.py).
CATEGORY_RISK = {
    "PATIENT_NAME": "high", "PERSON_NAME": "high", "DOB": "high", "ID": "high", "DATE": "high", "AGE": "high", "CONTACT": "high",
    "YEAR": "medium", "LOCATION": "medium", "ORGANIZATION": "medium", "OCCUPATION": "medium", "OTHER_QI": "low",
}  # fmt: skip


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def sha1_text(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()


@cache
def pattern(text: str) -> re.Pattern:
    """Case-insensitive, whole-word match of `text` (the evaluation harness's rule, so its numbers stay comparable).

    Numbers get a stricter boundary: `59` must not match inside `120/59`, `5.9` or `1,590`.
    """
    left = r"(?<!\w)" if text[:1].isalnum() or text[:1] == "_" else ""
    right = r"(?!\w)" if text[-1:].isalnum() or text[-1:] == "_" else ""
    if text[:1].isdigit():
        left = r"(?<![\w/.,:])"
    if text[-1:].isdigit():
        right = r"(?![\w/:]|[.,]\d)"
    return re.compile(left + re.escape(text) + right, re.IGNORECASE)


def count(text: str, needle: str) -> int:
    return len(pattern(needle).findall(text)) if needle.strip() else 0


def occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern(needle).finditer(text)] if needle.strip() else []


WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")


def words(text: str) -> list[tuple[str, int, int]]:
    """Word tokens as (lowercased token, start, end)."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in WORD.finditer(text)]


def distinct_items(items: Iterable[dict], *keep: str) -> list[dict]:
    """One item per (field, lowercased text), longest first, carrying the `keep` keys."""
    out, seen = [], set()
    for it in sorted(items, key=lambda it: -len(it["text"])):
        key = (it["field"], it["text"].lower())
        if key not in seen and it["text"].strip():
            seen.add(key)
            out.append({"field": it["field"], "text": it["text"], **{k: it.get(k) for k in keep}})
    return out


def ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def f1(tp: int, n_pred: int, n_gold: int) -> float | None:
    return round(2 * tp / (n_pred + n_gold), 4) if n_pred + n_gold else None
