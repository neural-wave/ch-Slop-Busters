"""Shared helpers: JSONL I/O, a minimal LLM client (Ollama or vLLM), chunking and text normalization.

Standard library only, so the harness runs on-prem without extra dependencies.
The backend is chosen with LLM_BACKEND=ollama (default, OLLAMA_HOST) or LLM_BACKEND=vllm (VLLM_URL,
an OpenAI-compatible `vllm serve` endpoint; see serve_vllm.sh).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "qwen-finetune"  # data/, rl/ and runs/ live in the fine-tune project
DATA_DIR = ROOT / "data"
EVAL_DATA = DATA_DIR / "eval_patient_doctor_encounters.jsonl"
GOLD_EVAL = Path(__file__).resolve().parent / "gold" / "eval_gold.jsonl"

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
if LLM_BACKEND not in ("ollama", "vllm"):
    raise SystemExit(f"LLM_BACKEND must be 'ollama' or 'vllm', got {LLM_BACKEND!r}")

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------- JSONL I/O

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlAppender:
    """Thread-safe append-only JSONL writer (used for resumable outputs and caches)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- LLM client

class LLMError(RuntimeError):
    pass


def chat(model: str, messages: list[dict], **kw) -> dict:
    """One non-streaming chat call on the configured backend (LLM_BACKEND); same arguments and result
    for both, see ollama_chat."""
    return (vllm_chat if LLM_BACKEND == "vllm" else ollama_chat)(model, messages, **kw)


def ollama_chat(model: str, messages: list[dict], *, fmt=None, num_ctx: int = 8192,
                temperature: float = 0.0, num_predict: int = -1, think: bool = False,
                timeout: float = 600, retries: int = 3, extra_options: dict | None = None) -> dict:
    """Call /api/chat (non-streaming). Returns {'content', 'prompt_tokens', 'gen_tokens', 'seconds'}.

    `fmt` may be "json" or a JSON schema (Ollama structured outputs).
    presence_penalty is pinned to 0: the qwen3.5 default (1.5) actively discourages
    copying input tokens, which is exactly what anonymization needs to do.
    """
    options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict,
               "presence_penalty": 0.0, "seed": 0}
    if extra_options:
        options.update(extra_options)
    body = {"model": model, "messages": messages, "stream": False, "think": think, "options": options}
    if fmt is not None:
        body["format"] = fmt
    data = json.dumps(body).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=data,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read())
            return {
                "content": out.get("message", {}).get("content", ""),
                "prompt_tokens": out.get("prompt_eval_count", 0),
                "gen_tokens": out.get("eval_count", 0),
                "seconds": time.time() - t0,
                "done_reason": out.get("done_reason"),
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise LLMError(f"ollama call failed after {retries} attempts: {last_err}")


def vllm_chat(model: str, messages: list[dict], *, fmt=None, num_ctx: int = 8192,
              temperature: float = 0.0, num_predict: int = -1, think: bool = False,
              timeout: float = 600, retries: int = 3, extra_options: dict | None = None) -> dict:
    """Same contract as ollama_chat, against vLLM's OpenAI-compatible /v1/chat/completions.

    `model` is the server's --served-model-name. num_ctx is fixed server-side (--max-model-len) and
    ignored here. Thinking is switched through the Qwen chat template; with think=True the server
    needs --reasoning-parser so that `content` holds only the answer.
    """
    body = {"model": model, "messages": messages, "stream": False, "temperature": temperature,
            "presence_penalty": 0.0, "seed": 0, "chat_template_kwargs": {"enable_thinking": think}}
    if num_predict > 0:
        body["max_tokens"] = num_predict
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    elif fmt is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt}}
    if extra_options:
        body.update(extra_options)
    data = json.dumps(body).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{VLLM_URL}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read())
            choice, usage = out["choices"][0], out.get("usage") or {}
            return {
                "content": choice["message"].get("content") or "",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "gen_tokens": usage.get("completion_tokens", 0),
                "seconds": time.time() - t0,
                "done_reason": choice.get("finish_reason"),
            }
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:  # bad request (e.g. prompt longer than --max-model-len): retrying won't help
                raise LLMError(f"vllm rejected the request ({e.code}): {e.read().decode(errors='replace')[:500]}")
            last_err = e
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise LLMError(f"vllm call failed after {retries} attempts: {last_err}")


def parse_json_loose(text: str):
    """Parse JSON from a model reply, tolerating ```json fences and leading/trailing chatter."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


_FLAT_OBJ = re.compile(r"\{[^{}]*\}")


def salvage_objects(text: str) -> list[dict]:
    """The complete flat JSON objects of a reply that parse_json_loose cannot read, in order.

    Small models sometimes repeat one list item until the token limit, so the reply ends mid-array
    ('{"entities": [{"text": "knee", ...}, {"text": "knee", "ty'). Every finished item is still valid.
    """
    out = []
    for m in _FLAT_OBJ.finditer(text or ""):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict):
            out.append(o)
    return out


def sha(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:24]


# ---------------------------------------------------------------- chunking

def chunk_lines(text: str, max_chars: int) -> list[str]:
    """Split text on line boundaries into chunks of at most ~max_chars.

    Joining the chunks with "\n" reproduces the original text exactly.
    A single line longer than max_chars becomes its own chunk.
    """
    lines = text.split("\n")
    chunks, cur, cur_len = [], [], 0
    for line in lines:
        add = len(line) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [line], len(line)
        else:
            cur.append(line)
            cur_len += add
    chunks.append("\n".join(cur))
    return chunks


# ---------------------------------------------------------------- normalization

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """Case/spacing-insensitive form used for all string matching in the evaluator.

    The dialogues are tokenized ("hi , martha ."), the notes are not, so we lowercase,
    treat hyphens/underscores as spaces, put spaces around punctuation and collapse whitespace.
    """
    t = text.lower().replace("’", "'")
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"([.,;:!?()\[\]\"])", r" \1 ", t)
    return _WS.sub(" ", t).strip()


def count_mentions(needle: str, haystack_norm: str) -> int:
    """Number of whole-word occurrences of `needle` in an already-normalized haystack."""
    n = norm(needle)
    if not n:
        return 0
    pat = r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])"
    return len(re.findall(pat, haystack_norm))
