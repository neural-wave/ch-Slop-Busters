"""Claude worker for attack A4: one Messages API request per prompt, structured JSON answers.

Usage: uv run --no-project --with anthropic python3 -I eval/bonus/_claude_worker.py --model claude-sonnet-5-5 --schema JSON IN OUT

Run by bonus/attack.py in its own environment (the official `anthropic` SDK, which the standard-library harness does
not have). IN has one {"key", "prompt"} per line; OUT gets one line per prompt, in completion order:
{"key", "guess": {...}, "model", "usage": {"input", "output"}} or {"key", "error": code}.

Each request: structured output (`output_config.format`, the schema closed with additionalProperties false), effort
low, server-side fallback "default" (beta server-side-fallback-2026-07-01). No temperature: Claude Sonnet 5.5 rejects
non-default sampling values, so answers are cached by the caller, not made reproducible here. Error codes:
`refusal:<category>` (the model declined), `max_tokens`, `invalid_json`, `api:<status>` or `api:connection` after the
SDK's retries (5, with backoff, on 408/409/429/5xx and connection errors). A failed authentication stops the run with
exit code 2 and every prompt marked `api:401`. Credentials: ANTHROPIC_API_KEY (or any source the SDK resolves).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic

MAX_TOKENS = 4096  # adaptive thinking at effort low shares this budget with the short JSON answer
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def closed(schema: dict) -> dict:
    """Structured outputs need every object closed."""
    out = dict(schema)
    if out.get("type") == "object":
        out["additionalProperties"] = False
        out["properties"] = {k: closed(v) for k, v in out.get("properties", {}).items()}
    return out


def ask(client: anthropic.Anthropic, model: str, schema: dict, prompt: str) -> dict:
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError as e:
        return {"error": f"api:{e.status_code}"}
    except anthropic.APIConnectionError:
        return {"error": "api:connection"}
    if resp.stop_reason == "refusal":
        category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        return {"error": f"refusal:{category or 'unknown'}"}
    if resp.stop_reason == "max_tokens":
        return {"error": "max_tokens"}
    text = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        guess = json.loads(text)
    except (TypeError, ValueError):
        return {"error": "invalid_json"}
    if not isinstance(guess, dict):
        return {"error": "invalid_json"}
    return {"guess": guess, "model": resp.model, "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--schema", required=True, help="JSON schema of the answer")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("input")
    p.add_argument("output")
    a = p.parse_args(argv)
    schema = closed(json.loads(a.schema))
    with open(a.input, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    client = anthropic.Anthropic(max_retries=5)
    lock = threading.Lock()
    with open(a.output, "w", encoding="utf-8") as out:

        def emit(row: dict) -> None:
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

        if not rows:
            return 0
        first = ask(client, a.model, schema, rows[0]["prompt"])  # one request first: bad credentials fail once, not N times
        emit({"key": rows[0]["key"], **first})
        if first.get("error") == "api:401":
            for r in rows[1:]:
                emit({"key": r["key"], "error": "api:401"})
            print("claude worker: authentication failed (set ANTHROPIC_API_KEY)", file=sys.stderr)
            return 2
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            for r, res in zip(rows[1:], ex.map(lambda r: ask(client, a.model, schema, r["prompt"]), rows[1:])):
                emit({"key": r["key"], **res})
    return 0


if __name__ == "__main__":
    sys.exit(main())
