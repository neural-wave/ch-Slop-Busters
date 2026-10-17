"""Reference "models" that calibrate the metrics.

Usage: python3 -I eval/adapters/baselines.py identity INPUT OUTPUT
       python3 -I eval/adapters/baselines.py oracle INPUT OUTPUT --labels LABELS

identity  copies the input: privacy must come out 0 and every utility metric 1.
oracle    replaces every whole-word occurrence of every gold PII string with [CATEGORY], longest first. It reads
          the gold, so it is an upper bound for privacy and shows what removing exactly the gold costs in utility.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if not __package__:  # run as a script: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import FIELDS, load_jsonl, pattern, write_jsonl  # noqa: E402


def oracle(rec: dict, gold: dict | None) -> dict:
    if not gold:
        return rec
    out = dict(rec)
    for f in FIELDS:
        text = rec[f]
        items = sorted((it for it in gold["pii"] if it["field"] == f), key=lambda it: -len(it["text"]))
        for it in items:
            tag = f"[{it['category']}]"
            text = pattern(it["text"]).sub(lambda m, tag=tag: tag, text)
        out[f] = re.sub(r"(\[[A-Z_]+\])([ \t]*\1)+", r"\1", text)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["identity", "oracle"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--labels", type=Path)
    a = p.parse_args(argv)
    recs = load_jsonl(a.input)
    if a.mode == "oracle":
        if not a.labels:
            p.error("oracle needs --labels")
        gold = {g["encounter_id"]: g for g in (json.loads(line) for line in a.labels.open(encoding="utf-8"))}
        recs = [oracle(r, gold.get(r["encounter_id"])) for r in recs]
    write_jsonl(a.output, recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
