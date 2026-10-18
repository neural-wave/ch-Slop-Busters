"""Adapter for unified-harness (this repo's anonymizer, src/unified_harness): run it on a suite, copy its outputs.

Usage: python3 -I eval/adapters/unified_harness.py INPUT OUTPUT [--src SRC_DIR] [anonymize options ...]

`unified-harness anonymize` writes <out-root>/<date>/anonymized-<date>-unified-harness.{jsonl,decisions.jsonl,
summary.json} and publishes nothing when a record fails. Its anonymization path is standard library only (`--mode
full` would need the `ml` extra), so this adapter imports it from SRC_DIR and runs it in-process, with
UNIFIED_HARNESS_HOME set to a fresh folder next to OUTPUT: the LLM reply cache starts cold on every run (a warm one
would make the measured cost meaningless) and the repo's outputs/ are never touched. It copies the records to OUTPUT,
the decisions to <OUTPUT stem>.decisions.jsonl (the explainability sidecar) and the summary (settings, model digests,
LLM calls and cache hits, Ollama VRAM, which the harness cannot see in the model's process tree) to
<OUTPUT stem>.summary.json, then removes the folder, whose cache holds record text. Extra arguments (--mode,
--preset, --no-codes, --features, --detector-model, ...) go to `anonymize` unchanged.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

DEFAULT_SRC = Path(__file__).resolve().parents[2] / "src"  # <repo>/eval/adapters/
WORK_PREFIX = ".unified-harness-"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    a, rest = p.parse_known_args(argv)
    sys.path.insert(0, str(a.src))
    from unified_harness import cli  # noqa: PLC0415 (importable once --src is on the path)
    from unified_harness.io import output_paths  # noqa: PLC0415

    for old in a.output.parent.glob(WORK_PREFIX + "*"):  # left by a killed run (timeout): its cache holds record text
        shutil.rmtree(old, ignore_errors=True)
    day = date.today()
    with tempfile.TemporaryDirectory(prefix=WORK_PREFIX, dir=a.output.parent) as work:
        os.environ["UNIFIED_HARNESS_HOME"] = work
        out_root = Path(work) / "outputs"
        rc = cli.main(["anonymize", "--input", str(a.input), "--out-root", str(out_root), "--date", day.isoformat(), *rest])
        out = output_paths(out_root, day)
        if rc != 0 or not out.records.exists():
            print(f"unified-harness: exit {rc}, nothing published", file=sys.stderr)
            return rc or 1
        shutil.copyfile(out.decisions, a.output.with_name(a.output.stem + ".decisions.jsonl"))
        shutil.copyfile(out.summary, a.output.with_name(a.output.stem + ".summary.json"))
        shutil.copyfile(out.records, a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
