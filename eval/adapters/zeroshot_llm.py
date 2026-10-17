"""Adapter for the zero-shot LLM baseline (<container>/zeroshot/anonymize.py): run it, check it finished, copy its output.

Usage: python3 -I eval/adapters/zeroshot_llm.py INPUT OUTPUT [--root ZEROSHOT_DIR] [anonymize.py options ...]

anonymize.py asks a local LLM (Ollama) for the PII spans of each chunk as JSON, zero-shot (task definition and label
set, no demonstrations), and replaces the spans with [TYPE]; text outside the spans is untouched. It writes
<out>/anonymized.jsonl; this adapter runs it with --out next to OUTPUT (<OUTPUT stem>.zeroshot/, kept so an
interrupted run resumes) and copies the file to OUTPUT only when every record is there. A record it failed on would
otherwise be missing, so the adapter exits 1 instead. Extra arguments (--mode, --model, --workers, ...) go to
anonymize.py unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "zeroshot"  # <container>/neuralwave/eval/adapters/


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a, rest = p.parse_known_args(argv)
    work = a.output.with_name(a.output.stem + ".zeroshot")
    cmd = [sys.executable, "-I", str(a.root / "anonymize.py"), "--input", str(a.input), "--out", str(work), *rest]
    rc = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL).returncode
    meta = json.loads((work / "anonymize_meta.json").read_text(encoding="utf-8")) if (work / "anonymize_meta.json").exists() else {}
    if rc != 0 or meta.get("n_missing", 1) != 0:
        print(f"zeroshot: exit {rc}, {meta.get('n_missing', '?')} records missing; rerun to retry them", file=sys.stderr)
        return rc or 1
    shutil.copyfile(work / "anonymized.jsonl", a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
