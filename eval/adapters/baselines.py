"""Reference "model" that calibrates the metrics.

Usage: python3 -I eval/adapters/baselines.py identity INPUT OUTPUT

identity  copies the input: privacy must come out 0 and every utility metric 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:  # run as a script: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import load_jsonl, write_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["identity"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args(argv)
    write_jsonl(a.output, load_jsonl(a.input))
    return 0


if __name__ == "__main__":
    sys.exit(main())
