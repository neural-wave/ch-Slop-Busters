"""Command line interface: `unified-harness <command>`.

Commands available now:
  doctor     check that a mode (rules | full) has everything it needs; exits 1 when not
  validate   check an anonymized file against its input (output contract); exits 1 on violations

The anonymize, evaluate, run, sweep, gate, attack, labels and show commands arrive with the pipeline
and the evaluation harness (see PLAN.md, milestones M1-M3).
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .environment import MODES, doctor
from .io import InputError, read_jsonl, validate_input, validate_output


def cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor(args.mode)
    if args.json:
        print(json.dumps(report.as_dict(), indent=1))
    else:
        for c in report.checks:
            print(f"[{'ok' if c.ok else 'FAIL':>4}] {c.name}: {c.detail}")
        print(f"mode {args.mode}: {'ready' if report.ok else 'NOT ready'}")
    return 0 if report.ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        original, anonymized = read_jsonl(args.original), read_jsonl(args.anonymized)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    errors = validate_input(original) + validate_output(original, anonymized)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"{len(anonymized)} records checked, {len(errors)} contract violation(s)")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified-harness", description="Offline, ICD-aware anonymization of patient-doctor encounters."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the environment for a run mode")
    p.add_argument("--mode", choices=MODES, default="full")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("validate", help="check an anonymized file against its input")
    p.add_argument("--original", required=True)
    p.add_argument("--anonymized", required=True)
    p.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
