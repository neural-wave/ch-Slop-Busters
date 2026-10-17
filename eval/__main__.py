"""Command line of the evaluation harness: one command per stage, and `all` for the whole pipeline.

Usage: python3 -I eval <command> [args]   (from the repo root; also `python3 -m eval <command>`)
       python3 -I eval <command> --help   the options of one command

Each command hands its remaining arguments to the main() of its module, imported only when it is used. A command
whose module does not exist yet prints "not implemented yet" and exits 2.

`all` runs STEPS in order with the options it was given: `--models`, `--suites`, `--icd`, `--force` (defaults:
every enabled model, every suite). A step gets only the options it declares; an optional step whose module does
not exist yet is skipped. Its `suites` step builds only missing or stale suites (`--if-stale`). A failing step,
or one that raises, does not stop the later ones, and `all` then exits 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0] or ".").resolve() == PKG:
    sys.path[0] = str(PKG.parent)  # `python3 -I eval` puts eval/ first; modules import absolutely (`from eval import ...`)
elif str(PKG.parent) not in sys.path:
    sys.path.insert(0, str(PKG.parent))

import argparse  # noqa: E402
import importlib  # noqa: E402
import traceback  # noqa: E402
from types import ModuleType  # noqa: E402

# (command, options of `all` it receives, optional): the order of `all`. `if_stale` is a fixed `--if-stale`
STEPS: list[tuple[str, tuple[str, ...], bool]] = [
    ("gold", (), True),
    ("suites", ("suites", "if_stale"), False),
    ("run", ("models", "suites", "force"), False),
    ("score", ("models", "suites", "icd"), False),
    ("attack", ("models", "suites"), True),
    ("drift", ("suites",), True),
    ("report", (), False),
]  # fmt: skip

# command -> (module with main(argv), one-line description)
COMMANDS: dict[str, tuple[str | None, str]] = {
    "suites": ("eval.suites", "build the suites from data/converted/ (all of them, or only stale ones with --if-stale)"),
    "run": ("eval.run", "run models on suites, measure time, RAM, VRAM (cached)"),
    "score": ("eval.score", "score finished runs: validity, privacy, utility, ICD"),
    "report": ("eval.report", "write data/eval/report.md and report.json"),
    "attack": ("eval.bonus.attack", "adversarial re-identification attacks (bonus)"),
    "robustness": ("eval.bonus.robustness", "robustness on unseen document formats (bonus)"),
    "drift": ("eval.bonus.drift", "corpus drift of every suite from ACI train"),
    "gold": ("eval.gold.merge_aci_gold", "merge the three ACI golds into data/eval/gold/"),
    "all": (None, "every step in order: " + " ".join(f"[{c}]" if opt else c for c, _, opt in STEPS) + " ([ ] = optional)"),
}  # fmt: skip


def load(cmd: str) -> ModuleType | None:
    """The command's module, or None when it does not exist yet (a missing import inside it still raises)."""
    path = COMMANDS[cmd][0]
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as e:
        if e.name and (path == e.name or path.startswith(e.name + ".")):
            return None
        raise


def call(cmd: str, argv: list[str]) -> int:
    mod = load(cmd)
    if mod is None:
        print(f"{cmd}: not implemented yet", file=sys.stderr)
        return 2
    return mod.main(argv) or 0


def pipeline(argv: list[str]) -> int:
    from eval import suites as S
    from eval.run import load_models
    from eval.score import DEFAULT_ICD

    models = load_models()
    p = argparse.ArgumentParser(prog="python3 -I eval all", description=COMMANDS["all"][1])
    p.add_argument("--models", nargs="+", default=[m for m, s in models.items() if s.get("enabled", True)], choices=list(models))
    p.add_argument("--suites", nargs="+", default=list(S.SUITES), choices=list(S.SUITES))
    p.add_argument("--icd", nargs="+", default=DEFAULT_ICD, help='ICD predictors for score: knn, lexical, "cmd:<command>"')
    p.add_argument("--force", action="store_true", help="rerun models even when a cached run matches")
    a = p.parse_args(argv)
    opts = {"models": ["--models", *a.models], "suites": ["--suites", *a.suites], "icd": ["--icd", *a.icd], "force": ["--force"] if a.force else [],
            "if_stale": ["--if-stale"]}  # fmt: skip
    failed = []
    for cmd, keys, optional in STEPS:
        if load(cmd) is None:
            print(f"{cmd}: not implemented yet{', skipped' if optional else ''}", file=sys.stderr)
            if not optional:
                failed.append(cmd)
            continue
        print(f"== {cmd}", flush=True)
        try:
            rc = call(cmd, [x for k in keys for x in opts[k]])
        except Exception:  # a crashing step is a failed step; Ctrl-C and SystemExit (SIGTERM, SIGHUP) still stop `all`
            traceback.print_exc()
            rc = 1
        if rc:
            failed.append(cmd)
        sys.stdout.flush()  # keep the step's output ahead of the next step's warnings on stderr
    if failed:
        print(f"failed steps: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -I eval", description=__doc__.splitlines()[0], epilog="\n".join(__doc__.splitlines()[2:]),
                                formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=120))  # fmt: skip
    sub = p.add_subparsers(dest="command", metavar="<command>", required=True)
    for name, (_, about) in COMMANDS.items():
        sub.add_parser(name, help=about, add_help=False)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        parser().parse_args(argv[:1])  # prints the help, or the error for an unknown command, and exits
        return 2
    cmd, rest = argv[0], argv[1:]
    return pipeline(rest) if cmd == "all" else call(cmd, rest)


if __name__ == "__main__":
    sys.exit(main())
