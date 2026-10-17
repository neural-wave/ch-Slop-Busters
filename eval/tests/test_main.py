"""The command line: every command is listed, a command whose module does not exist yet exits 2, a crashing step does
not stop `all`, and the stage modules also run as files."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CLI = importlib.import_module("eval.__main__")


def test_help_lists_every_command():
    out = subprocess.run([sys.executable, "-I", "eval", "--help"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    for cmd in ("suites", "run", "score", "report", "attack", "robustness", "drift", "gold", "all"):
        assert f"    {cmd} " in out


def test_unknown_module_is_not_implemented(monkeypatch, capsys):
    monkeypatch.setitem(CLI.COMMANDS, "nope", ("eval.bonus.does_not_exist", "test"))
    assert CLI.load("nope") is None
    assert CLI.call("nope", []) == 2
    assert "nope: not implemented yet" in capsys.readouterr().err


def test_every_step_is_a_command():
    assert all(cmd in CLI.COMMANDS for cmd, _, _ in CLI.STEPS)
    assert [c for c, _, optional in CLI.STEPS if not optional] == ["suites", "run", "score", "report"]


def step(monkeypatch, name: str, main) -> None:
    mod = types.ModuleType(f"eval.{name}")
    mod.main = main
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(CLI.COMMANDS, name, (mod.__name__, "test"))


def test_a_crashing_step_does_not_stop_all(monkeypatch, capsys):
    ran = []
    step(monkeypatch, "boom", lambda argv: 1 / 0)
    step(monkeypatch, "after", lambda argv: ran.append(argv) or 0)
    monkeypatch.setattr(CLI, "STEPS", [("boom", (), False), ("after", ("suites", "if_stale"), False)])
    assert CLI.pipeline(["--suites", "sadiqguru"]) == 1
    assert ran == [["--suites", "sadiqguru", "--if-stale"]]
    err = capsys.readouterr().err
    assert "ZeroDivisionError" in err and "failed steps: boom" in err


@pytest.mark.parametrize("module", ["run", "score", "suites", "report"])
def test_stage_module_runs_as_a_file(module):
    out = subprocess.run([sys.executable, "-I", f"eval/{module}.py", "--help"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert out.startswith(f"usage: python3 -I eval {module}")
