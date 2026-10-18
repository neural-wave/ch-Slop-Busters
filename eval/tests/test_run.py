"""run.code_sha1 (the model fingerprint in the run cache key), and run.measured: the timeout and the harness's own
SIGTERM/SIGHUP kill the model's whole tree, a command that cannot start is a failed run, wall time ends at the exit."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from eval import run as R

REPO = Path(__file__).resolve().parents[2]
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process tree from /proc")

CMD = ["model", "{input}"]


def tree(root: Path) -> None:
    for rel, text in {"src/pkg/a.py": "A = 1\n", "src/pkg/sub/b.py": "B = 2\n", "src/pkg/data.txt": "x\n", "src/pkg/__pycache__/a.pyc": "junk"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def sha(root: Path, fingerprint, cmd=CMD) -> str:
    return R.code_sha1(cmd, fingerprint, {"root": str(root)})


def test_glob_fingerprint(tmp_path):
    tree(tmp_path)
    fp = "{root}/src/**/*.py"
    h = sha(tmp_path, fp)
    assert h == sha(tmp_path, fp) and h != sha(tmp_path, None)
    (tmp_path / "src/pkg/data.txt").write_text("changed\n")
    assert sha(tmp_path, fp) == h  # not matched by the glob
    (tmp_path / "src/pkg/sub/b.py").write_text("B = 3\n")
    assert sha(tmp_path, fp) != h


def test_rename_changes_the_fingerprint(tmp_path):
    tree(tmp_path)
    h = sha(tmp_path, "{root}/src/**/*.py")
    (tmp_path / "src/pkg/sub/b.py").rename(tmp_path / "src/pkg/sub/c.py")
    assert sha(tmp_path, "{root}/src/**/*.py") != h


def test_location_independent(tmp_path):
    """Files are hashed with paths relative to the pattern's fixed prefix: a moved checkout with the same command keeps its hash."""
    tree(tmp_path / "one")
    tree(tmp_path / "two")
    assert sha(tmp_path / "one", "{root}/src/**/*.py") == sha(tmp_path / "two", "{root}/src/**/*.py")


def test_directory_file_and_list(tmp_path):
    tree(tmp_path)
    h_dir = sha(tmp_path, "{root}/src")
    (tmp_path / "src/pkg/__pycache__/a.pyc").write_text("other junk")
    assert sha(tmp_path, "{root}/src") == h_dir  # __pycache__ is skipped
    (tmp_path / "src/pkg/data.txt").write_text("changed\n")
    assert sha(tmp_path, "{root}/src") != h_dir  # a directory hashes every file
    h_list = sha(tmp_path, ["{root}/src/pkg/a.py", "{root}/src/pkg/data.txt"])
    assert h_list != sha(tmp_path, "{root}/src/pkg/a.py")
    (tmp_path / "src/pkg/a.py").write_text("A = 2\n")
    assert sha(tmp_path, ["{root}/src/pkg/a.py", "{root}/src/pkg/data.txt"]) != h_list


def test_command_is_part_of_the_hash(tmp_path):
    tree(tmp_path)
    assert sha(tmp_path, "{root}/src", ["model", "--fast"]) != sha(tmp_path, "{root}/src", ["model", "--slow"])


def test_missing_fingerprint_warns(tmp_path, capsys):
    assert sha(tmp_path, "{root}/nothing/*.py") == sha(tmp_path, None)
    assert "matches no file" in capsys.readouterr().err


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


@linux
def test_timeout_kills_the_tree(tmp_path):
    """A child in the group and a child that left it with setsid both die; the run is recorded as exit 124."""
    pids = tmp_path / "pids"
    code = ("import subprocess, sys, time\n"
            "a = subprocess.Popen(['sleep', '30'])\n"
            "b = subprocess.Popen(['sleep', '30'], start_new_session=True)\n"
            f"open({str(pids)!r}, 'w').write(f'{{a.pid}} {{b.pid}}')\n"
            "time.sleep(30)\n")  # fmt: skip
    t0 = time.perf_counter()
    res = R.measured([sys.executable, "-c", code], tmp_path, {}, tmp_path / "run.log", timeout=1)
    assert time.perf_counter() - t0 < 10
    assert res["exit_code"] == R.TIMEOUT_EXIT == 124 and res["timed_out"] is True
    children = [int(x) for x in pids.read_text().split()]
    deadline = time.time() + 5
    while any(alive(p) for p in children) and time.time() < deadline:
        time.sleep(0.05)
    assert not any(alive(p) for p in children)
    assert "timed out after 1 s" in (tmp_path / "run.log").read_text()


@linux
def test_normal_exit(tmp_path):
    res = R.measured([sys.executable, "-c", "print('hi'); raise SystemExit(3)"], tmp_path, {}, tmp_path / "run.log", timeout=30)
    assert res["exit_code"] == 3 and res["timed_out"] is False and res["peak_ram_mb"] > 0
    assert (tmp_path / "run.log").read_text() == "hi\n"


def test_relative_fingerprint_is_taken_from_cwd(tmp_path):
    tree(tmp_path)
    assert R.code_sha1(CMD, "src/**/*.py", {}, tmp_path) == sha(tmp_path, "{root}/src/**/*.py")


def wait_dead(pid: int, seconds: float = 5) -> bool:
    deadline = time.time() + seconds
    while alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    return not alive(pid)


@linux
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_signal_to_the_harness_kills_the_model(tmp_path, sig):
    """The model runs in its own session, so a SIGTERM or SIGHUP for the harness would orphan it; the harness kills it.
    A signal the harness ignores (nohup) stays ignored."""
    pid_file = tmp_path / "pid"
    model = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    harness = (f"import signal, sys; sys.path.insert(0, {str(REPO)!r}); from pathlib import Path; from eval import run as R\n"
               f"signal.signal(signal.SIGHUP, signal.SIG_IGN) if {sig == signal.SIGTERM} else None\n"
               f"R.measured([sys.executable, '-c', {model!r}], Path({str(tmp_path)!r}), {{}}, Path({str(tmp_path / 'run.log')!r}), timeout=60)\n")  # fmt: skip
    h = subprocess.Popen([sys.executable, "-I", "-c", harness])
    deadline = time.time() + 10
    while not (pid_file.exists() and pid_file.read_text()) and time.time() < deadline:
        time.sleep(0.05)
    child = int(pid_file.read_text())
    if sig == signal.SIGTERM:  # this harness ignores SIGHUP, as under nohup
        h.send_signal(signal.SIGHUP)
        time.sleep(0.3)
        assert h.poll() is None and alive(child)
    h.send_signal(sig)
    assert h.wait(10) == 128 + sig
    assert wait_dead(child)


def test_command_that_cannot_start(tmp_path):
    res = R.measured([str(tmp_path / "no-such-model")], tmp_path, {}, tmp_path / "run.log")
    assert res == {"exit_code": 127, "timed_out": False, "wall_s": None, "peak_ram_mb": None, "peak_vram_mb": None}
    assert "cannot start" in (tmp_path / "run.log").read_text()
    assert R.measured([sys.executable, "-c", "pass"], tmp_path / "no-such-dir", {}, tmp_path / "run.log")["exit_code"] == 127


@linux
def test_exit_at_the_deadline_is_not_a_timeout(tmp_path, monkeypatch):
    """The timer fires after the model exited but before the harness noticed: the run still counts as finished."""
    real = os.waitid
    monkeypatch.setattr(R.os, "waitid", lambda *a: (real(*a), time.sleep(1.0))[0])
    res = R.measured([sys.executable, "-c", "import time; time.sleep(0.3)"], tmp_path, {}, tmp_path / "run.log", timeout=0.8)
    assert res["exit_code"] == 0 and res["timed_out"] is False


@linux
def test_wall_time_ends_when_the_model_exits(tmp_path, monkeypatch):
    """A slow nvidia-smi sample still running when the model exits is not part of the model's wall time."""
    monkeypatch.setattr(R.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(R, "_vram_mb", lambda pids: time.sleep(1.5) or 0.0)
    res = R.measured([sys.executable, "-c", "pass"], tmp_path, {}, tmp_path / "run.log", timeout=30)
    assert res["exit_code"] == 0 and res["wall_s"] < 1.0


@linux
def test_interrupted_run_is_not_cached(tmp_path, monkeypatch):
    """An interrupted rerun leaves run.json with exit_code null and no output, so the next run starts over."""
    (tmp_path / "suites" / "toy").mkdir(parents=True)
    (tmp_path / "suites" / "toy" / "encounters.jsonl").write_text('{"encounter_id": "a"}\n')
    monkeypatch.setattr(R, "SUITES_DIR", tmp_path / "suites")
    monkeypatch.setattr(R, "RUNS_DIR", tmp_path / "runs")
    spec = {"cmd": ["{python}", "-I", "-c", "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])", "{input}", "{output}"]}
    first = R.run_one("m", spec, "toy")
    assert first["exit_code"] == 0 and R.run_one("m", spec, "toy") == first
    real = R.measured

    def interrupted(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(R, "measured", interrupted)
    with pytest.raises(KeyboardInterrupt):
        R.run_one("m", spec, "toy", force=True)
    run_dir = tmp_path / "runs" / "m" / "toy"
    assert json.loads((run_dir / "run.json").read_text())["exit_code"] is None and not (run_dir / "output.jsonl").exists()
    monkeypatch.setattr(R, "measured", real)
    assert R.run_one("m", spec, "toy")["exit_code"] == 0 and (run_dir / "output.jsonl").exists()
