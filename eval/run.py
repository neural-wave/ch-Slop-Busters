"""Run models on suites and record their cost: one fresh process per (model, suite).

Usage: python3 -I eval run [--models ad_rules fable_rules ...] [--suites aci_eval ...] [--force]

Defaults: every enabled model of models.toml on every suite. Writes `data/eval/runs/<model>/<suite>/`: output.jsonl
(the anonymized records), run.json (command, code fingerprint, exit code, wall time, ms/encounter, peak RAM of the
whole process tree, peak VRAM of its GPU processes) and run.log (stdout + stderr). A run is reused when run.json
shows the same input sha1, command, code_sha1 and exit code 0 and output.jsonl exists (`--force` reruns). code_sha1
hashes the command and the files of the model's `fingerprint` (models.toml), so a changed model reruns even when its
command did not change. While the model runs, run.json has `exit_code: null`: an interrupted run counts as failed.
A model may also write output.decisions.jsonl (the explainability sidecar, see bonus/explain.py); a precomputed
model's `<stem>.decisions.jsonl` beside its file is copied along.

A run that exceeds the model's `timeout` (default 3600 s) is killed with its whole process tree and recorded with
exit code 124 and `timed_out: true`. The model runs in its own session, so the harness kills its tree itself on Ctrl-C,
SIGTERM and SIGHUP too. Two cases escape: a descendant that daemonized (double fork, `setsid -f`) is no longer in the
tree, and a SIGKILLed harness kills nothing. A command that cannot start (missing executable or cwd) is recorded with
exit code 127.

Wall time runs from the start to the exit of the model's process. RAM is the peak of the summed RSS of the process
and all its children, sampled every 0.2 s, and at least the kernel's ru_maxrss. VRAM is the peak of the summed
`nvidia-smi` usage of the tree's processes, sampled every 0.5 s, so it includes the CUDA context. The challenge
budget is 4 GB at inference. Before the first run, every model loaded in the Ollama server (e.g. the LLM ICD coder of
`score`) is unloaded, so it does not hold memory while the models are measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import suites as S
from eval.coders import ollama
from eval.common import ADAPTERS, CONTAINER, HERE, REPO, RUNS_DIR, SUITES_DIR, load_jsonl, sha1_file, write_json, write_jsonl

VRAM_BUDGET_MB = 4096
DEFAULT_TIMEOUT_S = 3600
TIMEOUT_EXIT = 124  # the exit code of coreutils `timeout`
GLOB = re.compile(r"[*?[]")


def load_models(path: Path = HERE / "models.toml") -> dict[str, dict]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _fmt(s: str, ctx: dict) -> str:
    return s.format(**ctx)


def code_sha1(cmd: list[str], fingerprint: str | list[str] | None, ctx: dict, cwd: Path | None = None) -> str:
    """sha1 of the resolved command and of the files named by `fingerprint` (paths, directories or globs, with the
    command's placeholders, relative ones taken from `cwd`), each hashed with its path relative to the pattern's fixed
    prefix, in sorted path order."""
    h = hashlib.sha1(json.dumps(cmd).encode("utf-8"))
    files: dict[Path, str] = {}
    for pat in [fingerprint] if isinstance(fingerprint, str) else fingerprint or []:
        parts = (cwd or Path()).joinpath(_fmt(pat, ctx)).parts
        n = next((i for i, part in enumerate(parts) if GLOB.search(part)), len(parts))
        base = Path(*parts[:n])
        if n < len(parts):
            found = base.glob(str(Path(*parts[n:])))
        elif base.is_dir():
            found = (p for p in base.rglob("*") if not {"__pycache__", ".git"} & set(p.relative_to(base).parts))
        else:
            found, base = [base], base.parent
        hits = [p for p in found if p.is_file()]
        if not hits:
            print(f"warning: fingerprint {pat!r} matches no file", file=sys.stderr)
        files.update((p, str(p.relative_to(base))) for p in hits)
    for p in sorted(files):
        h.update(b"\x00" + files[p].encode("utf-8") + b"\x00" + p.read_bytes())
    return h.hexdigest()


def _tree(root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat", encoding="utf-8") as fh:
                    ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = set(), [root]
    while todo:
        p = todo.pop()
        out.add(p)
        todo += children.get(p, [])
    return out


def _rss_mb(pids: set[int]) -> float:
    total = 0
    for p in pids:
        try:
            with open(f"/proc/{p}/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except OSError:
            pass
    return total / 1024


def _vram_mb(pids: set[int]) -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    total = 0.0
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids:
            try:
                total += float(parts[1])
            except ValueError:
                pass
    return total


def _kill_tree(pid: int, walk: bool = True) -> None:
    """SIGKILL the process group `measured` started (`pid` leads it) and, with `walk`, every process under the live
    `pid`, so a child that left the group with setsid dies too."""
    pids = _tree(pid) if walk else set()
    for kill, target in [(os.killpg, pid)] + [(os.kill, p) for p in pids]:
        try:
            kill(target, signal.SIGKILL)
        except OSError:
            pass


def _interrupted(signum: int, frame) -> None:
    raise SystemExit(128 + signum)  # unwinds measured(), which kills the model's tree on the way out


def measured(cmd: list[str], cwd: Path, env: dict, log: Path, timeout: float = DEFAULT_TIMEOUT_S) -> dict:
    has_gpu = shutil.which("nvidia-smi") is not None
    peak = {"ram": 0.0, "vram": 0.0}
    done, fired, exited, lock = threading.Event(), threading.Event(), threading.Event(), threading.Lock()

    def poll() -> None:
        last_gpu = 0.0
        while not done.is_set():
            pids = _tree(proc.pid)
            peak["ram"] = max(peak["ram"], _rss_mb(pids))
            if has_gpu and time.perf_counter() - last_gpu >= 0.5:
                peak["vram"] = max(peak["vram"], _vram_mb(pids))
                last_gpu = time.perf_counter()
            done.wait(0.2)

    def expire() -> None:
        with lock:
            if not exited.is_set():
                fired.set()
                _kill_tree(proc.pid)

    th, timer = threading.Thread(target=poll, daemon=True), threading.Timer(timeout, expire)
    main = threading.current_thread() is threading.main_thread()
    handlers = {s: h for s in (signal.SIGTERM, signal.SIGHUP) if main and (h := signal.getsignal(s)) != signal.SIG_IGN}  # nohup stays nohup
    for s in handlers:
        signal.signal(s, _interrupted)
    try:
        with log.open("w", encoding="utf-8") as fh:
            t0 = time.perf_counter()
            try:
                proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as e:
                fh.write(f"[eval] cannot start {cmd[0]!r} in {str(cwd)!r}: {e}\n")
                return {"exit_code": 126 if isinstance(e, PermissionError) else 127, "timed_out": False, "wall_s": None, "peak_ram_mb": None,
                        "peak_vram_mb": None}  # fmt: skip
            try:
                th.start()
                timer.start()
                os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)  # exited, not reaped yet: expire() never signals a reused pid
                t1 = time.perf_counter()
                with lock:
                    exited.set()
                _, status, usage = os.wait4(proc.pid, 0)
            except BaseException:  # Ctrl-C, SIGTERM, SIGHUP: the child runs in its own session and would not get the signal
                _kill_tree(proc.pid)
                raise
            finally:
                timer.cancel()
                done.set()
                if th.is_alive():
                    th.join()
            # killed by our SIGKILL, not a run that exited on its own right at the deadline
            timed_out = fired.is_set() and os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
            if fired.is_set():
                _kill_tree(proc.pid, walk=False)  # group members forked while the tree was being killed; the reaped pid is not walked
            if timed_out:
                fh.write(f"\n[eval] timed out after {timeout} s, process tree killed\n")
    finally:
        for s, h in handlers.items():
            signal.signal(s, signal.SIG_DFL if h is None else h)
    rc = TIMEOUT_EXIT if timed_out else os.waitstatus_to_exitcode(status)
    return {"exit_code": rc, "timed_out": timed_out, "wall_s": round(t1 - t0, 2), "peak_ram_mb": round(max(peak["ram"], usage.ru_maxrss / 1024), 1),
            "peak_vram_mb": round(peak["vram"], 1)}  # fmt: skip


def run_one(model: str, spec: dict, suite: str, force: bool = False) -> dict:
    sdir = SUITES_DIR / suite
    out_dir = RUNS_DIR / model / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path, output = out_dir / "run.json", out_dir / "output.jsonl"
    inp = sdir / "encounters.jsonl"
    in_sha = sha1_file(inp)
    ctx = {"input": str(inp), "output": str(output), "labels": str(sdir / "labels.jsonl"), "suite": suite,
           "python": sys.executable, "here": str(ADAPTERS), "repo": str(REPO), "container": str(CONTAINER)}  # fmt: skip
    if spec.get("blind_codes"):
        blind = out_dir / "input_nocodes.jsonl"
        write_jsonl(blind, ({**r, "icd10_codes": []} for r in load_jsonl(inp)))
        ctx["input"] = str(blind)
    if "precomputed" in spec:
        src = Path(_fmt(spec["precomputed"], ctx))
        side = src.with_name(src.stem + ".decisions.jsonl")  # the explainability sidecar, copied along when it exists
        cmd = ["(precomputed)", str(src), sha1_file(src) if src.exists() else "missing"]  # a new delivery invalidates the cache
        cmd += [str(side), sha1_file(side)] if side.exists() else []
    else:
        cmd = [_fmt(c, ctx) for c in spec["cmd"]]
    cwd = Path(_fmt(spec.get("cwd", "{repo}"), ctx))
    code = code_sha1(cmd, spec.get("fingerprint"), ctx, cwd)
    if not force and info_path.exists():
        old = json.loads(info_path.read_text(encoding="utf-8"))
        if old.get("input_sha1") == in_sha and old.get("cmd") == cmd and old.get("code_sha1") == code and old.get("exit_code") == 0 and output.exists():
            return old
    n = sum(1 for _ in inp.open(encoding="utf-8"))
    info = {"model": model, "suite": suite, "description": spec.get("description", ""), "cmd": cmd, "code_sha1": code,
            "blind_codes": bool(spec.get("blind_codes")), "input_sha1": in_sha, "n_records": n}  # fmt: skip
    output.unlink(missing_ok=True)
    (out_dir / "output.decisions.jsonl").unlink(missing_ok=True)  # a model that stopped writing its sidecar leaves no stale one
    write_json(info_path, {**info, "exit_code": None})  # until the run ends: an interrupted run is never taken for a finished one
    if "precomputed" in spec:
        res = {"exit_code": 0 if src.exists() else 127, "timed_out": False, "wall_s": None, "peak_ram_mb": None, "peak_vram_mb": None}
        if src.exists():
            shutil.copyfile(src, output)
            if side.exists():
                shutil.copyfile(side, out_dir / "output.decisions.jsonl")
        (out_dir / "run.log").write_text(f"precomputed output {src} {'copied' if src.exists() else 'not found'}\n", encoding="utf-8")
    else:
        env = {**os.environ, **{k: _fmt(str(v), ctx) for k, v in spec.get("env", {}).items()}}
        res = measured(cmd, cwd, env, out_dir / "run.log", spec.get("timeout", DEFAULT_TIMEOUT_S))
    if spec.get("blind_codes") and res["exit_code"] == 0 and output.exists():
        codes = {r["encounter_id"]: r["icd10_codes"] for r in load_jsonl(inp)}
        write_jsonl(output, ({**r, "icd10_codes": codes.get(r.get("encounter_id"), r.get("icd10_codes"))} for r in load_jsonl(output)))
    info.update(res, ms_per_encounter=round(1000 * res["wall_s"] / n, 1) if res["wall_s"] and n else None,
                within_vram_budget=None if res["peak_vram_mb"] is None else res["peak_vram_mb"] <= VRAM_BUDGET_MB,
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))  # fmt: skip
    write_json(info_path, info)
    return info


def main(argv: list[str] | None = None) -> int:
    models = load_models()
    p = argparse.ArgumentParser(prog="python3 -I eval run", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=[m for m, s in models.items() if s.get("enabled", True)], choices=list(models))
    p.add_argument("--suites", nargs="+", default=list(S.SUITES), choices=list(S.SUITES))
    p.add_argument("--force", action="store_true", help="rerun even when a cached run matches")
    a = p.parse_args(argv)
    ollama.unload()  # an ICD coder model left loaded by `score` would hold (V)RAM while the models are measured; no-op without a server
    built = S.ensure(a.suites)
    failed = 0
    for m in a.models:
        if not models[m].get("enabled", True):
            print(f"{m}: disabled in models.toml, skipped")
            continue
        for s in a.suites:
            if s not in built or ("suites" in models[m] and s not in models[m]["suites"]):
                continue
            info = run_one(m, models[m], s, a.force)
            ok = info["exit_code"] == 0
            failed += not ok
            print(f"{m:18s} {s:15s} {'ok ' if ok else 'TIMEOUT' if info.get('timed_out') else 'FAIL'} rc={info['exit_code']} {info['wall_s']}s "
                  f"{info['ms_per_encounter']} ms/enc  RAM {info['peak_ram_mb']} MB  VRAM {info['peak_vram_mb']} MB")
            if not ok:
                print(f"    see {RUNS_DIR / m / s / 'run.log'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
