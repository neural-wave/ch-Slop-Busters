"""VRAM and latency probe: every runnable model on a few records, measured together with the Ollama server it uses.

Usage: python3 -I eval probe [--models ...] [--suite aci_eval] [--records 10] [--timeout 900]

`run` measures a model's own process tree over a whole suite, so a model whose LLM runs in the shared Ollama server
(zeroshot, unified_harness) shows no VRAM, and its wall time includes loading. The probe is short instead: every model
with a command runs twice, on the suite's first record and on its first N records (blinded like in `run` for a
`blind_codes` model), each time after every Ollama model is unloaded. Every 0.2 s it samples the RSS and the
nvidia-smi memory of the model's process tree and, apart, of the Ollama server's tree (every `ollama serve` and the
runners it starts), so a model that loads an LLM there pays for it and nothing else counts. Then

    latency = (wall(N) - wall(1)) / (N - 1)     the time a record adds once the model is loaded
    startup = wall(1) - latency                 process start and model loading

Writes data/eval/probe/<model>/<suite>.json (the probe's inputs, outputs and logs stay in data/eval/probe/<model>/n<k>/):

    {"model", "suite", "records", "gpu", "ollama_env", "runs": {"1": run, "<N>": run}, "latency_ms_per_encounter",
     "startup_s", "peak_ram_mb", "peak_vram_mb", "ollama_peak_ram_mb", "ollama_peak_vram_mb", "total_peak_vram_mb",
     "within_vram_budget"}

    run = {"exit_code", "wall_s", "peak_ram_mb", "peak_vram_mb", "ollama_peak_ram_mb", "ollama_peak_vram_mb",
           "total_peak_vram_mb"}

Peaks are the larger of the two runs; total = the peak of the model's and the server's VRAM summed per sample. Run it
alone on the GPU: another process's VRAM never counts, but its load slows the model down. A precomputed model has
nothing to run and is skipped; a failed run leaves the latency empty.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import run as RUN
from eval import suites as S
from eval.coders import ollama
from eval.common import ADAPTERS, CONTAINER, OUT, REPO, SUITES_DIR, load_jsonl, write_json, write_jsonl

PROBE_DIR = OUT / "probe"
KEYS = ("peak_ram_mb", "peak_vram_mb", "ollama_peak_ram_mb", "ollama_peak_vram_mb", "total_peak_vram_mb")


def ollama_pids() -> set[int]:
    """The process trees of every `ollama serve` on this machine (the server and the runners it starts)."""
    out: set[int] = set()
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/cmdline", "rb") as fh:
                    argv = fh.read().split(b"\x00")
            except OSError:
                continue
            if len(argv) > 1 and os.path.basename(argv[0]) == b"ollama" and argv[1] == b"serve":
                out |= RUN._tree(int(d))
    return out


def ollama_env() -> dict[str, str]:
    """The LLAMA_ARG_* and OLLAMA_* settings of the running server (the runners inherit them), for the record."""
    for p in sorted(ollama_pids()):
        try:
            env = Path(f"/proc/{p}/environ").read_bytes().split(b"\x00")
        except OSError:
            continue
        pairs = dict(e.decode(errors="replace").split("=", 1) for e in env if b"=" in e)
        if pairs:
            return {k: v for k, v in sorted(pairs.items()) if k.startswith(("LLAMA_ARG_", "OLLAMA_NUM_PARALLEL"))}
    return {}


def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def sample(cmd: list[str], cwd: Path, env: dict, log: Path, timeout: float, server=ollama_pids) -> dict:
    """Runs `cmd` and samples its tree and the Ollama server's tree every 0.2 s until it exits."""
    has_gpu = shutil.which("nvidia-smi") is not None
    peak = dict.fromkeys(KEYS, 0.0)
    done = threading.Event()

    def poll() -> None:
        while not done.is_set():
            pids = RUN._tree(proc.pid)
            srv = server() - pids
            vals = {"peak_ram_mb": RUN._rss_mb(pids), "ollama_peak_ram_mb": RUN._rss_mb(srv) if srv else 0.0}
            if has_gpu:
                vals["peak_vram_mb"], vals["ollama_peak_vram_mb"] = RUN._vram_mb(pids), RUN._vram_mb(srv) if srv else 0.0
                vals["total_peak_vram_mb"] = vals["peak_vram_mb"] + vals["ollama_peak_vram_mb"]
            for k, v in vals.items():
                peak[k] = max(peak[k], v)
            done.wait(0.2)

    with log.open("w", encoding="utf-8") as fh:
        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            fh.write(f"[probe] cannot start {cmd[0]!r}: {e}\n")
            return {"exit_code": 127, "wall_s": None, **dict.fromkeys(KEYS)}
        th = threading.Thread(target=poll, daemon=True)
        th.start()
        try:
            rc = proc.wait(timeout)
        except subprocess.TimeoutExpired:
            RUN._kill_tree(proc.pid)
            proc.wait()
            rc = RUN.TIMEOUT_EXIT
        except BaseException:
            RUN._kill_tree(proc.pid)
            raise
        wall = time.perf_counter() - t0
        done.set()
        th.join()
    return {"exit_code": rc, "wall_s": round(wall, 2), **{k: round(v, 1) for k, v in peak.items()}}


def summary(model: str, suite: str, n: int, runs: dict[str, dict]) -> dict:
    one, many = runs.get("1"), runs.get(str(n))
    ok = all(r and r["exit_code"] == 0 for r in (one, many))
    lat = (many["wall_s"] - one["wall_s"]) / (n - 1) if ok and n > 1 and many["wall_s"] > one["wall_s"] else None
    peaks = {k: max((r[k] for r in runs.values() if r and r.get(k) is not None), default=None) for k in KEYS}
    total = peaks["total_peak_vram_mb"]
    return {"model": model, "suite": suite, "records": n, "gpu": gpu_name(), "ollama_env": ollama_env(), "runs": runs,
            "latency_ms_per_encounter": round(1000 * lat, 1) if lat is not None else None,
            "startup_s": round(one["wall_s"] - lat, 2) if lat is not None else None, **peaks,
            "within_vram_budget": None if total is None else total <= RUN.VRAM_BUDGET_MB}  # fmt: skip


def probe(model: str, spec: dict, suite: str, n: int, timeout: float) -> dict:
    recs = load_jsonl(SUITES_DIR / suite / "encounters.jsonl")[:n]
    if spec.get("blind_codes"):
        recs = [{**r, "icd10_codes": []} for r in recs]
    n = len(recs)
    base = PROBE_DIR / model
    runs = {}
    for k in sorted({1, n}):
        d = base / f"n{k}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        write_jsonl(d / "input.jsonl", recs[:k])
        ctx = {"input": str(d / "input.jsonl"), "output": str(d / "output.jsonl"), "labels": str(SUITES_DIR / suite / "labels.jsonl"), "suite": suite,
               "python": sys.executable, "here": str(ADAPTERS), "repo": str(REPO), "container": str(CONTAINER)}  # fmt: skip
        env = {**os.environ, **{key: RUN._fmt(str(v), ctx) for key, v in spec.get("env", {}).items()}}
        ollama.unload()  # the run loads its own Ollama model, if any: its load and memory are its own
        r = sample([RUN._fmt(c, ctx) for c in spec["cmd"]], Path(RUN._fmt(spec.get("cwd", "{repo}"), ctx)), env, d / "run.log", timeout)
        if r["exit_code"] == 0 and not (d / "output.jsonl").exists():
            r["exit_code"] = 1  # exited cleanly without an output
        runs[str(k)] = r
    res = summary(model, suite, n, runs)
    write_json(base / f"{suite}.json", res)
    return res


def main(argv: list[str] | None = None) -> int:
    models = RUN.load_models()
    runnable = [m for m, s in models.items() if s.get("enabled", True) and "cmd" in s]
    p = argparse.ArgumentParser(prog="python3 -I eval probe", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--models", nargs="+", default=runnable, choices=runnable)
    p.add_argument("--suite", default="aci_eval", choices=list(S.SUITES))
    p.add_argument("--records", type=int, default=10)
    p.add_argument("--timeout", type=float, default=900, help="seconds per run before its process tree is killed")
    a = p.parse_args(argv)
    S.ensure([a.suite])
    failed = 0
    for m in a.models:
        r = probe(m, models[m], a.suite, a.records, a.timeout)
        bad = [k for k, run in r["runs"].items() if run["exit_code"] != 0]
        failed += bool(bad)
        print(f"{m:26s} VRAM {r['peak_vram_mb']} + Ollama {r['ollama_peak_vram_mb']} = {r['total_peak_vram_mb']} MB  "
              f"RAM {r['peak_ram_mb']} + Ollama {r['ollama_peak_ram_mb']} MB  latency {r['latency_ms_per_encounter']} ms/enc  "
              f"startup {r['startup_s']} s" + (f"  FAILED runs {bad} (see {PROBE_DIR / m})" if bad else ""), flush=True)  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
