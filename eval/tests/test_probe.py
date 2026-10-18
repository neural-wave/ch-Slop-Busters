"""bonus/probe.py: the latency and startup arithmetic, the Ollama server's processes counted apart from the model's,
and a probe end to end with a fake model (one-record and N-record runs, blinded input, the per-model file)."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from eval.bonus import probe as P

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process tree from /proc")


def run(wall: float, rc: int = 0, vram: float = 0.0, ovram: float = 0.0) -> dict:
    return {"exit_code": rc, "wall_s": wall, "peak_ram_mb": 90.0, "peak_vram_mb": vram, "ollama_peak_ram_mb": 0.0, "ollama_peak_vram_mb": ovram,
            "total_peak_vram_mb": vram + ovram}  # fmt: skip


def test_summary_latency_startup_and_peaks(monkeypatch):
    monkeypatch.setattr(P, "gpu_name", lambda: "Toy GPU")
    monkeypatch.setattr(P, "ollama_env", lambda: {})
    s = P.summary("m", "toy", 10, {"1": run(3.0, ovram=2850.0), "10": run(7.5, vram=100.0, ovram=2900.0)})
    assert s["latency_ms_per_encounter"] == 500.0 and s["startup_s"] == 2.5  # (7.5 - 3.0) / 9; 3.0 - 0.5
    assert s["total_peak_vram_mb"] == 3000.0 and s["ollama_peak_vram_mb"] == 2900.0 and s["within_vram_budget"] is True
    assert P.summary("m", "toy", 10, {"1": run(3.0, rc=1), "10": run(7.5)})["latency_ms_per_encounter"] is None  # a failed run
    assert P.summary("m", "toy", 10, {"1": run(3.0), "10": run(2.9)})["startup_s"] is None  # not slower than one record: noise
    assert P.summary("m", "toy", 10, {"1": run(3.0, ovram=5000.0), "10": run(7.5)})["within_vram_budget"] is False


@linux
def test_server_processes_are_measured_apart(tmp_path):
    server = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(60 * 2**20); time.sleep(60)"])
    try:
        time.sleep(0.5)
        cmd = [sys.executable, "-c", "import time; time.sleep(0.8)"]
        r = P.sample(cmd, tmp_path, {}, tmp_path / "run.log", 30, server=lambda: {server.pid})
        assert r["exit_code"] == 0 and 0.7 < r["wall_s"] < 3 and r["ollama_peak_ram_mb"] >= 60 and r["peak_ram_mb"] < r["ollama_peak_ram_mb"]
    finally:
        server.kill()
        server.wait()


@linux
def test_probe_end_to_end(tmp_path, monkeypatch):
    recs = [{"encounter_id": f"e{i}", "icd10_codes": [{"code": "I10"}], "note": "x", "dialogue": "y"} for i in range(4)]
    (tmp_path / "suites" / "toy").mkdir(parents=True)
    (tmp_path / "suites" / "toy" / "encounters.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(P, "SUITES_DIR", tmp_path / "suites")
    monkeypatch.setattr(P, "PROBE_DIR", tmp_path / "probe")
    monkeypatch.setattr(P.ollama, "unload", lambda *a, **kw: None)
    model = "import json, sys, time; recs = open(sys.argv[1]).readlines(); time.sleep(0.3 + 0.2 * len(recs)); open(sys.argv[2], 'w').writelines(recs)"
    res = P.probe("m", {"cmd": ["{python}", "-I", "-c", model, "{input}", "{output}"], "blind_codes": True}, "toy", 4, 60)
    assert set(res["runs"]) == {"1", "4"} and all(r["exit_code"] == 0 for r in res["runs"].values())
    assert abs(res["latency_ms_per_encounter"] - 200) < 80 and res["records"] == 4
    assert json.loads((tmp_path / "probe" / "m" / "n4" / "input.jsonl").read_text().splitlines()[0])["icd10_codes"] == []  # blinded like `run`
    assert json.loads((tmp_path / "probe" / "m" / "toy.json").read_text())["latency_ms_per_encounter"] == res["latency_ms_per_encounter"]
