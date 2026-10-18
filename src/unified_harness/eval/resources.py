"""Efficiency measurement: runtime, peak RAM, peak VRAM (bonus task c)."""

from __future__ import annotations

import json
import resource
import shutil
import subprocess
import sys
import urllib.request

from ..environment import ollama_url


def peak_rss_mb() -> float:
    """Peak resident set size of this process (macOS reports bytes, Linux kilobytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


RUNNER_NAMES = ("ollama", "llama-server")  # Ollama's server and its model runner processes


def nvidia_ollama_vram_mb() -> float | None:
    """GPU memory of the Ollama processes as nvidia-smi reports it (MiB), or None without an NVIDIA GPU or
    when nvidia-smi cannot see an Ollama process (for example Ollama in another container).

    On CUDA this is the real footprint: model weights plus the CUDA context, KV cache and compute buffers.
    Ollama's own /api/ps figure covers the weights and cache only and reads about 1.3-1.6 GB lower.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = []
    for line in out.splitlines():
        name, _, mem = line.rpartition(",")
        if any(r in name.lower() for r in RUNNER_NAMES):
            try:
                used.append(float(mem.strip()))
            except ValueError:
                continue
    return round(sum(used), 1) if used else None


def vram_source() -> str:
    """Which measurement `ollama_vram_mb` uses on this machine."""
    return "nvidia-smi" if nvidia_ollama_vram_mb() is not None else "ollama"


def ollama_vram_mb(model: str | None = None) -> float:
    """VRAM held by Ollama: nvidia-smi on NVIDIA GPUs (all Ollama processes, so a conservative figure for the
    4 GB guard); otherwise Ollama's own report for `model` (all loaded models when None), as on Apple
    unified memory where nvidia-smi does not exist."""
    measured = nvidia_ollama_vram_mb()
    if measured is not None:
        return measured
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{ollama_url()}/api/ps", timeout=5) as resp:
            ps = json.loads(resp.read())
    except Exception:  # noqa: BLE001
        return float("nan")  # unknown: callers must not read this as 0 MB
    tot = 0
    for m in ps.get("models", []):
        if model is None or m.get("name") == model or m.get("model") == model:
            tot += m.get("size_vram", 0)
    return round(tot / (1024 * 1024), 1)


def torch_vram_mb() -> float:
    """Peak GPU memory used by torch in this process (CUDA or Apple MPS), 0 if CPU only.

    The running peak is sampled by the detectors after every load and forward pass (models.note_vram);
    the current allocation is taken into account too.
    """
    if "torch" not in sys.modules:  # never import torch just to measure it
        return 0.0
    import torch

    peak = 0.0
    if "unified_harness.detect.encoders" in sys.modules:
        peak = sys.modules["unified_harness.detect.encoders"].note_vram()
    if torch.cuda.is_available():
        return round(max(peak, torch.cuda.max_memory_reserved() / (1024 * 1024)), 1)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            return round(max(peak, torch.mps.driver_allocated_memory() / (1024 * 1024)), 1)
        except Exception:  # noqa: BLE001
            return round(peak, 1)
    return round(peak, 1)
