"""Efficiency measurement: runtime, peak RAM, peak VRAM (bonus task c)."""

from __future__ import annotations

import json
import resource
import sys
import urllib.request

from ..environment import ollama_url


def peak_rss_mb() -> float:
    """Peak resident set size of this process (macOS reports bytes, Linux kilobytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def ollama_vram_mb(model: str | None = None) -> float:
    """VRAM currently held by Ollama models (all loaded models when `model` is None)."""
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
