"""Environment checks for the two run modes (`unified-harness doctor`).

`rules` needs nothing beyond the standard library. `full` needs the ML extras, the pinned encoder
weights in the models directory and the pinned LLM on a local Ollama server. A missing requirement is
reported and makes the check fail; the pipeline never falls back to a weaker mode on its own.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import tomllib
import urllib.request
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

MODES = ("rules", "llm", "full")  # llm: rules + one Ollama model, no PyTorch
ML_PACKAGES = ("torch", "transformers", "gliner")
REVISION_FILE = ".revision"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def load_manifest() -> dict:
    """The pinned model manifest shipped with the package (`models.toml`)."""
    return tomllib.loads(files("unified_harness").joinpath("models.toml").read_text(encoding="utf-8"))


def models_dir() -> Path:
    return Path(os.environ.get("UNIFIED_HARNESS_MODELS", "models"))


def ollama_url() -> str:
    return os.environ.get("UNIFIED_HARNESS_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    mode: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def as_dict(self) -> dict:
        return {"mode": self.mode, "ok": self.ok, "checks": [c.__dict__ for c in self.checks]}


def check_hf_model(entry: dict, root: Path) -> Check:
    """Every pinned file present and the revision marker equal to the pin."""
    folder = root / entry["dir"]
    missing = [f for f in entry["files"] if not (folder / f).is_file()]
    marker = folder / REVISION_FILE
    revision = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if missing:
        return Check(
            entry["repo"], False, f"missing in {folder}: {', '.join(missing)} (run scripts/download_models.py)"
        )
    if revision != entry["revision"]:
        return Check(entry["repo"], False, f"revision {revision!r} in {folder} does not match pin {entry['revision']}")
    return Check(entry["repo"], True, f"{folder} @ {entry['revision'][:12]}")


def ollama_models(url: str, timeout: float = 5.0) -> dict[str, str]:
    """{model name: digest} of a local Ollama server; raises OSError when it is unreachable."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local calls never use a proxy
    with opener.open(f"{url}/api/tags", timeout=timeout) as resp:
        tags = json.loads(resp.read())
    return {m.get("name", ""): m.get("digest", "") for m in tags.get("models", [])}


def check_ollama_model(entry: dict, available: dict[str, str] | None, url: str) -> Check:
    name = entry["name"]
    if available is None:
        return Check(name, False, f"Ollama is not reachable at {url}")
    if name not in available:
        return Check(name, False, f"not pulled on {url} (run: ollama pull {name})")
    pin = entry.get("digest", "")
    if pin and available[name] != pin:
        return Check(name, False, f"digest {available[name][:12]} does not match pin {pin[:12]}")
    return Check(name, True, f"{url} digest {available[name][:12]}")


def detector_digest(model: str) -> str | None:
    """The pinned digest of an allowed LLM detector model, or None when it is not pinned."""
    return load_manifest().get("detector_models", {}).get(model)


def doctor(
    mode: str,
    *,
    root: Path | None = None,
    url: str | None = None,
    detector_model: str | None = None,
    allow_unpinned: bool = False,
) -> Report:
    """Check everything `mode` needs. Never downloads or starts anything."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    report = Report(mode)
    report.checks.append(Check("python", True, f"{platform.python_version()} ({platform.machine()})"))
    if mode == "rules":
        return report
    manifest = load_manifest()
    if mode == "full":
        for pkg in ML_PACKAGES:
            found = importlib.util.find_spec(pkg) is not None
            report.checks.append(Check(pkg, found, "installed" if found else "missing (install the [ml] extra)"))
        root = root or models_dir()
        for entry in manifest["huggingface"].values():
            if mode in entry["used_by"]:
                report.checks.append(check_hf_model(entry, root))
    url = url or ollama_url()
    try:
        available = ollama_models(url)
    except OSError:
        available = None
    for key, entry in manifest["ollama"].items():
        if mode in entry["used_by"] and not (key == "detector" and detector_model):
            report.checks.append(check_ollama_model(entry, available, url))
    if detector_model:
        pin = detector_digest(detector_model)
        if pin is None and allow_unpinned:
            if available is None:
                report.checks.append(Check(detector_model, False, f"Ollama is not reachable at {url}"))
            elif detector_model not in available:
                report.checks.append(
                    Check(detector_model, False, f"not pulled on {url} (run: ollama pull {detector_model})")
                )
            else:
                digest = available[detector_model]
                report.checks.append(Check(detector_model, True, f"UNPINNED (experiment), digest {digest[:12]}"))
        elif pin is None:
            report.checks.append(
                Check(detector_model, False, "not pinned in models.toml [detector_models] (or pass --allow-unpinned)")
            )
        else:
            report.checks.append(check_ollama_model({"name": detector_model, "digest": pin}, available, url))
    return report
