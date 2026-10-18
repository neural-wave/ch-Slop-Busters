from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import unified_harness
from unified_harness.environment import REVISION_FILE, check_hf_model, check_ollama_model, doctor, load_manifest

ROOT = Path(__file__).resolve().parents[2]
UNREACHABLE = "http://127.0.0.1:9"  # discard port: connection refused, no network involved


def test_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == unified_harness.__version__


def test_manifest_pins_exact_revisions():
    manifest = load_manifest()
    for entry in manifest["huggingface"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry["repo"]
        assert entry["files"]
    assert manifest["ollama"]["detector"]["digest"]


def test_rules_mode_needs_nothing(tmp_path):
    report = doctor("rules", root=tmp_path, url=UNREACHABLE)
    assert report.ok


def test_full_mode_fails_clearly_without_models_or_ollama(tmp_path):
    report = doctor("full", root=tmp_path, url=UNREACHABLE)
    assert not report.ok
    details = " | ".join(c.detail for c in report.checks if not c.ok)
    assert "run scripts/download_models.py" in details
    assert "Ollama is not reachable" in details


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        doctor("regex-only")


def test_hf_model_check_requires_the_pinned_revision(tmp_path):
    entry = load_manifest()["huggingface"]["deid_roberta"]
    folder = tmp_path / entry["dir"]
    folder.mkdir()
    for f in entry["files"]:
        (folder / f).write_text("x")
    (folder / REVISION_FILE).write_text("0" * 40)
    assert not check_hf_model(entry, tmp_path).ok
    (folder / REVISION_FILE).write_text(entry["revision"])
    assert check_hf_model(entry, tmp_path).ok


def test_ollama_check_compares_digests():
    entry = {"name": "m:tag", "digest": "abc"}
    assert check_ollama_model(entry, {"m:tag": "abc"}, "u").ok
    assert not check_ollama_model(entry, {"m:tag": "def"}, "u").ok
    assert not check_ollama_model(entry, {}, "u").ok


def _fake_ollama(models: dict, pull_lines: list[dict]):
    """A local stand-in for the Ollama API: /api/tags and a streaming /api/pull."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps({"models": [{"name": k, "digest": v} for k, v in models.items()]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.end_headers()
            for line in pull_lines:
                self.wfile.write((json.dumps(line) + "\n").encode())
            if not any("error" in line for line in pull_lines):
                models["new:1b"] = "a" * 64

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_missing_model_is_downloaded_with_progress():
    from unified_harness.environment import ensure_model

    server, url = _fake_ollama(
        {},
        [{"status": "pulling manifest"}, {"status": "pulling", "total": 100, "completed": 100}, {"status": "success"}],
    )
    messages = []
    try:
        assert ensure_model("new:1b", url, messages.append) is True
        assert ensure_model("new:1b", url, messages.append) is False  # now present: no second download
    finally:
        server.shutdown()
    assert any("100% of" in m for m in messages) and messages[-1] == "new:1b downloaded"


def test_failed_download_explains_what_to_do():
    from unified_harness.environment import ensure_model

    server, url = _fake_ollama({}, [{"error": "pull model manifest: network unreachable"}])
    try:
        with pytest.raises(RuntimeError, match="DETECTOR_MODEL"):
            ensure_model("new:1b", url, lambda m: None)
    finally:
        server.shutdown()
