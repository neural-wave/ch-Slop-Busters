"""common: paths of the package layout and the OLLAMA_HOST normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import common as C


def test_paths():
    assert C.HERE == Path(__file__).resolve().parents[1] and C.HERE.name == "eval"
    assert C.REPO == C.HERE.parent and C.CONTAINER == C.REPO.parent
    assert C.ADAPTERS == C.HERE / "adapters" and C.OUT == C.REPO / "data" / "eval"


@pytest.mark.parametrize(("host", "url"), [
    (None, "http://127.0.0.1:11434"), ("", "http://127.0.0.1:11434"), ("0.0.0.0", "http://0.0.0.0:11434"),
    ("127.0.0.1:11500", "http://127.0.0.1:11500"), ("gpu-box", "http://gpu-box:11434"), ("https://ollama.lan/", "https://ollama.lan:443"),
    (":11434", "http://127.0.0.1:11434"), ("::1", "http://[::1]:11434"), ("[::1]", "http://[::1]:11434"), ("[::1]:11500", "http://[::1]:11500"),
    ("gpu-box:abc", "http://gpu-box:11434"), ("gpu-box:99999", "http://gpu-box:11434"), ("http://gpu-box", "http://gpu-box:80"),
    ("http://gpu-box:8080/ollama/", "http://gpu-box:8080/ollama"),
])  # fmt: skip
def test_ollama_url(host, url):
    assert C.ollama_url(host) == url
