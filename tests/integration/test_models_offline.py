"""The pinned encoder weights load with networking disabled (skipped when weights or extras are absent)."""

from __future__ import annotations

import importlib.util
import socket

import pytest

from unified_harness.environment import check_hf_model, load_manifest, models_dir

pytestmark = pytest.mark.models

MANIFEST = load_manifest()["huggingface"]


def _ready(key: str) -> bool:
    return importlib.util.find_spec("transformers") is not None and check_hf_model(MANIFEST[key], models_dir()).ok


@pytest.fixture
def offline(monkeypatch):
    """Hub offline flags plus a socket guard, so any network attempt fails the test."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def refuse(*args, **kwargs):
        raise OSError("network access attempted during an offline model load")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.mark.skipif(not _ready("deid_roberta"), reason="deid_roberta weights or [ml] extra missing")
def test_deid_roberta_loads_offline(offline):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    path = models_dir() / MANIFEST["deid_roberta"]["dir"]
    tok = AutoTokenizer.from_pretrained(path, add_prefix_space=True, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True).eval()
    enc = tok("seen by doctor varga today", return_tensors="pt")
    assert model(**enc).logits.shape[-1] == len(model.config.id2label)


@pytest.mark.skipif(
    not (_ready("gliner_pii") and importlib.util.find_spec("gliner")), reason="GLiNER weights or [ml] extra missing"
)
def test_gliner_pii_loads_offline(offline):
    from gliner import GLiNER

    path = models_dir() / MANIFEST["gliner_pii"]["dir"]
    model = GLiNER.from_pretrained(str(path), map_location="cpu", local_files_only=True)
    spans = model.predict_entities("Marta Lindqvist works as a mail carrier in Oakridge.", ["person", "location"])
    assert any(s["label"] == "person" for s in spans)


@pytest.mark.skipif(
    not (_ready("deid_roberta") and _ready("gliner_pii") and importlib.util.find_spec("gliner")),
    reason="encoder weights or [ml] extra missing",
)
def test_pipeline_loaders_use_the_pinned_local_models(offline):
    from unified_harness.config import make_config
    from unified_harness.detect.encoders import load

    cfg = make_config("balanced", detectors=("rules", "deid", "gliner"), device="cpu")
    for name in ("deid", "gliner"):
        assert load(name, cfg) is not None
