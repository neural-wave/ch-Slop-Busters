"""Core pipeline invariants on synthetic text (fast, rules only, no challenge data)."""

from __future__ import annotations

import pytest

from unified_harness.config import make_config
from unified_harness.detect.encoders import VRAM_BUDGET_MB, VRAM_ESTIMATE_MB, place
from unified_harness.pipeline import Anonymizer
from unified_harness.text.normalize import Doc, normalize, words_to_int


def _anon(text_dialogue: str, text_note: str = "", codes=None, preset="balanced") -> dict:
    rec = {
        "encounter_id": "T1",
        "dataset": "test",
        "dialogue": text_dialogue,
        "note": text_note,
        "icd10_codes": codes or [],
    }
    out, _, _ = Anonymizer(make_config(preset)).anonymize(rec)
    return out


def test_normalization_is_one_to_one():
    s = "Inspection reveals ’x’ “y” a​b – m²"
    assert len(normalize(s)) == len(s)
    d = Doc("note", s)
    assert len(d.view) == len(s) and len(d.lower) == len(s)


@pytest.mark.parametrize(
    "text",
    [
        "bp 140/90",
        "hemoglobin a1c 8.3",
        "4/6 systolic murmur",
        "aspirin 81 mg",
        "ejection fraction thirty five percent",
        "pain 6/10",
        "250/50 dose",
    ],
)
def test_clinical_numbers_untouched(text):
    out = _anon(f"[doctor] okay so your {text} today .")
    assert text in out["dialogue"]


@pytest.mark.parametrize(
    "dialogue,gone,kept",
    [
        (
            "[doctor] hello mr . parkinson , how is the parkinson's disease ?",
            ["mr . parkinson"],
            ["parkinson's disease"],
        ),
        ("[doctor] your lyme titer is positive , ms . lyme .", ["ms . lyme"], ["lyme titer"]),
        ("[doctor] doctor white said your white blood cell count is fine .", ["doctor white"], ["white blood cell"]),
        (
            "[doctor] next patient is grace hill . date of birth 3/4/1956 . mrn 5-1-7-2-9-4 .",
            ["grace", "hill", "3/4/1956", "5-1-7-2-9-4"],
            [],
        ),
    ],
)
def test_eponyms_and_names(dialogue, gone, kept):
    out = _anon(dialogue)["dialogue"].lower()
    for g in gone:
        assert g not in out
    for k in kept:
        assert k in out


def test_code_evidence_kept_with_codes():
    codes = [
        {"code": "Y93.53", "description": "Activity, golf"},
        {"code": "S39.012A", "description": "Strain of muscle, fascia and tendon of lower back, initial encounter"},
    ]
    out = _anon("[patient] i was golfing and hurt my back .", codes=codes, preset="aggressive")
    assert "golfing" in out["dialogue"]


def test_ages_banded_never_exact():
    out = _anon("[doctor] so felix is a 56 -year-old male and you are 100 .", "Felix Brandt is a 56-year-old male.")
    assert "56" not in out["dialogue"] + out["note"] and "100" not in out["dialogue"]
    assert "[50s]" in out["note"] and "[90+]" in out["dialogue"]


def test_spelled_numbers():
    assert words_to_int("nineteen fifty seven") == 1957
    assert words_to_int("one twenty") == 120
    assert words_to_int("seventy four") == 74


def test_vram_placement_respects_budget():
    for dets in [("rules", "deid"), ("rules", "deid", "gliner"), ("rules", "deid", "gliner", "llm")]:
        placed = place(dets, "auto")
        total = sum(VRAM_ESTIMATE_MB[d] for d, dev in placed.items() if dev != "cpu") + (
            VRAM_ESTIMATE_MB["llm"] if "llm" in dets else 0
        )
        assert total <= VRAM_BUDGET_MB
