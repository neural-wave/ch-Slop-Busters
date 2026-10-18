"""Core invariants of the pipeline (fast, rules only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_wave.config import make_config
from neural_wave.io import read_jsonl, validate_output
from neural_wave.models import VRAM_BUDGET_MB, VRAM_ESTIMATE_MB, place
from neural_wave.pipeline import Anonymizer
from neural_wave.text import Doc, normalize, words_to_int

ROOT = Path(__file__).resolve().parents[1]
EVAL = read_jsonl(ROOT / "resources/data/eval_patient_doctor_encounters.jsonl")


def _anon(text_dialogue: str, text_note: str = "", codes=None, preset="balanced") -> dict:
    rec = {"encounter_id": "T1", "dataset": "test", "dialogue": text_dialogue, "note": text_note, "icd10_codes": codes or []}
    out, _, _ = Anonymizer(make_config(preset)).anonymize(rec)
    return out


def test_schema_and_codes_preserved_byte_for_byte():
    an = Anonymizer(make_config("aggressive"))
    outs = [an.anonymize(r)[0] for r in EVAL[:15]]
    assert validate_output(EVAL[:15], outs) == []
    for o, a in zip(EVAL[:15], outs):
        assert json.dumps(o["icd10_codes"], ensure_ascii=False) == json.dumps(a["icd10_codes"], ensure_ascii=False)
        assert list(o) == list(a)


def test_deterministic():
    an1, an2 = Anonymizer(make_config("balanced")), Anonymizer(make_config("balanced"))
    for r in EVAL[:5]:
        assert an1.anonymize(r)[0] == an2.anonymize(r)[0]


def test_normalization_is_one_to_one():
    s = "Inspection reveals ’x’ “y” a​b – m²"
    assert len(normalize(s)) == len(s)
    d = Doc("note", s)
    assert len(d.view) == len(s) and len(d.lower) == len(s)


@pytest.mark.parametrize("text", ["bp 140/90", "hemoglobin a1c 8.3", "4/6 systolic murmur", "aspirin 81 mg", "ejection fraction thirty five percent", "pain 6/10", "250/50 dose"])
def test_clinical_numbers_untouched(text):
    out = _anon(f"[doctor] okay so your {text} today .")
    assert text in out["dialogue"]


@pytest.mark.parametrize(
    "dialogue,gone,kept",
    [
        ("[doctor] hello mr . parkinson , how is the parkinson's disease ?", ["mr . parkinson"], ["parkinson's disease"]),
        ("[doctor] your lyme titer is positive , ms . lyme .", ["ms . lyme"], ["lyme titer"]),
        ("[doctor] doctor white said your white blood cell count is fine .", ["doctor white"], ["white blood cell"]),
        ("[doctor] next patient is grace hill . date of birth 3/4/1956 . mrn 3-8-4-9-2-0 .", ["grace", "hill", "3/4/1956", "3-8-4-9-2-0"], []),
    ],
)
def test_eponyms_and_names(dialogue, gone, kept):
    out = _anon(dialogue)["dialogue"].lower()
    for g in gone:
        assert g not in out
    for k in kept:
        assert k in out


def test_code_evidence_kept_with_codes():
    codes = [{"code": "Y93.53", "description": "Activity, golf"}, {"code": "S39.012A", "description": "Strain of muscle, fascia and tendon of lower back, initial encounter"}]
    out = _anon("[patient] i was golfing and hurt my back .", codes=codes, preset="aggressive")
    assert "golfing" in out["dialogue"]


def test_ages_banded_never_exact():
    out = _anon("[doctor] so tyler is a 56 -year-old male and you are 100 .", "Tyler Green is a 56-year-old male.")
    assert "56" not in out["dialogue"] + out["note"] and "100" not in out["dialogue"]
    assert "[50s]" in out["note"] and "[90+]" in out["dialogue"]


def test_spelled_numbers():
    assert words_to_int("nineteen fifty seven") == 1957
    assert words_to_int("one twenty") == 120
    assert words_to_int("seventy four") == 74


def test_vram_placement_respects_budget():
    for dets in [("rules", "deid"), ("rules", "deid", "gliner"), ("rules", "deid", "gliner", "llm")]:
        placed = place(dets, "auto")
        total = sum(VRAM_ESTIMATE_MB[d] for d, dev in placed.items() if dev != "cpu") + (VRAM_ESTIMATE_MB["llm"] if "llm" in dets else 0)
        assert total <= VRAM_BUDGET_MB
