"""Regression tests for the 40 findings confirmed by the adversarial review (one or more per finding)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neural_wave.config import make_config
from neural_wave.eval.privacy import score
from neural_wave.pipeline import Anonymizer
from neural_wave.spans import Span, resolve_overlaps
from neural_wave.text import Doc

ROOT = Path(__file__).resolve().parents[1]


def anon(note: str = "", dialogue: str = "", codes=None, preset: str = "balanced") -> dict:
    rec = {"encounter_id": "T", "dataset": "t", "dialogue": dialogue, "note": note, "icd10_codes": codes or []}
    return Anonymizer(make_config(preset)).anonymize(rec)[0]


# ---------------------------------------------------------------- privacy leaks


def test_partial_overlap_is_unioned_not_dropped():
    spans = [Span("note", 0, 15, "ADDRESS", "x"), Span("note", 10, 23, "PHONE", "y")]
    (only,) = resolve_overlaps(spans)
    assert (only.start, only.end) == (0, 23)
    assert "960 418" not in anon("Secretary 01632 960 418 | Bleep 2231")["note"]


@pytest.mark.parametrize("text", ["Age: 77 yrs", "Mr. Smith, aged 77 years, presents."])
def test_age_with_years_unit_after_age_cue(text):
    assert "77" not in anon(text)["note"]


def test_duration_in_years_is_kept():
    assert "5 years" in anon("Symptoms for 5 years.")["note"]


@pytest.mark.parametrize(
    "note,leak",
    [
        ("Mr. José Ramírez is a 45-year-old male.", ["Jos", "Ram", "írez"]),
        ("Ms. Renée Okafor and Dr. Wiśniewski saw Zoë Müller today.", ["Ren", "Okafor", "niewski", "Müller"]),
        ("PATIENT: BÉLANGER-ROY, Jean-François", ["BÉLANGER", "Jean"]),
    ],
)
def test_accented_names(note, leak):
    out = anon(note)["note"]
    for piece in leak:
        assert piece not in out


def test_accented_name_in_lowercase_dialogue():
    out = anon(dialogue="[doctor] hi josé , how are you ?\n[patient] fine . i saw dr. lópez .")["dialogue"]
    assert "jos" not in out and "lópez" not in out


def test_turkish_dotted_i_keeps_offsets():
    out = anon("PATIENT: İlhan Yılmaz\nMr. Yılmaz is a 45-year-old male. BP 120/80.")["note"]
    assert "İlhan" not in out and "Yılmaz" not in out and "45" not in out and "120/80" in out
    d = Doc("note", "İİİ abc")
    assert len(d.lower) == len(d.raw)


def test_unicode_hyphens():
    out = anon("A 45‑year‑old man, DOB 03‑04‑1956, phone 555‑123‑4567.")["note"]
    assert "45" not in out and "1956" not in out and "4567" not in out


def test_shield_on_one_word_does_not_keep_whole_organisation():
    out = anon("She works at Kierland Golf Club. She reports chest pain while golfing.")["note"]
    assert "Kierland" not in out and "golfing" in out


def test_code_evidence_index_notes_do_not_shield_new_york():
    out = anon("Lives in New York. Heart failure.", codes=[{"code": "I50.9", "description": "Heart failure, unspecified"}])["note"]
    assert "New York" not in out and "Heart failure" in out


def test_title_with_stopword_first_name_keeps_surname_removed():
    out = anon(dialogue="[patient] i saw doctor ray castellano last week .")["dialogue"]
    assert "castellano" not in out


def test_surname_comma_first_header_and_initials():
    out = anon("PATIENT: DUFRESNE, Jerome  (PACS displays initials J.D.)")["note"]
    assert "DUFRESNE" not in out and "Jerome" not in out and "J.D." not in out


def test_sentence_initial_common_word_name():
    out = anon("Mark Hill is a 54-year-old male. Mark reports elbow pain.", "[doctor] so mark tell me about it .")
    assert "Mark" not in out["note"] and "mark" not in out["dialogue"]


def test_vocative_after_so_but_colour_word_kept():
    out = anon(dialogue="[doctor] hey there mark , hi .\n[doctor] so mark tell me . it is so white .")["dialogue"]
    assert "mark" not in out and "so white" in out


# ---------------------------------------------------------------- over-removal


@pytest.mark.parametrize(
    "note,kept",
    [
        ("She denies urgent care or emergency room visits.", "denies urgent care"),
        ("Well-healed surgical scars.", "surgical scars"),
        ("Follow up at Heart Failure Clinic.", "Heart Failure Clinic"),
        ("Target BP is 130/80.", "Target BP"),
        ("CVS: regular rate.", "CVS:"),
        ("GE junction normal.", "GE junction"),
        ("Pfizer vaccine given.", "Pfizer vaccine"),
        ("It only works for the barky cough.", "barky cough"),
        ("Fracture of the tarsus.", "tarsus"),
        ("History of kawasaki disease.", "kawasaki"),
        ("Rapid cycling bipolar disorder.", "cycling"),
        ("Guillain barre syndrome.", "barre"),
        ("Incarcerated hernia.", "Incarcerated hernia"),
        ("Dilated foramen of Munro. Fitzpatrick skin type IV.", "Munro"),
        ("CK 2000, viral load 150000 copies/mL, 2000 kcal diet.", "2000 kcal"),
        ("Neutrophils 12000.", "12000"),
        ("Gestational age: 32+4, former 34+3-week-old.", "32+4"),
        ("A 5-day-old wound on the shin.", "5-day-old wound"),
        ("Glucose (ref 70-99).", "70-99"),
        ("G2P1A0. ECG S1Q3T3.", "S1Q3T3"),
        ("Takes Indole-3-carbinol. Uses 2 puffs in Welixa.", "Welixa"),
        ("Fever to 102 F. 16 F Foley placed.", "102 F"),
        ("Prednisone taper 40-30-20-10 mg.", "40-30-20-10"),
    ],
)
def test_clinical_text_survives(note, kept):
    assert kept in anon(note, preset="aggressive")["note"]


@pytest.mark.parametrize(
    "dialogue,kept",
    [
        ("[doctor] your temperature you're ninety seven . eight .", "ninety seven"),
        ("[patient] i'm about a seven on the pain scale .", "a seven"),
        ("[doctor] you're five eight tall .", "five eight"),
        ("[doctor] on the 5th metatarsal , after the 2nd dose .", "5th metatarsal"),
        ("[patient] my head is swimming .", "swimming"),
    ],
)
def test_spoken_clinical_values_survive(dialogue, kept):
    assert kept in anon(dialogue=dialogue, preset="aggressive")["dialogue"]


# ---------------------------------------------------------------- critic gaps


def test_spoken_digit_identifiers():
    out = anon(dialogue="[patient] my medical record number is one two three four five six . my phone number is five five five one two three four five six seven")["dialogue"]
    assert "one two three four" not in out and "five five five one" not in out


def test_relatives_names_and_ages():
    out = anon(dialogue="[patient] my daughter sarah is 45 , my husband john is seventy one .")["dialogue"]
    assert "sarah" not in out and "45" not in out and "john" not in out and "seventy one" not in out


def test_more_date_formats():
    out = anon("Admitted 10/9, extubated on 10/12. Seen 15OCT2026.")["note"]
    assert "10/9" not in out and "10/12" not in out and "15OCT2026" not in out


def test_phone_in_parentheses_keeps_balance():
    out = anon("Call me (405-555-0160).")["note"]
    assert "[PHONE])" in out and "(405" not in out


def test_newlines_between_identical_placeholders_survive():
    out = anon("Re: John Smith\nJohn reported pain.")["note"]
    assert out.count("\n") == 1


# ---------------------------------------------------------------- constraints / robustness


def test_decisions_do_not_embed_names():
    rec = {"encounter_id": "T", "dataset": "t", "dialogue": "[doctor] i'm doctor jacob ward .", "note": "", "icd10_codes": []}
    _, dec, _ = Anonymizer(make_config("balanced")).anonymize(rec)
    assert all("jacob" not in json.dumps(d).lower() for d in dec)


@pytest.mark.parametrize("codes", [[None], ["I10"], [{"code": None, "description": None}], None])
def test_malformed_codes_do_not_crash(codes):
    rec = {"encounter_id": "T", "dataset": "t", "dialogue": "", "note": "Mr. Smith has hypertension.", "icd10_codes": codes}
    out, _, _ = Anonymizer(make_config("balanced")).anonymize(rec)
    assert out["icd10_codes"] == codes and "Smith" not in out["note"]


def test_uppercase_speaker_tags():
    out = anon(dialogue="[DOCTOR] hi , andrew . how are you ?\n[PATIENT] fine .")["dialogue"]
    assert "andrew" not in out


def test_no_codes_cli_on_record_without_codes(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text(json.dumps({"encounter_id": "A", "dataset": "t", "dialogue": "", "note": "Mr. Smith."}) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "neural_wave.cli", "anonymize", "-i", str(inp), "-o", str(out), "--no-codes"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text().splitlines()[0])
    assert list(rec) == ["encounter_id", "dataset", "dialogue", "note"] and "Smith" not in rec["note"]


def test_bom_input(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text("﻿" + json.dumps({"encounter_id": "A", "dataset": "t", "dialogue": "", "note": "x", "icd10_codes": []}) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "neural_wave.cli", "anonymize", "-i", str(inp), "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_ollama_calls_bypass_proxies():
    from neural_wave import models

    assert models.OLLAMA_URL.startswith("http://127.0.0.1")
    assert not any(type(h).__name__ == "ProxyHandler" and h.proxies for h in models.LOCAL_HTTP.handlers)


# ---------------------------------------------------------------- metrics


def test_privacy_score_counts_partial_survival_as_leak():
    orig = [{"encounter_id": "E", "dataset": "t", "dialogue": "", "note": "Secretary 01632 960 418."}]
    out = [{"encounter_id": "E", "dataset": "t", "dialogue": "", "note": "[ADDRESS] 960 418."}]
    gold = {"E": {"encounter_id": "E", "pii": [{"field": "note", "text": "01632 960 418", "category": "ID", "risk": "high"}], "keep": []}}
    assert score(orig, out, gold)["removed"]["ALL"] == 0.0


def test_concept_recall_ignores_removed_pii():
    from neural_wave.eval.utility import concept_recall

    orig = [{"encounter_id": "E", "dialogue": "", "note": "Seen at Mercy General Hospital for hypertension."}]
    out = [{"encounter_id": "E", "dialogue": "", "note": "Seen at [FACILITY] for hypertension."}]
    gold = {"E": {"pii": [{"field": "note", "text": "Mercy General Hospital"}]}}
    assert concept_recall(orig, out, gold=gold)["concept_recall"] == 1.0


def test_evaluate_reports_dropped_records(tmp_path):
    orig = ROOT / "resources/data/eval_patient_doctor_encounters.jsonl"
    short = tmp_path / "short.jsonl"
    short.write_text("".join(orig.read_text(encoding="utf-8").splitlines(keepends=True)[:3]), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "neural_wave.cli", "evaluate", "--original", str(orig), "--anonymized", str(short), "--icd", "none", "--name", "t_dropped", "--out-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "t_dropped.json").read_text())
    assert any("record count differs" in e for e in res["schema_errors"])
