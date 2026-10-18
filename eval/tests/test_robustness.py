"""The robust-set converters (scripts/convert/robust_*.py) on tiny inputs, and the robustness table of bonus/robustness.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from eval.bonus import robustness as R

CONVERT = Path(__file__).resolve().parents[2] / "scripts" / "convert"
ICD = {"E11.9": "Type 2 diabetes mellitus without complications"}


@pytest.fixture
def conv(monkeypatch):
    """The converter modules, with the ICD-10-CM table fixed (the real one is read from data/kaggle)."""
    monkeypatch.syspath_prepend(str(CONVERT))
    for name in ("common", "validate", "robust_discharge", "robust_medconv", "robust_synthetic"):
        sys.modules.pop(name, None)
    import common
    import robust_discharge
    import robust_medconv
    import robust_synthetic
    import validate

    monkeypatch.setattr(common, "icd10cm", lambda: ICD)
    monkeypatch.setattr(validate, "icd10cm", lambda: ICD)
    yield common, validate, robust_discharge, robust_medconv, robust_synthetic
    for name in ("common", "validate", "robust_discharge", "robust_medconv", "robust_synthetic"):
        sys.modules.pop(name, None)


def discharge_row(i: int) -> dict:
    j = {"hospital": {"name": "Apollo Hospital", "address": "12 Park Road, Chennai 600006", "npi": "1234567890"},
         "patient": {"name": "Karan Malhotra", "dob": "1992-07-15", "mrn": "MRN-301928"}, "insurance": "Star Health",
         "diagnosis": "Type 2 Diabetes Mellitus", "hospital_course": "Admitted for poor glycemic control. Insulin started.",
         "procedures": [{"desc": "HbA1c test", "cpt": "83036"}], "medications": ["Metformin 500mg BID"], "icd_10": "E11.9"}  # fmt: skip
    return {"filename": f"med_doc_discharge_summary_{200001 + i}_noisy.jpg", "document_type": "discharge_summary", "json_data": json.dumps(j)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read(d: Path) -> tuple[list[dict], list[dict], dict]:
    return ([json.loads(x) for x in (d / "encounters.jsonl").open()], [json.loads(x) for x in (d / "labels.jsonl").open()],
            json.loads((d / "manifest.json").read_text()))  # fmt: skip


def test_robust_discharge_layouts_and_gold(conv, tmp_path):
    _, validate, rd, _, _ = conv
    write_csv(tmp_path / "src" / "discharge_summaries_ground_truth.csv", [discharge_row(i) for i in range(3)])
    rd.main(["--src", str(tmp_path / "src"), "--out", str(tmp_path / "out"), "--layouts", "form", "letter", "dictation"])
    encs, labs, manifest = read(tmp_path / "out" / "robust_discharge")
    assert [lab["meta"]["layout"] for lab in labs] == ["form", "letter", "dictation"] and manifest["stage"] == "inject"
    assert [e["encounter_id"] for e in encs] == ["DS200001", "DS200002", "DS200003"] and encs[2]["note"] == encs[2]["note"].lower()
    assert encs[0]["icd10_codes"] == [{"code": "E11.9", "description": ICD["E11.9"]}]
    errs, _, _ = validate.check(tmp_path / "out" / "robust_discharge")
    assert errs == []
    dictation = {(it["text"].lower(), it["category"], it["label"]) for it in labs[2]["pii"]}
    assert {("karan", "PATIENT_NAME", "patient.name"), ("3 0 1 9 2 8", "ID", "patient.mrn")} <= dictation
    assert {lab for _, _, lab in dictation} >= {"injected:admission_date", "injected:discharge_date", "injected:physician", "injected:phone", "injected:age"}
    assert labs[2]["keep"][0]["category"] == "CODE_EVIDENCE" and labs[2]["keep"][0]["why"] == "E11.9"
    # the default keeps only the dictation layout, which devp_discharge does not render
    rd.main(["--src", str(tmp_path / "src"), "--out", str(tmp_path / "default")])
    assert [e["encounter_id"] for e in read(tmp_path / "default" / "robust_discharge")[0]] == ["DS200003"]


def test_robust_discharge_is_seeded_by_row(conv):
    rd = conv[2]
    first, again = rd.render(4, discharge_row(4)), rd.render(4, discharge_row(4))
    assert first == again and first[0]["note"] != rd.render(7, discharge_row(4))[0]["note"]


def test_robust_medconv_injects_labelled_pii(conv, tmp_path):
    _, validate, _, rm, _ = conv
    rows = [{"Conversation": f"[|Human|] I have had a cough for {k} weeks. What should I do? [|AI|] Hi, drink fluids and rest. [|Human|] Thanks."}
            for k in range(5)] + [{"Conversation": "[|Human|] only one turn"}]  # fmt: skip
    write_csv(tmp_path / "src" / "test.csv", rows)
    rm.main(["--src", str(tmp_path / "src"), "--out", str(tmp_path / "out"), "--n", "4"])
    encs, labs, manifest = read(tmp_path / "out" / "robust_medconv")
    assert len(encs) == 4 and all(lab["pii_gold"] == "partial" and lab["icd10_source"] == "none" for lab in labs)
    assert validate.check(tmp_path / "out" / "robust_medconv")[0] == []
    for e, lab in zip(encs, labs):
        assert e["dialogue"].startswith("[patient] ") and "\n[doctor] " in e["dialogue"] and "Regards, Dr. " in e["dialogue"]
        assert lab["pii"] and all(it["label"].startswith("injected:") for it in lab["pii"])
        assert {it["category"] for it in lab["pii"]} >= {"PATIENT_NAME", "PERSON_NAME"}
        for it in lab["pii"]:  # an ordinary-word name is labelled only as the full name or after a cue
            assert it["text"].lower() not in rm.COMMON_NAMES
    assert rm.build(rows, 4) == rm.build(rows, 4)


def test_robust_synthetic_copies_and_folds_nested_items(conv, tmp_path):
    _, validate, _, _, rs = conv
    note = "Name: PRICE, William\nWill Price is a 67 y/o man. He will need a cast. Seen by Dr. Anika Sorensen."
    rec = {"encounter_id": "SYN001", "dataset": "er_triage", "dialogue": "", "note": note,
           "icd10_codes": [{"code": "S52.531A", "description": "Colles' fracture of right radius"}]}  # fmt: skip
    gold = {"encounter_id": "SYN001", "pii": [
        {"field": "note", "text": "Price", "category": "PATIENT_NAME", "risk": "high"},
        {"field": "note", "text": "Will Price", "category": "PATIENT_NAME", "risk": "high"},
        {"field": "note", "text": "Anika Sorensen", "category": "PERSON_NAME", "risk": "high"},
        {"field": "note", "text": "Sorensen", "category": "PERSON_NAME", "risk": "high"},
        {"field": "note", "text": "67 y/o", "category": "AGE", "risk": "high"}],
        "keep": [{"field": "note", "text": "will need", "category": "NOT_A_NAME", "why": "verb"}]}  # fmt: skip
    (tmp_path / "src" / "robustness").mkdir(parents=True)
    (tmp_path / "src" / "gold" / "robust_synthetic").mkdir(parents=True)
    (tmp_path / "src" / "robustness" / "synthetic.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "src" / "gold" / "robust_synthetic" / "SYN001.json").write_text(json.dumps(gold))
    rs.main(["--src", str(tmp_path / "src"), "--out", str(tmp_path / "out")])
    encs, labs, manifest = read(tmp_path / "out" / "robust_synthetic")
    assert encs == [rec] and validate.check(tmp_path / "out" / "robust_synthetic")[0] == []
    # `Price` survives on its own occurrence (PRICE, William); `Sorensen` only occurs inside `Anika Sorensen`
    assert [it["text"] for it in labs[0]["pii"]] == ["PRICE", "Will Price", "67 y/o", "Anika Sorensen"]
    assert "1 gold strings gave no span" in manifest["pii_gold"]["notes"] and labs[0]["keep"][0]["text"] == "will need"


def score(privacy: float, high: float, terms: float) -> dict:
    return {"privacy": {"removed": privacy, "removed_high_risk": high}, "terms": {"kept": terms}}


def test_table_reads_scores_and_drift_ranks(tmp_path, monkeypatch):
    scores = tmp_path / "scores"
    for model, s, val in (("a", "robust_discharge", score(0.9, 1.0, 0.99)), ("a", "robust_synthetic", {"error": "run failed"}),
                          ("b", "robust_medconv", score(0.5, 0.25, 1.0))):  # fmt: skip
        (scores / model).mkdir(parents=True, exist_ok=True)
        (scores / model / f"{s}.json").write_text(json.dumps(val))
    (tmp_path / "drift.json").write_text(json.dumps({"complexity_rank": {"robust_discharge": 11, "robust_medconv": 5}}))
    monkeypatch.setattr(R, "SCORES_DIR", scores)
    monkeypatch.setattr(R, "DRIFT", tmp_path / "drift.json")
    lines = R.table(["a", "b"]).splitlines()
    assert lines[2] == "| complexity rank | 11 |  |  | 5 |  |  | - |  |  |"
    assert lines[3] == "| a | 0.900 | 1.000 | 0.990 | - | - | - | err | err | err |"
    assert lines[4] == "| b | - | - | - | 0.500 | 0.250 | 1.000 | - | - | - |"


def test_main_converts_missing_sets_builds_suites_and_prints(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(R, "convert", lambda force: calls.append(("convert", force)) or [])
    monkeypatch.setattr(R.S, "ensure", lambda names, force=False: calls.append(("ensure", tuple(names), force)) or {
        n: {"records": 2, "pii_items": 3, "pii_gold": ["exhaustive"], "keep_items": 0, "with_icd10": 0} for n in names if n != "robust_medconv"})
    monkeypatch.setattr(R, "SCORES_DIR", tmp_path / "none")
    assert R.main(["--force"]) == 1  # robust_medconv did not build
    assert calls == [("convert", True), ("ensure", R.ROBUST, True)]
    out = capsys.readouterr()
    assert "robust_discharge     2 records" in out.out and "no scores on the robust suites yet" in out.out and "failed: robust_medconv" in out.err
