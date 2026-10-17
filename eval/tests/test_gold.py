"""Merged ACI gold: taxonomy rules, the nesting rule, keep and concept merging, levels in metrics.privacy, deterministic
output, the suite built from it, and (when data/ is there) the A+D cross-check."""

from __future__ import annotations

import json

import pytest

from eval import metrics as M
from eval import suites as S
from eval.common import CATEGORY_RISK, REPO, load_jsonl
from eval.gold import audit_gold as A
from eval.gold import merge_aci_gold as G
from eval.gold import taxonomy as T

REAL_COMMON_WORDS = M.common_words  # conftest swaps it for a toy list in every test

DIALOGUE = ("Doctor: Good morning, Martha Collins. How is Martha today? Patient: Martha is fine, I played golf at Pebble Beach "
            "and soccer in the Sunset Soccer League. Doctor: Born on march 3, 1970? Patient: Yes. Doctor: And dax, start the note.")
NOTE = "Martha Collins, 53-year-old, plays golf. DOB 03/03/1970. Suspected Parkinson's disease. Next visit in 2025."
REC = {"encounter_id": "e1", "dataset": "toy", "dialogue": DIALOGUE, "note": NOTE, "icd10_codes": [{"code": "G20", "description": "Parkinson's disease"}]}


def h(field: str, text: str, cat: str) -> dict:
    return {"field": field, "text": text, "category": cat, "risk": CATEGORY_RISK[cat], "label": cat}


def raw(field: str, text: str, label: str) -> dict:
    return {"field": field, "text": text, "label": label}


HARNESS = {"pii": [h("dialogue", "martha", "PATIENT_NAME"), h("dialogue", "collins", "PATIENT_NAME"), h("note", "Martha Collins", "PATIENT_NAME"),
                   h("dialogue", "march 3, 1970", "DOB"), h("note", "03/03/1970", "DOB"), h("note", "53-year-old", "AGE")],
           "keep": [{"field": "dialogue", "text": "golf", "category": "CODE_EVIDENCE", "why": "activity code"}]}  # fmt: skip
VM = {"pii": [raw("dialogue", "martha collins", "NAME_PATIENT"), raw("dialogue", "pebble beach", "LOCATION"), raw("dialogue", "golf", "ACTIVITY"),
              raw("dialogue", "march 3, 1970", "DATE"), raw("note", "53-year-old", "AGE"), raw("note", "2025", "DATE"), raw("dialogue", "soccer", "ACTIVITY")],
      "concepts": [{"field": "note", "text": "Parkinson's disease", "type": "DIAGNOSIS", "icd_relevant": True},
                   {"field": "note", "text": "martha collins", "type": "FINDING", "icd_relevant": False}]}  # fmt: skip
KETAI = {"pii": [raw("dialogue", "Martha Collins", "PATIENT_NAME"), raw("note", "Martha Collins", "PATIENT_NAME"), raw("dialogue", "march 3, 1970", "DATE"),
                 raw("note", "53-year-old", "AGE"), raw("note", "03/03/1970", "DATE"), raw("dialogue", "sunset soccer league", "ORGANIZATION")],
         "concepts": [{"field": "note", "text": "parkinson's disease", "type": "diagnosis", "icd_relevant": False},
                      {"field": "note", "text": "DOB", "type": "lab_or_test", "icd_relevant": False}],
         "not_pii": [{"field": "note", "text": "Parkinson's", "why": "eponym"}, {"field": "dialogue", "text": "dax", "why": "software"},
                     {"field": "dialogue", "text": "soccer", "why": "ordinary sport"}]}  # fmt: skip
GOLDS = {"harness": HARNESS, "vm": VM, "ketai": KETAI}


def by_text(items: list[dict], field: str = "dialogue") -> dict[str, dict]:
    return {it["text"].lower(): it for it in items if it["field"] == field}


@pytest.fixture
def merged() -> tuple[dict, dict]:
    return G.merge_encounter(REC, GOLDS)


# taxonomy ------------------------------------------------------------------------------------------------------------


def test_labels_map_to_common_categories():
    assert [T.category("vm", lab, "x") for lab in ("NAME_PATIENT", "NAME_RELATIVE", "NAME_CLINICIAN", "CONTACT", "ACTIVITY", "TEMPORAL", "BOGUS")] == [
        "PATIENT_NAME", "PERSON_NAME", "PERSON_NAME", "CONTACT", "OTHER_QI", "OTHER_QI", "OTHER_QI"]  # fmt: skip
    assert [T.category("ketai", lab, "x") for lab in ("CLINICIAN_NAME", "OTHER_PERSON_NAME", "OTHER_QUASI", "ID")] == ["PERSON_NAME", "PERSON_NAME", "OTHER_QI", "ID"]
    assert T.category("harness", "YEAR", "2019") == "YEAR" and T.category("harness", "NEW_THING", "x") == "OTHER_QI"
    assert CATEGORY_RISK["CONTACT"] == "high" and T.DIRECT == {"PATIENT_NAME", "PERSON_NAME", "DOB", "DATE", "ID", "CONTACT"}
    assert T.concept_type("lab_or_test") == "TEST" and T.concept_type("anatomy") == "ANATOMY" and T.concept_type("FINDING") == "FINDING"


def test_date_becomes_dob_or_year():
    text = "Born on march 3, 1970. Seen on june 2, 2020, in 2021. His date of birth is a long story, june 2 again"
    assert T.category("vm", "DATE", "march 3, 1970", text) == "DOB"  # cue right before
    assert T.category("ketai", "DATE", "june 2, 2020", text) == "DATE"
    assert T.category("ketai", "DATE", "june 2, 2020", text, dob_elsewhere=True) == "DOB"  # the harness says DOB
    assert T.category("vm", "DATE", "2021", text) == "YEAR"
    assert T.category("vm", "DATE", "june 2", "date of birth is a long story, june 2") == "DATE"  # cue more than 25 characters before


def test_majority_and_tie_priority():
    assert T.majority({"harness": "DATE", "vm": "YEAR", "ketai": "YEAR"}) == "YEAR"
    assert T.majority({"harness": "OTHER_QI", "ketai": "DATE"}) == "OTHER_QI"  # harness > ketai
    assert T.majority({"vm": "LOCATION", "ketai": "ORGANIZATION"}) == "ORGANIZATION"  # ketai > vm


# nesting -------------------------------------------------------------------------------------------------------------


def test_longest_span_is_canonical_with_nested_votes(merged):
    row, log = merged
    d = by_text(row["pii"])
    mc = d["martha collins"]
    assert mc["text"] == "Martha Collins" and mc["category"] == "PATIENT_NAME" and mc["direct"] and mc["risk"] == "high"
    assert mc["sources"] == ["harness", "vm", "ketai"] and mc["votes"] == 3 and mc["nested_votes"] == 1  # harness only through martha/collins
    assert mc["source_labels"] == {"vm": "NAME_PATIENT", "ketai": "PATIENT_NAME"} and mc["label"] == "PATIENT_NAME"
    assert mc["contains"] == ["collins", "martha"] and mc["standalone_occurrences"] == 1 and "within" not in mc


def test_nested_item_survives_only_with_standalone_occurrences(merged):
    row, log = merged
    d = by_text(row["pii"])
    assert d["martha"]["within"] == ["Martha Collins"] and d["martha"]["standalone_occurrences"] == 2
    assert d["martha"]["votes"] == 3 and d["martha"]["nested_votes"] == 2  # inherits the container's votes
    assert "collins" not in d  # only inside `Martha Collins`: absorbed
    assert [(a["text"], a["sources"], a["within"]) for a in log["absorbed"]] == [("collins", ["harness"], ["Martha Collins"])]


def test_merge_pii_spans_and_votes():
    rec = {"dialogue": "Dr. Gutierrez saw me. Gutierrez said so.", "note": "x"}
    items, absorbed, _ = G.merge_pii(rec, {"harness": [h("dialogue", "gutierrez", "PERSON_NAME")],
                                           "ketai": [{"field": "dialogue", "text": "dr. gutierrez", "category": "PERSON_NAME", "label": "CLINICIAN_NAME"}]})  # fmt: skip
    d = by_text(items)
    assert d["dr. gutierrez"]["votes"] == 2 and d["gutierrez"]["standalone_occurrences"] == 1 and d["gutierrez"]["votes"] == 2 and not absorbed


def test_category_conflict_and_dob(merged):
    row, log = merged
    d = by_text(row["pii"])
    assert d["march 3, 1970"]["category"] == "DOB" and d["march 3, 1970"]["source_labels"] == {"harness": "DOB", "vm": "DATE", "ketai": "DATE"}
    assert by_text(row["pii"], "note")["2025"]["category"] == "YEAR"
    lists = row["pii_by_source"]
    assert {it["text"]: it["category"] for it in lists["vm"]}["march 3, 1970"] == "DOB"  # the harness says DOB
    assert {it["text"]: it["category"] for it in lists["ketai"]}["03/03/1970"] == "DOB"
    assert lists["harness"] == G.distinct_items(HARNESS["pii"], "category", "risk", "label")  # the old suite list, unchanged


def test_keep_conflicts_and_rejected_votes(merged):
    row, log = merged
    d, keep = by_text(row["pii"]), by_text(row["keep"])
    assert "golf" not in d and keep["golf"]["category"] == "CODE_EVIDENCE"  # VM alone said PII: keep wins
    assert [(x["text"], x["winner"]) for x in log["pii_keep"]] == [("soccer", "keep"), ("golf", "keep")]
    assert "soccer" not in d and d["sunset soccer league"]["votes"] == 1  # ketai lists `soccer` as not PII: no vote through nesting
    assert keep["dax"]["category"] == "NOT_A_NAME" and keep["soccer"]["sources"] == ["ketai"]
    assert by_text(row["keep"], "note")["parkinson's"]["category"] == "EPONYM"  # inside a concept


def test_pii_wins_with_two_votes():
    golds = {"harness": {"pii": [], "keep": HARNESS["keep"]}, "vm": {"pii": [raw("dialogue", "golf", "ACTIVITY")]},
             "ketai": {"pii": [raw("dialogue", "golf", "OTHER_QUASI")]}}  # fmt: skip
    row, log = G.merge_encounter(REC, golds)
    assert by_text(row["pii"])["golf"]["votes"] == 2 and row["keep"] == [] and log["pii_keep"][0]["winner"] == "pii"


def test_concepts_union(merged):
    row, log = merged
    c = by_text(row["concepts"], "note")
    assert c["parkinson's disease"] == {"field": "note", "text": "parkinson's disease", "type": "DIAGNOSIS", "icd_relevant": True,
                                        "icd_relevant_votes": 1, "sources": ["vm", "ketai"], "votes": 2}  # fmt: skip
    assert c["dob"]["type"] == "TEST" and "martha collins" not in c  # equal to a direct PII item: dropped
    assert [x["text"] for x in log["dropped_concepts"]] == ["martha collins"]


# levels and privacy ----------------------------------------------------------------------------------------------------


def test_level_filters():
    assert all(T.in_level({"field": "note", "text": "x"}, lvl) for lvl in T.LEVELS)  # no vote metadata: every level
    assert [T.in_level({"votes": 1, "direct": False}, lvl) for lvl in T.LEVELS] == [True, False, False]
    assert [T.in_level({"votes": 2, "direct": False}, lvl) for lvl in T.LEVELS] == [True, True, False]
    assert [T.in_level({"votes": 1, "direct": True}, lvl) for lvl in T.LEVELS] == [True, True, True]


def labels_of(row: dict) -> dict[str, dict]:
    return {row["encounter_id"]: {"encounter_id": row["encounter_id"], "pii_gold": "exhaustive", "language": "en", **row}}


def test_privacy_levels_and_sources(merged):
    row, _ = merged
    anon = {**REC, "dialogue": DIALOGUE.replace("Martha Collins", "[PATIENT]").replace("Martha", "[PATIENT]").replace("march 3, 1970", "[DOB]"),
            "note": NOTE.replace("Martha Collins", "[PATIENT]").replace("03/03/1970", "[DOB]").replace("53-year-old", "[AGE]").replace("2025", "[YEAR]")}  # fmt: skip
    p = M.privacy([REC], [anon], labels_of(row))
    assert p["levels"]["direct"]["removed"] == 1.0 and p["levels"]["consensus"]["removed"] == 1.0 and p["removed"] == 1.0
    assert p["levels"]["union"]["removed"] == 0.8182 and p["levels"]["union"]["entity_removed"] == 0.7778  # two single-vote items left
    assert p["levels"]["union"]["items"] == 9 and p["levels"]["consensus"]["items"] == 6 and p["levels"]["direct"]["items"] == 5
    assert p["entity_removed"] == 1.0 and p["counts"]["entity:union"] == [7, 9] and p["counts"]["level:union"] == [9, 11]
    assert p["breakdown"]["votes:1"] == 0.3333 and p["counts"]["src:vm"] == [7, 8] and p["counts"]["ALL"] == p["counts"]["level:consensus"] == [8, 8]
    assert list(p["by_source"]) == ["harness", "vm", "ketai"] and p["by_source"]["harness"]["removed"] == 1.0
    assert p["by_source"]["vm"] == {"removed": 0.5, "removed_high_risk": 1.0, "entity_removed": 0.5714, "items": 7, "occurrences": 8}  # VM's own list
    assert [x["text"] for x in p["leaks"]] == ["sunset soccer league", "pebble beach"] and p["leaks"][0]["votes"] == 1


def test_levels_coincide_without_metadata(original, clean, labels):
    clean[2]["dialogue"] = "Doctor: [PATIENT], how is [LOCATION]? Patient: Boston is cold."
    p = M.privacy(original, clean, labels)
    assert {lvl: v["removed"] for lvl, v in p["levels"].items()} == dict.fromkeys(T.LEVELS, p["removed"]) == dict.fromkeys(T.LEVELS, 0.9091)
    assert p["by_source"] == {} and p["entity_removed"] == p["breakdown"]["items_fully_removed"] == 0.8889


# files: merge, audit, suite --------------------------------------------------------------------------------------------


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """The synthetic encounter as the three source files, plus a train record the merge must skip."""
    harness, vm, out = tmp_path / "converted" / "aci_bench", tmp_path / "vm", tmp_path / "gold" / "aci_eval"
    harness.mkdir(parents=True)
    vm.mkdir()
    train = {**REC, "encounter_id": "t1"}
    (harness / "encounters.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (train, REC)))
    labs = [{"encounter_id": "t1", "pii_gold": "exhaustive", "icd10_source": "gold", "meta": {"split": "train"}, "pii": [], "keep": []},
            {"encounter_id": "e1", "pii_gold": "exhaustive", "icd10_source": "gold", "meta": {"split": "eval", "language": "en"}, **HARNESS}]  # fmt: skip
    (harness / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labs))
    (harness / "manifest.json").write_text('{"license": "test", "language": "en"}')
    (vm / "e1.json").write_text(json.dumps({"encounter_id": "e1", "phi": [{"field": i["field"], "text": i["text"], "type": i["label"], "tier": 1} for i in VM["pii"]],
                                            "medical_concepts": VM["concepts"]}))  # fmt: skip
    ketai = tmp_path / "eval_gold.jsonl"
    ketai.write_text(json.dumps({"encounter_id": "e1", "pii": [{"field": i["field"], "text": i["text"], "category": i["label"]} for i in KETAI["pii"]],
                                 "medical_concepts": KETAI["concepts"],
                                 "ambiguous_terms": [{"field": t["field"], "text": t["text"], "is_pii": False, "comment": t["why"]} for t in KETAI["not_pii"]]}) + "\n")  # fmt: skip
    for k, v in {"HARNESS": harness, "VM_DIR": vm, "KETAI": ketai, "OUT_DIR": out}.items():
        monkeypatch.setattr(G, k, v)
    return tmp_path, out


def test_main_is_deterministic(sources, capsys):
    tmp, out = sources
    files = ("merged.jsonl", "merged_stats.json", "merged_review.md")
    assert G.main([]) == 0
    first = {f: (out / f).read_bytes() for f in files}
    assert G.main([]) == 0 and first == {f: (out / f).read_bytes() for f in files} and not (out / "merged.prev.jsonl").exists()
    rows = load_jsonl(out / "merged.jsonl")
    assert [r["encounter_id"] for r in rows] == ["e1"] and list(rows[0]) == ["encounter_id", "pii", "pii_by_source", "keep", "concepts"]
    stats = json.loads((out / "merged_stats.json").read_text())
    assert stats["exact_overlap"]["all_three"] == 2 and stats["pii"]["levels"] == {"union": 9, "consensus": 6, "direct": 5}
    assert stats["inputs"]["vm"]["files"] == 1 and len(stats["merged_sha1"]) == 40
    (tmp / "vm" / "e1.json").write_text(json.dumps({"encounter_id": "e1", "phi": [], "medical_concepts": []}))
    G.main([])
    assert (out / "merged.prev.jsonl").read_bytes() == first["merged.jsonl"]


def test_missing_sources(sources, capsys):
    tmp, out = sources
    (tmp / "eval_gold.jsonl").unlink()
    G.main([])
    assert "merging without it" in capsys.readouterr().err and json.loads((out / "merged_stats.json").read_text())["exact_overlap"]["all_three"] is None
    (tmp / "converted" / "aci_bench" / "labels.jsonl").unlink()
    (out / "merged.jsonl").unlink()
    assert G.main([]) == 0 and not (out / "merged.jsonl").exists()


def test_audit(sources, monkeypatch):
    _, out = sources
    assert A.main([]) == 1  # no merged gold yet
    assert G.main(["--audit"]) == 0
    res = json.loads((out / "audit.json").read_text())["summary"]
    assert res["verbatim"]["vm"] == {"items": 9, "not_whole_word": 0, "not_substring": 0}
    assert res["single_votes"]["items"] == 3 and res["possibly_missed"] == {} and res["diff"] is None
    assert "## Possibly missed (0)" in (out / "audit.md").read_text()
    assert [k for k, _, _ in A.candidates("aged 53, 03/03/1970, Dr. Lee, 555-123-4567, mrn 1234567, 2000 IU, may 2020")] == [
        "AGE", "DATE", "DATE", "TITLE+NAME", "CONTACT", "ID"]  # fmt: skip


@pytest.fixture
def suite(sources, monkeypatch):
    tmp, out = sources
    monkeypatch.setattr(S, "CONVERTED", tmp / "converted")
    monkeypatch.setattr(S, "SUITES_DIR", tmp / "suites")
    monkeypatch.setattr(S, "GOLD_DIR", tmp / "gold")
    monkeypatch.setattr(S, "VM_ANNOTATIONS", tmp / "vm")
    monkeypatch.setattr(S, "SUITES", {"aci_eval": {"source": "aci_bench", "where": {"split": "eval"}, "role": "main", "about": "toy"}})
    return tmp, out


def test_suite_uses_the_merged_gold(suite, capsys):
    tmp, out = suite
    old = S.ensure(["aci_eval"])["aci_eval"]
    assert old["gold_sources"] == ["harness", "vm"] and old["pii_alt_records"] == 1 and "merged_sha1" not in old  # no merged gold yet
    G.main([])
    assert S.stale("aci_eval", old) == "source changed"
    info = S.ensure(["aci_eval"])["aci_eval"]
    assert info["gold_sources"] == ["harness", "vm", "ketai"] and info["pii_levels"] == {"union": 9, "consensus": 6, "direct": 5}
    assert info["merged_sha1"] == json.loads((out / "merged_stats.json").read_text())["merged_sha1"] and info["input_sha1"] == old["input_sha1"]
    lab = load_jsonl(tmp / "suites" / "aci_eval" / "labels.jsonl")[0]
    assert "pii_alt" not in lab and lab["pii"] == load_jsonl(out / "merged.jsonl")[0]["pii"] and set(lab["pii_by_source"]) == {"harness", "vm", "ketai"}
    assert S.ensure(["aci_eval"])["aci_eval"] == info  # up to date
    S.main(["--only", "aci_eval", "--no-merged"])
    off = S.ensure(["aci_eval"])["aci_eval"]
    assert off["merged_gold"] is False and off["gold_sources"] == ["harness", "vm"] and S.stale("aci_eval", off) == ""  # kept on rebuilds
    assert S.stale("aci_eval", off, use_merged=True) == "merged gold switched on"
    S.main(["--only", "aci_eval", "--merged", "--if-stale"])
    assert "merged_gold" not in json.loads((tmp / "suites" / "aci_eval" / "suite.json").read_text())


# real data (skipped without data/) -------------------------------------------------------------------------------------

SUITE_DIR, RUN = REPO / "data" / "eval" / "suites" / "aci_eval", REPO / "data" / "eval" / "runs" / "ad_night" / "aci_eval" / "output.jsonl"


def real_suite() -> tuple[list[dict], dict[str, dict]]:
    if not (SUITE_DIR / "labels.jsonl").exists():
        pytest.skip("data/eval/suites/aci_eval is not built")
    labels = {g["encounter_id"]: g for g in load_jsonl(SUITE_DIR / "labels.jsonl")}
    if not all("pii_by_source" in g for g in labels.values()):
        pytest.skip("aci_eval is not built from the merged gold")
    return load_jsonl(SUITE_DIR / "encounters.jsonl"), labels


def test_ad_night_harness_gold_cross_check(monkeypatch):
    """by_source.harness reproduces the A+D cross-check (the old aci_eval privacy) exactly."""
    original, labels = real_suite()
    if not RUN.exists():
        pytest.skip("no ad_night run on aci_eval")
    monkeypatch.setattr(M, "common_words", REAL_COMMON_WORDS)
    p = M.privacy(original, load_jsonl(RUN), labels)
    assert p["by_source"]["harness"]["removed"] == pytest.approx(0.8388, abs=1e-4) and p["by_source"]["vm"]["removed"] == pytest.approx(0.5398, abs=1e-4)

