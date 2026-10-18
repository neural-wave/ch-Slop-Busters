"""Merging the two silver-label formats (synthetic encounter, invented names)."""

from __future__ import annotations

from unified_harness.eval.labels import apply_resolutions, dedupe, from_upstream, merge_encounter

RECORD = {
    "encounter_id": "SYN9",
    "dialogue": "[doctor] hi ann lee . how was the golf trip to boulder ?\n[patient] i hurt my wrist golfing .",
    "note": "Ann Lee is a 61-year-old retired pilot. Phalen's test positive. Lives in Boulder.",
}
UPSTREAM = {
    "encounter_id": "SYN9",
    "pii": [
        {"field": "dialogue", "text": "ann", "category": "PATIENT_NAME", "risk": "high"},
        {"field": "dialogue", "text": "lee", "category": "PATIENT_NAME", "risk": "high"},
        {"field": "note", "text": "Ann", "category": "PATIENT_NAME", "risk": "high"},
        {"field": "note", "text": "Lee", "category": "PATIENT_NAME", "risk": "high"},
        {"field": "note", "text": "61-year-old", "category": "AGE", "risk": "high"},
        {"field": "note", "text": "Phalen", "category": "PERSON_NAME", "risk": "high"},
    ],
    "keep": [{"field": "dialogue", "text": "golfing", "category": "CODE_EVIDENCE", "why": "activity code"}],
}
KETAI = {
    "encounter_id": "SYN9",
    "pii": [
        {"field": "dialogue", "text": "ann lee", "category": "PATIENT_NAME", "tier": "direct"},
        {"field": "note", "text": "Ann Lee", "category": "PATIENT_NAME", "tier": "direct"},
        {"field": "note", "text": "retired pilot", "category": "OCCUPATION", "tier": "quasi"},
        {"field": "note", "text": "Boulder", "category": "LOCATION", "tier": "quasi"},
        {"field": "dialogue", "text": "golfing", "category": "OTHER_QUASI", "tier": "quasi"},
    ],
    "medical_concepts": [
        {"field": "note", "text": "Phalen's test", "type": "test", "icd_relevant": True},
        {"field": "dialogue", "text": "wrist", "type": "anatomy", "icd_relevant": True},
    ],
    "ambiguous_terms": [{"field": "note", "text": "Phalen's test", "is_pii": False, "comment": "eponym"}],
}


def _keys(items):
    return {(i["field"], i["text"]) for i in items}


def test_overlapping_strings_count_as_agreement():
    m = merge_encounter(RECORD, UPSTREAM, KETAI)
    pii = {(p["field"], p["text"]): p for p in m["pii"]}
    assert ("dialogue", "ann lee") not in pii and ("note", "Ann Lee") not in pii
    assert pii[("note", "Ann")]["sources"] == ["upstream", "ketai"]
    assert pii[("note", "61-year-old")]["sources"] == ["upstream"]


def test_single_source_strings_are_added_in_the_upstream_vocabulary():
    m = merge_encounter(RECORD, UPSTREAM, KETAI)
    pii = {(p["field"], p["text"]): p for p in m["pii"]}
    assert pii[("note", "retired pilot")] == {
        "field": "note",
        "text": "retired pilot",
        "category": "OCCUPATION",
        "risk": "medium",
        "tier": "quasi",
        "sources": ["ketai"],
    }
    assert pii[("note", "Boulder")]["category"] == "LOCATION"


def test_pii_against_clinical_becomes_a_conflict_in_both_directions():
    m = merge_encounter(RECORD, UPSTREAM, KETAI)
    conflicts = {(c["field"], c["text"]): c for c in m["conflicts"]}
    assert conflicts[("dialogue", "golfing")]["pii_from"] == "ketai"  # upstream keeps it as code evidence
    assert conflicts[("note", "Phalen")]["pii_from"] == "upstream"  # Ketai marks the eponym clinical
    assert ("note", "Phalen") not in _keys(m["pii"]) and ("dialogue", "golfing") not in _keys(m["pii"])


def test_resolutions_move_conflicts_into_pii_or_keep_or_drop_them():
    m = merge_encounter(RECORD, UPSTREAM, KETAI)
    m = apply_resolutions(
        m,
        {
            ("note", "phalen"): {"decision": "keep", "category": "EPONYM"},
            ("dialogue", "golfing"): {"decision": "neither"},
        },
    )
    assert m["conflicts"] == []
    assert ("note", "Phalen") in _keys(m["keep"])
    assert ("dialogue", "golfing") not in _keys(m["pii"])
    assert ("dialogue", "golfing") in _keys(m["keep"])  # the upstream keep trap itself stays


def test_without_ketai_labels_the_upstream_label_is_kept_as_is():
    m = merge_encounter(RECORD, UPSTREAM, None)
    assert _keys(m["pii"]) == _keys(from_upstream(UPSTREAM)["pii"])
    assert all(p["sources"] == ["upstream"] for p in m["pii"])


def test_dedupe_merges_sources_case_insensitively():
    items = [
        {"field": "note", "text": "Boulder", "sources": ["ketai"]},
        {"field": "note", "text": "boulder", "sources": ["upstream"]},
    ]
    assert dedupe(items) == [{"field": "note", "text": "Boulder", "sources": ["ketai", "upstream"]}]
