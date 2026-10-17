"""Silver labels: merge the two teams' annotations into one schema.

Sources
  upstream  Alind+Davide format, one file per encounter: {"pii": [{field, text, category, risk}],
            "keep": [{field, text, category, why}]} (labels/upstream/<split>/).
  ketai     Ketai format, one JSONL line per encounter: {"pii": [{field, text, category, tier}],
            "medical_concepts": [...], "ambiguous_terms": [...]} (labels/ketai/eval_gold.jsonl).

Merged schema (labels/merged/<split>/<encounter_id>.json), readable by eval.privacy.score:
  pii        [{field, text, category, risk, tier, sources}]   category and risk in the upstream vocabulary
  keep       [{field, text, category, why, sources}]          strings that must survive anonymization
  concepts   [{field, text, type, icd_relevant}]              Ketai's medical concepts (utility metric)
  conflicts  [{field, text, pii_from, keep_from, ...}]        PII in one source, clinical in the other

Two strings agree when their occurrences overlap in the original field (whole-word, case-insensitive),
so Ketai's full names ("Ann Lee") and upstream's name tokens ("Ann", "Lee") count as one finding. A
string found only by one source is added with that source. Conflicts are left out of both lists until
they are adjudicated (see scripts/labels/merge.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

DIRECT = {"PATIENT_NAME", "PERSON_NAME", "DOB", "ID", "DATE"}
DEFAULT_RISK = {
    "PATIENT_NAME": "high",
    "PERSON_NAME": "high",
    "DOB": "high",
    "ID": "high",
    "DATE": "high",
    "AGE": "high",
    "YEAR": "medium",
    "LOCATION": "medium",
    "ORGANIZATION": "medium",
    "OCCUPATION": "medium",
    "OTHER_QI": "low",
}
KETAI_CATEGORY = {
    "PATIENT_NAME": "PATIENT_NAME",
    "CLINICIAN_NAME": "PERSON_NAME",
    "OTHER_PERSON_NAME": "PERSON_NAME",
    "DATE": "DATE",
    "CONTACT": "ID",
    "ID": "ID",
    "AGE": "AGE",
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORGANIZATION",
    "OCCUPATION": "OCCUPATION",
    "OTHER_QUASI": "OTHER_QI",
}
KEEP_CATEGORIES = {"EPONYM", "CODE_EVIDENCE", "CLINICAL_NUMBER", "NOT_A_NAME"}


def tier(category: str) -> str:
    return "direct" if category in DIRECT else "quasi"


def pattern(text: str) -> re.Pattern:
    """Whole-word, case-insensitive pattern (same matching rule as eval.privacy)."""
    left = r"(?<!\w)" if text[:1].isalnum() or text[:1] == "_" else ""
    right = r"(?!\w)" if text[-1:].isalnum() or text[-1:] == "_" else ""
    return re.compile(left + re.escape(text) + right, re.IGNORECASE)


def occurrences(text: str, field_text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern(text).finditer(field_text)]


def _overlaps(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> bool:
    b = list(b)
    return any(s < e2 and s2 < e for s, e in a for s2, e2 in b)


def _covered(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    """Every occurrence in `a` overlaps some occurrence in `b`."""
    return bool(a) and all(any(s < e2 and s2 < e for s2, e2 in b) for s, e in a)


def load_upstream(folder: Path) -> dict[str, dict]:
    return {
        json.loads(p.read_text(encoding="utf-8"))["encounter_id"]: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(folder).glob("*.json"))
    }


def load_ketai(path: Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["encounter_id"]] = rec
    return out


def from_upstream(label: dict, source: str = "upstream") -> dict:
    """An upstream-format label as a merged record (single source)."""
    return {
        "encounter_id": label["encounter_id"],
        "pii": [
            {
                **p,
                "risk": p.get("risk") or DEFAULT_RISK[p["category"]],
                "tier": tier(p["category"]),
                "sources": [source],
            }
            for p in label.get("pii", [])
        ],
        "keep": [{**k, "sources": [source]} for k in label.get("keep", [])],
        "concepts": [],
        "conflicts": [],
    }


def merge_encounter(record: dict, upstream: dict | None, ketai: dict | None) -> dict:
    """Union of both sources for one encounter, with agreement marked in `sources`."""
    eid = record["encounter_id"]
    merged = from_upstream(upstream) if upstream else {"encounter_id": eid, "pii": [], "keep": [], "conflicts": []}
    merged["concepts"] = list((ketai or {}).get("medical_concepts", []))
    if not ketai:
        return merged

    def occ(item: dict) -> list[tuple[int, int]]:
        return occurrences(item["text"], record[item["field"]])

    up_pii = [(p, occ(p)) for p in merged["pii"]]
    up_keep = [(k, occ(k)) for k in merged["keep"]]
    ketai_clinical = [t for t in ketai.get("medical_concepts", [])] + [
        t for t in ketai.get("ambiguous_terms", []) if not t.get("is_pii")
    ]
    clinical_occ = [(t, occ(t)) for t in ketai_clinical if t.get("field") in ("dialogue", "note")]

    # upstream PII that Ketai marked as clinical -> conflict
    kept_pii = []
    for p, o in up_pii:
        hits = [t for t, to in clinical_occ if t["field"] == p["field"] and _overlaps(o, to)]
        if hits:
            merged["conflicts"].append(
                {
                    "field": p["field"],
                    "text": p["text"],
                    "pii_from": "upstream",
                    "pii_category": p["category"],
                    "keep_from": "ketai",
                    "clinical_text": hits[0]["text"],
                }
            )
        else:
            kept_pii.append((p, o))
    up_pii = kept_pii

    added = []
    for k in ketai.get("pii", []):
        if k.get("field") not in ("dialogue", "note"):
            continue
        category = KETAI_CATEGORY.get(str(k.get("category", "")).upper())
        if category is None:
            continue
        o = occ(k)
        if not o:
            continue
        same_field_pii = [(p, po) for p, po in up_pii if p["field"] == k["field"]]
        keep_hits = [kp for kp, ko in up_keep if kp["field"] == k["field"] and _overlaps(o, ko)]
        if keep_hits:
            merged["conflicts"].append(
                {
                    "field": k["field"],
                    "text": k["text"],
                    "pii_from": "ketai",
                    "pii_category": category,
                    "keep_from": "upstream",
                    "keep_category": keep_hits[0]["category"],
                    "clinical_text": keep_hits[0]["text"],
                }
            )
            continue
        for p, po in same_field_pii:
            if _overlaps(o, po) and "ketai" not in p["sources"]:
                p["sources"].append("ketai")
        if _covered(o, [s for _, po in same_field_pii for s in po]):
            continue
        added.append(
            {
                "field": k["field"],
                "text": k["text"],
                "category": category,
                "risk": DEFAULT_RISK[category],
                "tier": tier(category),
                "sources": ["ketai"],
            }
        )
    merged["pii"] = dedupe([p for p, _ in up_pii] + added)

    # Ketai eponyms (ambiguous, not PII, also a medical concept) become keep traps when new
    concepts = {(c["field"], c["text"].lower()) for c in ketai.get("medical_concepts", [])}
    for t in ketai.get("ambiguous_terms", []):
        if t.get("is_pii") or t.get("field") not in ("dialogue", "note"):
            continue
        if not any(f == t["field"] and (t["text"].lower() in c or c in t["text"].lower()) for f, c in concepts):
            continue
        o = occ(t)
        if not o or any(_overlaps(o, ko) for kp, ko in up_keep if kp["field"] == t["field"]):
            continue
        if any(_overlaps(o, po) for p, po in up_pii if p["field"] == t["field"]):
            continue  # already recorded as a conflict from the PII side
        merged["keep"].append(
            {
                "field": t["field"],
                "text": t["text"],
                "category": "EPONYM",
                "why": t.get("comment", ""),
                "sources": ["ketai"],
            }
        )
    return merged


def dedupe(items: list[dict]) -> list[dict]:
    """One item per (field, lower-cased text); later duplicates only add their sources."""
    seen: dict[tuple[str, str], dict] = {}
    for item in items:
        key = (item["field"], item["text"].lower())
        if key in seen:
            seen[key]["sources"] = sorted(set(seen[key]["sources"]) | set(item["sources"]))
        else:
            seen[key] = item
    return list(seen.values())


def apply_resolutions(merged: dict, decisions: dict[tuple[str, str], dict]) -> dict:
    """Move adjudicated conflicts into `pii` or `keep` (or drop them); unresolved ones stay listed."""
    still = []
    for c in merged["conflicts"]:
        d = decisions.get((c["field"], c["text"].lower()))
        if d is None:
            still.append(c)
            continue
        if d["decision"] == "pii":
            category = d.get("category") or c["pii_category"]
            merged["pii"].append(
                {
                    "field": c["field"],
                    "text": c["text"],
                    "category": category,
                    "risk": DEFAULT_RISK[category],
                    "tier": tier(category),
                    "sources": [c["pii_from"], "adjudicated"],
                }
            )
        elif d["decision"] == "keep":
            category = d.get("category") if d.get("category") in KEEP_CATEGORIES else "NOT_A_NAME"
            merged["keep"].append(
                {
                    "field": c["field"],
                    "text": c["text"],
                    "category": category,
                    "why": d.get("why", ""),
                    "sources": [c["keep_from"], "adjudicated"],
                }
            )
    merged["conflicts"] = still
    merged["pii"], merged["keep"] = dedupe(merged["pii"]), dedupe(merged["keep"])
    return merged


def write_merged(folder: Path, merged: Iterable[dict]) -> int:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for m in merged:
        (folder / f"{m['encounter_id']}.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
        n += 1
    return n


def ketai_as_gold(rec: dict) -> dict:
    """A raw Ketai label record in scorer form, keeping its own categories and tiers (cross-checks)."""
    return {
        "encounter_id": rec["encounter_id"],
        "pii": [
            {**p, "risk": DEFAULT_RISK[KETAI_CATEGORY[p["category"]]], "tier": p.get("tier") or "direct"}
            for p in rec.get("pii", [])
        ],
        "keep": [],
        "concepts": rec.get("medical_concepts", []),
    }
