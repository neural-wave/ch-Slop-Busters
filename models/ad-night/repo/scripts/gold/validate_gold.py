"""Validate gold annotation files against the source encounters.

Usage:
    uv run python scripts/gold/validate_gold.py outputs/gold/eval/D2N004.json [...]
    uv run python scripts/gold/validate_gold.py outputs/gold/eval   # whole directory
    uv run python scripts/gold/validate_gold.py --data=outputs/robustness/synthetic.jsonl outputs/gold/robust_synthetic

Checks: JSON shape, known categories, every `text` occurs (case-insensitive, whole word) in the
declared field of the encounter. Prints errors and a per-file summary; exit code 1 on any error.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = [
    ROOT / "resources/data/patient_doctor_encounters.jsonl",
    ROOT / "resources/data/eval_patient_doctor_encounters.jsonl",
]
PII_CATS = {
    "PATIENT_NAME", "PERSON_NAME", "DOB", "ID", "DATE", "AGE", "YEAR",
    "LOCATION", "ORGANIZATION", "OCCUPATION", "OTHER_QI",
}
KEEP_CATS = {"EPONYM", "CODE_EVIDENCE", "CLINICAL_NUMBER", "NOT_A_NAME"}
RISKS = {"high", "medium", "low"}


def pattern(text: str) -> re.Pattern:
    """Whole-word, case-insensitive pattern for an annotated surface string."""
    left = r"(?<!\w)" if text[:1].isalnum() or text[:1] == "_" else ""
    right = r"(?!\w)" if text[-1:].isalnum() or text[-1:] == "_" else ""
    return re.compile(left + re.escape(text) + right, re.IGNORECASE)


def load_encounters(extra: list[Path] | None = None) -> dict[str, dict]:
    recs = {}
    for p in DATA + (extra or []):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs[r["encounter_id"]] = r
    return recs


def validate(path: Path, recs: dict[str, dict]) -> list[str]:
    errs = []
    try:
        g = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"{path.name}: invalid JSON ({e})"]
    eid = g.get("encounter_id")
    if eid not in recs:
        return [f"{path.name}: unknown encounter_id {eid!r}"]
    rec = recs[eid]
    for kind, cats in (("pii", PII_CATS), ("keep", KEEP_CATS)):
        items = g.get(kind)
        if not isinstance(items, list):
            errs.append(f"{eid}: '{kind}' must be a list")
            continue
        seen = set()
        for i, it in enumerate(items):
            f, t, c = it.get("field"), it.get("text"), it.get("category")
            where = f"{eid} {kind}[{i}] {t!r}"
            if f not in ("dialogue", "note"):
                errs.append(f"{where}: bad field {f!r}")
                continue
            if not t or not isinstance(t, str):
                errs.append(f"{where}: empty text")
                continue
            if c not in cats:
                errs.append(f"{where}: bad category {c!r} (allowed {sorted(cats)})")
            if kind == "pii" and it.get("risk", "high") not in RISKS:
                errs.append(f"{where}: bad risk {it.get('risk')!r}")
            n = len(pattern(t).findall(rec[f]))
            if n == 0:
                errs.append(f"{where}: not found (whole word, case-insensitive) in {f}")
            key = (f, t.lower())
            if key in seen:
                errs.append(f"{where}: duplicate item")
            seen.add(key)
    return errs


def main(argv: list[str]) -> int:
    extra = [Path(a.split("=", 1)[1]) for a in argv if a.startswith("--data=")]
    argv = [a for a in argv if not a.startswith("--data=")]
    recs = load_encounters(extra)
    paths: list[Path] = []
    for a in argv or [str(ROOT / "outputs/gold")]:
        p = Path(a)
        paths += sorted(q for q in p.rglob("*.json") if q.name != "dev_ids.json") if p.is_dir() else [p]
    total = 0
    for p in paths:
        errs = validate(p, recs)
        total += len(errs)
        for e in errs:
            print("ERROR", e)
    print(f"validated {len(paths)} file(s), {total} error(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
