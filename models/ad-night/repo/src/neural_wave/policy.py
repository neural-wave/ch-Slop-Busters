"""Deterministic policy: each detected span -> action + replacement text.

Typed placeholders by default (the organizers' own example uses `[PERSON]`, `[YEAR]`), and
generalization where the class keeps clinical meaning (ages -> decade band, infants kept as infants).
Honorifics are kept (`mr . [PATIENT]`) because sex can matter for coding.
"""

from __future__ import annotations

from .config import Config
from .identity import Entity
from .spans import Span

PLACEHOLDER = {
    "PATIENT_NAME": "[PATIENT]",
    "CLINICIAN_NAME": "[CLINICIAN]",
    "STAFF_NAME": "[STAFF]",
    "RELATIVE_NAME": "[RELATIVE]",
    "PERSON_NAME": "[PERSON]",
    "DOB": "[DOB]",
    "ID": "[ID]",
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "URL": "[URL]",
    "ADDRESS": "[ADDRESS]",
    "DATE": "[DATE]",
    "YEAR": "[YEAR]",
    "LOCATION": "[LOCATION]",
    "FACILITY": "[FACILITY]",
    "ORGANIZATION": "[ORGANIZATION]",
    "INSURER": "[INSURER]",
    "SPORTS_TEAM": "[SPORTS_TEAM]",
    "OCCUPATION": "[OCCUPATION]",
    "HOLIDAY": "[HOLIDAY]",
    "ACTIVITY": "[ACTIVITY]",
    "OTHER_QI": "[DETAIL]",
}

ACTION = {"AGE": "GENERALIZE", "YEAR": "GENERALIZE", "OCCUPATION": "GENERALIZE", "LOCATION": "GENERALIZE",
          "ORGANIZATION": "GENERALIZE", "FACILITY": "GENERALIZE", "INSURER": "GENERALIZE",
          "SPORTS_TEAM": "GENERALIZE", "HOLIDAY": "GENERALIZE", "ACTIVITY": "GENERALIZE", "OTHER_QI": "GENERALIZE"}  # fmt: skip

WHY = {
    "PATIENT_NAME": "direct identifier (patient name)",
    "CLINICIAN_NAME": "direct identifier (clinician name links the record to a practice)",
    "STAFF_NAME": "direct identifier (staff name)",
    "RELATIVE_NAME": "direct identifier (relative's name)",
    "PERSON_NAME": "direct identifier (person name)",
    "DOB": "direct identifier (date of birth); age kept as a band elsewhere",
    "ID": "direct identifier (record / account number)",
    "PHONE": "direct identifier (phone)",
    "EMAIL": "direct identifier (e-mail)",
    "URL": "direct identifier (URL)",
    "ADDRESS": "direct identifier (address)",
    "DATE": "absolute date links to the visit; relative timing is kept",
    "YEAR": "calendar year narrows identity; durations and 'N years ago' are kept",
    "AGE": "exact age is a strong quasi-identifier; band keeps age-dependent coding",
    "LOCATION": "place narrows identity",
    "FACILITY": "named facility narrows identity",
    "ORGANIZATION": "employer / company narrows identity",
    "INSURER": "insurer narrows identity",
    "SPORTS_TEAM": "team fandom reveals region",
    "OCCUPATION": "occupation is a quasi-identifier",
    "HOLIDAY": "named holiday dates the visit",
    "ACTIVITY": "hobby is a rare-word quasi-identifier (not code evidence here)",
    "OTHER_QI": "rare personal detail (quasi-identifier)",
}


def age_band(value: int, unit: str, mode: str) -> tuple[str, bool]:
    """Return (replacement, whole_expression) for an age."""
    unit = (unit or "year").lower()
    if unit.startswith(("month", "week", "day")):
        months = value if unit.startswith("month") else value / 4.3 if unit.startswith("week") else value / 30
        return ("[infant]" if months < 12 else "[toddler]" if months < 36 else "[child]"), True
    if value >= 90:
        return "[90+]", False
    if mode == "coarse":
        return ("[adult]" if value >= 18 else "[minor]"), False
    if value < 2:
        return "[infant]", False
    if value < 13:
        return "[child]", False
    if value < 18:
        return "[teen]", False
    return f"[{value // 10 * 10}s]", False


def entity_labels(ents: list[Entity]) -> None:
    """Assign placeholders: one `[PATIENT]`; numbered labels when a role has several people."""
    by_role: dict[str, list[Entity]] = {}
    for e in ents:
        by_role.setdefault(e.role, []).append(e)
    for role, group in by_role.items():
        base = {"patient": "PATIENT", "clinician": "CLINICIAN", "staff": "STAFF", "relative": "RELATIVE"}.get(role, "PERSON")
        if role == "patient" or len(group) == 1:
            for e in group:
                e.label = f"[{base}]"
        else:
            for i, e in enumerate(sorted(group, key=lambda x: x.key), 1):
                e.label = f"[{base}_{i}]"


def replacement(span: Span, cfg: Config, ents: dict[str, Entity]) -> tuple[int, int, str, str]:
    """(start, end, replacement, action) for one span."""
    if span.cat == "AGE" and span.value is not None:
        rep, whole = age_band(span.value, span.meta.get("unit", "year"), cfg.age_mode)
        if whole or "num_start" not in span.meta:
            return span.start, span.end, rep, "GENERALIZE"
        return span.meta["num_start"], span.meta["num_end"], rep, "GENERALIZE"
    if span.entity and span.entity in ents and ents[span.entity].label:
        return span.start, span.end, ents[span.entity].label, "REPLACE"
    return span.start, span.end, PLACEHOLDER.get(span.cat, "[REDACTED]"), ACTION.get(span.cat, "REPLACE")


def apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping (start, end, replacement) edits; adjacent same placeholders merge."""
    edits = sorted(edits)
    merged: list[list] = []
    for s, e, r in edits:
        if merged and merged[-1][2] == r and r.startswith("[") and text[merged[-1][1] : s].strip(" \t") == "" and s - merged[-1][1] <= 2:
            merged[-1][1] = e  # "[PATIENT] [PATIENT]" -> "[PATIENT]"
        else:
            merged.append([s, e, r])
    out, pos = [], 0
    for s, e, r in merged:
        out.append(text[pos:s])
        out.append(r)
        pos = e
    out.append(text[pos:])
    return "".join(out)
