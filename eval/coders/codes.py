"""ICD-10-CM code normalization: one canonical spelling per code, so predictions and gold compare as exact sets.

Adapted from daniel-eval/src/icd_evaluation/data.py (normalize_codes): formatting only (upper case, canonical dot
after position 3), no vocabulary filtering (a syntactically valid hallucination stays a false positive) and no
hierarchy or prefix matching (I11.0 and I11.9 differ). Changes: an invalid code gives None instead of raising, so one
bad code does not discard a whole prediction; characters other than letters, digits and dots are dropped first
(` e11.9,` and `(E11.9)` are E11.9), but a dot in the wrong place stays invalid (`I.10`).

Usage:
    normalize("i110") -> "I11.0";  normalize("made up") -> None
    normalize_set(["I10", {"code": "I10"}, "E11.9"]) -> ["I10", "E11.9"]   (first-seen order, invalid dropped)
"""

from __future__ import annotations

import re

SYNTAX = re.compile(r"[A-Z][0-9][A-Z0-9](?:\.?[A-Z0-9]{1,4})?")
JUNK = re.compile(r"[^A-Z0-9.]")


def normalize(value) -> str | None:
    """The canonical code (`E11.9`, `I10`), or None when `value` is not a code string or `{"code": ...}` object."""
    if isinstance(value, dict):
        value = value.get("code")
    if not isinstance(value, str):
        return None
    code = JUNK.sub("", value.upper()).strip(".")
    if not SYNTAX.fullmatch(code):
        return None
    code = code.replace(".", "")
    return code[:3] + ("." + code[3:] if len(code) > 3 else "")


def normalize_set(values) -> list[str]:
    """Canonical codes without duplicates, in first-seen order (a ranked list stays ranked); invalid ones dropped."""
    out: dict[str, None] = {}
    for v in values if isinstance(values, (list, tuple)) else []:
        if (c := normalize(v)) is not None:
            out.setdefault(c)
    return list(out)


def count_invalid(values) -> int:
    """How many entries normalize_set drops as invalid (a non-list counts as one)."""
    if not isinstance(values, (list, tuple)):
        return 1
    return sum(normalize(v) is None for v in values)


def category(code: str) -> str:
    """The 3-character category: E11.9 -> E11."""
    return code[:3]
