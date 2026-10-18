"""Verify before writing: cross-field leak check, shield integrity, residual scan.

A failed check never reverts an edit (that could resurrect PHI); it triggers an additional
anonymizing edit (`repair`) or is reported.
"""

from __future__ import annotations

import re

from . import lexicons as L
from .spans import NAME_CATS, Shield, Span
from .text import normalize

# Only direct identifiers are propagated by the leak repair (quasi-identifier detections are lower
# precision; propagating them would multiply false positives such as a clinical phrase).
LEAK_CATS = set(NAME_CATS) | {"DOB", "ID", "PHONE", "EMAIL", "URL", "ADDRESS", "DATE"}


def _wb(s: str) -> re.Pattern:
    left = r"(?<![a-z0-9])" if s[:1].isalnum() else ""
    right = r"(?![a-z0-9])" if s[-1:].isalnum() else ""
    return re.compile(left + re.escape(s) + right)


def removed_strings(applied: list[Span], raw: dict[str, str]) -> dict[str, Span]:
    """Distinct removed surface strings (lowercase) worth checking for leaks."""
    out: dict[str, Span] = {}
    for s in applied:
        if s.cat not in LEAK_CATS:
            continue
        txt = normalize(raw[s.field][s.start : s.end]).lower().strip(" .,'")
        txt = txt.removesuffix("'s")
        if len(txt) < 3 or L.is_common_word(txt) or txt in L.NAME_STOP:
            continue
        out.setdefault(txt, s)
    return out


def find_leaks(applied: list[Span], raw: dict[str, str], anon: dict[str, str]) -> list[tuple[str, str, Span]]:
    """(field, string, example_span) for removed strings still present in any output field."""
    leaks = []
    for txt, sp in removed_strings(applied, raw).items():
        rx = _wb(txt)
        for f, out in anon.items():
            if rx.search(normalize(out).lower()):
                leaks.append((f, txt, sp))
    return leaks


def repair_spans(leaks, raw: dict[str, str], covered: dict[str, list[tuple[int, int]]], shields: list[Shield]) -> list[Span]:
    """Locate leaked strings in the original text (outside existing edits and hard shields)."""
    new = []
    for f, txt, sp in leaks:
        low = "".join(c.lower() if len(c.lower()) == 1 else c for c in normalize(raw[f]))
        for m in _wb(txt).finditer(low):
            s, e = m.start(), m.end()
            if any(a < e and s < b for a, b in covered.get(f, [])):
                continue
            if any(sh.field == f and sh.hard and sh.start < e and s < sh.end for sh in shields):
                continue
            new.append(Span(f, s, e, sp.cat, "verify:leak_repair", sp.conf, f"removed elsewhere in the encounter ({sp.cat}); cross-field leak repaired", entity=sp.entity))
    return new


def shield_integrity(shields: list[Shield], edits: dict[str, list[tuple[int, int, str]]], raw: dict[str, str], anon: dict[str, str]) -> dict:
    """Every hard-shielded span must be untouched; report violations (should always be 0)."""
    violations = 0
    for sh in shields:
        if not sh.hard:
            continue
        for s, e, _ in edits.get(sh.field, []):
            if s < sh.end and sh.start < e:
                violations += 1
    # Independent string-level check: every shielded string still occurs at least as often.
    missing = 0
    by_field: dict[str, dict[str, int]] = {}
    for sh in shields:
        txt = raw[sh.field][sh.start : sh.end]
        by_field.setdefault(sh.field, {})
        by_field[sh.field][txt] = by_field[sh.field].get(txt, 0) + 1
    for f, counts in by_field.items():
        for txt, n in counts.items():
            if anon[f].count(txt) < min(n, raw[f].count(txt)):
                missing += 1
    return {"shield_spans": len(shields), "edit_overlaps": violations, "missing_strings": missing}
