"""Deterministic renderer: applies the policy's JSON decisions (rl/spec.py) to an encounter.

The model never rewrites text. This module turns its decisions plus the regex pre-pass (generalize.find_spans)
into replacement intervals and splices them in, so `icd10_codes` and every untouched word stay byte-identical.

- Encounter-level linking: an entity without `ctx` replaces every whole-word, case-insensitive occurrence in the
  dialogue and the note. With `ctx`, only occurrences inside a match of `ctx` are replaced.
- Span limits (TRAINING_PLAN.md §2): an entity is rejected if it is not found verbatim, has more than 6 words,
  crosses a line, is a speaker tag, has an unknown type, or collides with protected medical text without `ctx`.
- `keep` vetoes regex spans. A vetoed span is left in the text and logged as retained.
- AGE and DATE are generalized (generalize.render). A quasi identifier with a valid `cat` becomes a category word
  ("mount baker" -> "a mountain"). Everything else becomes [TYPE].

plan() computes all candidate intervals once; apply() renders any subset of decisions, which is what the reward
needs to measure the marginal effect of each decision.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import generalize  # noqa: E402
from anonymize import SPEAKER_TAG, ICD_STOPWORDS, _words  # noqa: E402
from common import norm  # noqa: E402

from spec import MAX_SPAN_WORDS, QUASI_TYPES, norm_type, render_category, short_reason  # noqa: E402

FIELDS = ("dialogue", "note")
UNPROTECTED_TYPES = {"AGE", "DATE", "CONTACT", "ID"}   # as anonymize.is_protected: never kept as diagnosis words


def find_all(needle: str, text: str) -> list[tuple[int, int]]:
    """Whole-word, case-insensitive occurrences of needle in text (the rule of anonymize.apply_entities)."""
    if not needle:
        return []
    pat = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _inside(iv, ivs) -> bool:
    return any(s <= iv[0] and iv[1] <= e for s, e in ivs)


def _overlaps(iv, ivs) -> bool:
    return any(iv[0] < e and s < iv[1] for s, e in ivs)


def terms_of(descs: list[str]) -> frozenset:
    """Content words of diagnosis descriptions (the record's icd10_codes, or the model's own dx in free mode)."""
    return frozenset(w for d in descs for w in _words(str(d)) if len(w) >= 3 and w not in ICD_STOPWORDS)


def is_term(text: str, terms: frozenset) -> bool:
    words = [w for w in _words(text) if w not in ICD_STOPWORDS]
    return bool(words) and all(w in terms or w in ("left", "right") for w in words)


@dataclass
class Decision:
    kind: str                  # "entity" | "regex" | "keep"
    index: int                 # position in the model's list (regex: position in the regex pass)
    text: str
    type: str | None = None
    ctx: str | None = None
    cat: str | None = None
    reason: str = ""
    status: str = "applied"    # applied | rejected:<why> | duplicate | vetoed | noop
    intervals: dict = field(default_factory=dict)    # field -> [(start, end)]

    @property
    def ok(self) -> bool:
        return self.status == "applied"


@dataclass
class Plan:
    record: dict
    decisions: list[Decision]
    protect_terms: frozenset

    def by_kind(self, kind: str) -> list[Decision]:
        return [d for d in self.decisions if d.kind == kind]


def replacement(d: Decision, occurrence: str) -> str:
    return _replacement(d.type, d.cat if d.type in QUASI_TYPES else None, occurrence)


@lru_cache(maxsize=65536)
def _replacement(typ: str, cat, occurrence: str) -> str:
    if cat and (word := render_category(cat)):
        return generalize._match_case(occurrence, word)
    return generalize.render(occurrence, typ)


def plan(record: dict, output: dict | None, *, protect_terms: frozenset | None = None,
         protected: dict | None = None, use_regex: bool = True) -> Plan:
    """All decisions of one encounter with their status and candidate intervals.

    protect_terms: diagnosis words (terms_of the record's code descriptions, or of the model's dx). An entity made
      only of them, or (with `protected`) one whose occurrences lie inside protected medical text, needs `ctx`.
    protected: {field: [(start, end)]} of medical text that must not be touched without ctx (the reward passes the
      silver concept occurrences; inference passes nothing).
    """
    texts = {f: record.get(f) or "" for f in FIELDS}
    output = output if isinstance(output, dict) else {}
    protect_terms = protect_terms or frozenset()
    protected = protected or {}
    decisions: list[Decision] = []

    keeps = []
    for j, k in enumerate(_items(output.get("keep"))):
        t = str(k.get("text", "")).strip()
        keeps.append(Decision("keep", j, t, reason=short_reason(k.get("reason")), status="applied" if t else "rejected:empty"))
    keep_norms = [norm(k.text) for k in keeps if k.ok]

    # regex pre-pass: unambiguous ages and dates, linked across fields like entities
    if use_regex:
        seen = set()
        for f in FIELDS:
            for s in generalize.find_spans(texts[f]):
                key = s["text"].lower()
                if key in seen:
                    continue
                seen.add(key)
                d = Decision("regex", len(seen) - 1, s["text"], type=s["type"], reason="regex")
                n = norm(s["text"])
                if any(n == kn or f" {n} " in f" {kn} " for kn in keep_norms):
                    d.status = "vetoed"
                else:
                    d.intervals = {g: find_all(s["text"], texts[g]) for g in FIELDS}
                decisions.append(d)

    seen = set()
    for i, e in enumerate(_items(output.get("entities"))):
        t = str(e.get("text", "")).strip()
        ctx = str(e.get("ctx", "") or "").strip() or None
        d = Decision("entity", i, t, type=norm_type(e.get("type")), ctx=ctx, cat=e.get("cat"),
                     reason=short_reason(e.get("reason")))
        d.status = _check(d, texts, keep_norms, protect_terms, protected)
        key = (t.lower(), (ctx or "").lower())
        if d.ok and key in seen:
            d.status = "duplicate"
        if d.ok:
            seen.add(key)
        decisions.append(d)
    decisions += keeps
    return Plan(record, decisions, protect_terms)


def _items(x) -> list[dict]:
    return [e for e in x if isinstance(e, dict)] if isinstance(x, list) else []


def _check(d: Decision, texts: dict, keep_norms: list, protect_terms: frozenset, protected: dict) -> str:
    t = d.text
    if d.type is None:
        return "rejected:bad_type"
    if len(t) < 2:
        return "rejected:empty"
    if "\n" in t or (d.ctx and "\n" in d.ctx):
        return "rejected:multiline"
    if len(t.split()) > MAX_SPAN_WORDS:
        return "rejected:too_long"
    if SPEAKER_TAG.fullmatch(t) or re.fullmatch(r"\[?\s*(?:doctor|patient|patient_guest)\s*\]?", t, re.IGNORECASE):
        return "rejected:speaker_tag"
    if norm(t) in keep_norms:
        return "rejected:in_keep"
    if d.ctx:
        if find_all(t, d.ctx) == []:
            return "rejected:ctx_mismatch"
        if len(d.ctx.split()) > 2 * MAX_SPAN_WORDS:
            return "rejected:too_long"
        ivs = {}
        for f in FIELDS:
            ivs[f] = sorted({(cs + s, cs + e) for cs, ce in find_all(d.ctx, texts[f])
                             for s, e in find_all(t, texts[f][cs:ce])})
        if not any(ivs.values()):
            return "rejected:not_found"
        d.intervals = ivs
        return "applied"
    ivs = {f: find_all(t, texts[f]) for f in FIELDS}
    if not any(ivs.values()):
        return "rejected:not_found"
    if d.type not in UNPROTECTED_TYPES:
        if protect_terms and is_term(t, protect_terms):
            return "rejected:needs_ctx"
        if any(_inside(iv, protected.get(f, [])) for f in FIELDS for iv in ivs[f]):
            return "rejected:needs_ctx"
    d.intervals = ivs
    return "applied"


def apply(p: Plan, active=None) -> tuple[dict, dict]:
    """Render the encounter with the applied decisions (or the subset `active`, a set of ids from decision_id).

    Returns ({field: text}, {decision id: {field: [(start, end, replacement)]}}). Overlaps resolve longest first,
    then earliest, so the result does not depend on the order of the decisions.
    """
    texts = {f: p.record.get(f) or "" for f in FIELDS}
    cands = {f: [] for f in FIELDS}
    for d in p.decisions:
        if d.kind == "keep" or not d.ok or (active is not None and decision_id(d) not in active):
            continue
        for f in FIELDS:
            for s, e in d.intervals.get(f, []):
                cands[f].append((s, e, d))
    out, used = {}, {}
    for f in FIELDS:
        taken = []
        for s, e, d in sorted(cands[f], key=lambda c: (-(c[1] - c[0]), c[0], c[2].kind != "regex")):
            if _overlaps((s, e), taken):
                continue
            taken.append((s, e))
            rep = replacement(d, texts[f][s:e])
            used.setdefault(decision_id(d), {}).setdefault(f, []).append((s, e, rep))
        pieces, last = [], 0
        for s, e, rep in sorted(((s, e, r) for u in used.values() for s, e, r in u.get(f, [])), key=lambda x: x[0]):
            pieces += [texts[f][last:s], rep]
            last = e
        pieces.append(texts[f][last:])
        out[f] = "".join(pieces)
    return out, used


def decision_id(d: Decision) -> tuple:
    return (d.kind, d.index)


def explanation_log(p: Plan, used: dict) -> list[dict]:
    """One entry per decision: what happened to the span, the rule that fired, the type and the model's reason."""
    log = []
    for d in p.decisions:
        reps = used.get(decision_id(d), {})
        n = sum(len(v) for v in reps.values())
        entry = {"source": "regex" if d.kind == "regex" else "model", "kind": d.kind, "text": d.text}
        if d.kind == "keep":
            entry.update(action="retained", reason=d.reason or "listed in keep by the model")
        elif d.status == "vetoed":
            entry.update(type=d.type, action="retained", rule="regex span vetoed by the model's keep list")
        elif not d.ok:
            entry.update(type=d.type, action="ignored", rule=d.status, reason=d.reason)
        else:
            first = next((r for v in reps.values() for _, _, r in v), None)
            action = ("removed" if first == f"[{d.type}]" else "retained" if first is None or first == d.text
                      else "category" if d.cat and render_category(d.cat) == (first or "").lower() else "generalized")
            gtype = d.type if d.type in ("AGE", "DATE") else "PERSON" if d.type.endswith("_NAME") else d.type
            entry.update(type=d.type, action=action, replacement=first, occurrences=n,
                         rule=generalize.explain(gtype, "removed" if action == "category" else action),
                         reason=d.reason)
            if d.ctx:
                entry["ctx"] = d.ctx
        log.append(entry)
    return log


def render_record(record: dict, output: dict | None, *, protect_terms: frozenset | None = None) -> tuple[dict, list]:
    """Anonymized copy of the record (icd10_codes and every other field copied) and its explanation log."""
    p = plan(record, output, protect_terms=protect_terms)
    texts, used = apply(p)
    out = {**record, **{f: texts[f] for f in FIELDS if f in record}}
    out["icd10_codes"] = record.get("icd10_codes")
    return out, explanation_log(p, used)
