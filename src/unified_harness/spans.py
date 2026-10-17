"""Span and shield data model shared by detectors, policy, verification and explanations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Privacy categories emitted by detectors (also the keys of the policy table).
NAME_CATS = ("PATIENT_NAME", "CLINICIAN_NAME", "STAFF_NAME", "RELATIVE_NAME", "PERSON_NAME")
DIRECT_CATS = NAME_CATS + ("DOB", "ID", "PHONE", "EMAIL", "URL", "ADDRESS", "DATE", "AGE")
QUASI_CATS = (
    "YEAR", "LOCATION", "FACILITY", "ORGANIZATION", "INSURER", "SPORTS_TEAM", "OCCUPATION",
    "HOLIDAY", "ACTIVITY", "OTHER_QI",
)  # fmt: skip

# Higher wins when two candidate spans overlap with equal length.
PRIORITY = {
    "DOB": 100, "ID": 95, "EMAIL": 95, "PHONE": 95, "URL": 95, "ADDRESS": 90, "DATE": 85, "AGE": 80,
    "PATIENT_NAME": 75, "CLINICIAN_NAME": 74, "STAFF_NAME": 73, "RELATIVE_NAME": 72, "PERSON_NAME": 70,
    "FACILITY": 60, "INSURER": 58, "ORGANIZATION": 57, "SPORTS_TEAM": 56, "LOCATION": 55, "YEAR": 50,
    "OCCUPATION": 45, "HOLIDAY": 40, "ACTIVITY": 30, "OTHER_QI": 20,
}  # fmt: skip


@dataclass
class Span:
    """A candidate piece of private information in one field (offsets into the original text)."""

    field: str
    start: int
    end: int
    cat: str
    source: str  # detector + rule id, e.g. "rule:title_name"
    conf: float = 1.0
    reason: str = ""
    entity: str | None = None  # identity-graph key for person names
    value: int | None = None  # parsed number for ages / years
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, s: int, e: int) -> bool:
        return self.start < e and s < self.end


@dataclass
class Shield:
    """A locked clinical span: no anonymization edit may touch it."""

    field: str
    start: int
    end: int
    kind: str  # clinical_number | eponym | code_evidence | drug | activity_context
    reason: str
    hard: bool = True


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Non-overlapping edit spans: longest first, then category priority, then confidence.

    A lower-ranked span that only *partially* overlaps the chosen ones is not dropped (its uncovered
    part would stay in the text, e.g. a phone number half-covered by a false address): the winner is
    widened to the union instead, keeping the winner's category. Copies are made because candidate
    lists are reused across repair rounds.
    """
    order = sorted(spans, key=lambda s: (-(s.end - s.start), -PRIORITY.get(s.cat, 0), -s.conf, s.start))
    chosen: list[Span] = []
    for s in order:
        hits = [c for c in chosen if c.field == s.field and c.overlaps(s.start, s.end)]
        if not hits:
            chosen.append(s)
            continue
        lo, hi = min([s.start] + [c.start for c in hits]), max([s.end] + [c.end for c in hits])
        if len(hits) == 1 and (lo, hi) == (hits[0].start, hits[0].end):
            continue  # fully contained
        w = replace(hits[0], start=lo, end=hi, meta=dict(hits[0].meta))
        w.meta.pop("num_start", None)  # an age widened beyond its number is replaced as a whole
        w.meta.pop("num_end", None)
        chosen = [c for c in chosen if c not in hits] + [w]
    return sorted(chosen, key=lambda s: (s.field, s.start))
