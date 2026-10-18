"""Shared types and helpers for the anonymization pipeline."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

FIELDS = ("dialogue", "note")

# Tier-1 = direct identifiers, tier-2 = quasi-identifiers / rare non-medical context.
TIER = {
    "NAME_PATIENT": 1, "NAME_RELATIVE": 1, "NAME_CLINICIAN": 1, "NAME_OTHER": 1, "AGE": 1, "DATE": 1,
    "LOCATION": 1, "ORGANIZATION": 1, "CONTACT": 1, "ID": 1,
    "OCCUPATION": 2, "ACTIVITY": 2, "LIFE_EVENT": 2, "TEMPORAL": 2, "PERSONAL_DETAIL": 2,
}
NAME_TYPES = {"NAME_PATIENT", "NAME_RELATIVE", "NAME_CLINICIAN", "NAME_OTHER"}

# Product names of ambient scribes / ASR systems that naive NER mistakes for person names.
NEVER_PHI = {"dax", "dragon", "nuance", "epic", "doctor", "patient", "nurse", "covid", "hey", "hi", "hello"}


@dataclass
class Span:
    field: str
    start: int
    end: int
    text: str
    type: str
    source: str               # regex | names | ner | gliner | llm | propagation
    confidence: float = 1.0
    reason: str = ""
    generalized: str = ""     # suggested replacement for tier-2 items (from the LLM)
    meta: dict = field(default_factory=dict)

    @property
    def tier(self) -> int:
        return TIER.get(self.type, 2)

    def overlaps(self, other: "Span") -> bool:
        return self.field == other.field and self.start < other.end and other.start < self.end

    def to_dict(self) -> dict:
        return asdict(self)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_all(text: str, needle: str, flags=re.IGNORECASE):
    """All word-bounded occurrences of needle (case-insensitive) as (start, end)."""
    if not needle.strip():
        return []
    pat = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pat, text, flags)]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()
