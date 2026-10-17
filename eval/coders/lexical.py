"""Lexical coder, zero-shot: ranks the label space (every code of the suite and of ACI) by the share of idf-weighted
words of each code's description that occur in the text. Sensitive to exactly the words anonymization may remove.
Ranked: the top 20 codes.
"""

from __future__ import annotations

import math
from collections import Counter

from eval.coders.base import Coder, OnResult, Prediction
from eval.coders.text import record_text, terms
from eval.common import sha1_text


def label_space(*record_lists: list[dict]) -> dict[str, str]:
    """code -> description, first description seen wins."""
    space: dict[str, str] = {}
    for recs in record_lists:
        for r in recs:
            for c in r.get("icd10_codes") or []:
                space.setdefault(c["code"], c["description"])
    return space


class Lexical(Coder):
    ranked = True
    inputs = "both"

    def __init__(self, label_space: dict[str, str], name: str = "lexical"):
        self.name = name
        self.codes = sorted(label_space)
        self.desc = {c: set(terms(label_space[c])) for c in self.codes}
        df = Counter(t for ts in self.desc.values() for t in ts)
        self.idf = {t: math.log((1 + len(self.codes)) / (1 + n)) + 1 for t, n in df.items()}
        self.config = f"lexical:{sha1_text(*(f'{c}={label_space[c]}' for c in self.codes))}"  # unchanged: old cache files stay valid

    def predict(self, records: list[dict], on_result: OnResult = None) -> dict[str, Prediction]:
        out = {}
        for r in records:
            present = set(terms(record_text(r)))
            scored = []
            for c in self.codes:
                ts = self.desc[c]
                if not ts:
                    continue
                total = sum(self.idf[t] for t in ts)
                hit = sum(self.idf[t] for t in ts if t in present)
                scored.append((-(hit / total), -hit, c))
            out[r["encounter_id"]] = Prediction([c for _, _, c in sorted(scored)[:20]])
        return out
