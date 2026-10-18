"""kNN coder: TF-IDF cosine k-nearest neighbours over the 140 ACI train encounters (original text, gold codes).

Only clinical words count (the ICD-10-CM description vocabulary), so removing names and places does not move it. It
is supervised: it can only predict train codes (146 of them, 33 of the 112 distinct eval codes). A record never
retrieves itself (leave-one-out on aci_dev). Ranked: the top 20 codes by summed neighbour similarity.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from eval.coders.base import Coder, OnResult, Prediction
from eval.coders.text import aci, clinical_stems, record_text, terms
from eval.common import sha1_text


class Knn(Coder):
    ranked = True
    inputs = "both"

    def __init__(self, k: int = 7, name: str = "knn", train: list[dict] | None = None, vocab: frozenset[str] | None = None):
        self.name, self.k = name, k
        train = aci()[0] if train is None else train
        self.ids = [e["encounter_id"] for e in train]
        self.codes = [[c["code"] for c in e["icd10_codes"]] for e in train]
        self.vocab = clinical_stems() if vocab is None else vocab
        tfs = [Counter(terms(record_text(e), self.vocab)) for e in train]
        df = Counter(t for tf in tfs for t in tf)
        self.idf = {t: math.log((1 + len(tfs)) / (1 + n)) + 1 for t, n in df.items()}
        self.vecs = [self._vec(tf) for tf in tfs]
        self.config = f"knn:k={k}:clinical:train={sha1_text(*self.ids)}"  # unchanged from scripts/eval: old cache files stay valid

    def _vec(self, tf: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(n)) * self.idf[t] for t, n in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def predict(self, records: list[dict], on_result: OnResult = None) -> dict[str, Prediction]:
        out = {}
        for r in records:
            q = self._vec(Counter(terms(record_text(r), self.vocab)))
            sims = sorted(((sum(w * v.get(t, 0.0) for t, w in q.items()), i) for i, v in enumerate(self.vecs) if self.ids[i] != r["encounter_id"]), reverse=True)
            score: dict[str, float] = defaultdict(float)
            for s, i in sims[: self.k]:
                for c in self.codes[i]:
                    score[c] += s
            out[r["encounter_id"]] = Prediction([c for c, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:20])
        return out
