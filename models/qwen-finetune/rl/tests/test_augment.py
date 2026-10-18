"""Invariance tests of the augmentations (TRAINING_PLAN.md §5): every train encounter x 20 seeds, no model.

  python3 -m unittest rl/tests/test_augment.py
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parents[3] / "term-generalizer")]
import generalize as G  # noqa: E402
from common import count_mentions, norm  # noqa: E402

import augment as A  # noqa: E402
from data import TRAIN_DATA, TRAIN_GOLD, load  # noqa: E402
from render import find_all  # noqa: E402

RECS, GOLDS = load(TRAIN_DATA, TRAIN_GOLD)
POOLS = A.build_pools(RECS, GOLDS)
SEEDS = 20
FIELDS = ("dialogue", "note")


def counts(rec, items) -> dict:
    n = {f: norm(rec.get(f) or "") for f in FIELDS}
    return {(e["field"], norm(e["text"])): count_mentions(e["text"], n[e["field"]]) for e in items}


def spans(g):
    """Medical text: concepts, non-PII ambiguous terms, evidence spans."""
    out = list(g.get("medical_concepts", [])) + [t for t in g.get("ambiguous_terms", []) if not t.get("is_pii")]
    return out + [s for ev in g.get("evidence", []) or [] for s in ev.get("spans", [])]


def runs(**kw):
    for eid in sorted(GOLDS):
        for seed in range(SEEDS):
            rng = random.Random(f"{eid}:{seed}")
            yield eid, seed, A.augment(RECS[eid], GOLDS[eid], rng, POOLS, p_layout=0, p_window=0, **kw)


class TestInvariance(unittest.TestCase):
    def test_medical_text_and_codes_unchanged(self):
        bad = []
        for eid, seed, (rec, g, meta) in runs():
            if rec["icd10_codes"] != RECS[eid]["icd10_codes"]:
                bad.append((eid, seed, "codes"))
            old, new = counts(RECS[eid], spans(GOLDS[eid])), counts(rec, spans(g))
            diff = [(k, old.get(k), v) for k, v in new.items() if old.get(k, v) != v]
            diff += [(k, v, None) for k, v in old.items() if v and k not in new]
            if diff:
                bad.append((eid, seed, "concept counts", diff[:3], meta["ops"]))
        # a few silver spans start with punctuation (". five centimeter ..."), which de-spacing cannot keep verbatim
        self.assertLessEqual(len(bad), len(GOLDS) * SEEDS // 200, bad[:5])

    def test_pii_replaced_with_recorded_counts(self):
        bad = []
        for eid, seed, (rec, g, meta) in runs(p_case=0, p_punct=0, p_tags=0):
            old_rec = RECS[eid]
            cx_protected = A.Crossover(old_rec, GOLDS[eid], random.Random(0), POOLS).protected
            surrogates = {e["text"].lower() for e in g["pii"] if e["text"] != e["orig_text"]}
            for e in g["pii"]:
                f, old_t, new_t = e["field"], e["orig_text"], e["text"]
                occ = find_all(old_t, old_rec[f])
                free = [iv for iv in occ if not any(iv[0] < pe and ps < iv[1] for ps, pe in cx_protected[f])]
                if not free:
                    continue
                # skip: "twenty twenty" -> "twenty twenty three"; "2019" that is the surrogate of "2020" (shift -1)
                if old_t.lower() != new_t.lower() and not find_all(old_t, new_t) and old_t.lower() not in surrogates:
                    # the original string survives only where it overlaps medical text
                    left = count_mentions(old_t, norm(rec[f]))
                    if left > len(occ) - len(free):
                        bad.append((eid, seed, "original survives", old_t, new_t))
                n_new = count_mentions(new_t, norm(rec[f]))
                if n_new < len(free):
                    bad.append((eid, seed, "count", old_t, new_t, len(free), n_new))
        self.assertLessEqual(len(bad), len(GOLDS) * SEEDS // 100, bad[:10])   # <= 1% of augmentations

    def test_sex_and_age_band(self):
        bad = []
        for eid, seed, (rec, g, meta) in runs(p_case=0, p_punct=0, p_tags=0):
            if A.patient_sex(rec) != A.patient_sex(RECS[eid]):
                bad.append((eid, seed, "sex"))
            for e in g["pii"]:
                if e["category"] == "AGE" and e["text"] != e["orig_text"]:
                    if G.generalize_age(e["text"]) != G.generalize_age(e["orig_text"]):
                        bad.append((eid, seed, e["orig_text"], e["text"]))
        self.assertEqual(bad, [], bad[:10])

    def test_spans_locatable_after_style(self):
        bad = []
        for eid, seed, (rec, g, meta) in runs(p_case=1, p_punct=1, p_tags=1):
            for e in g["pii"] + spans(g):
                if not find_all(e["text"], rec.get(e["field"]) or ""):
                    bad.append((eid, seed, e["field"], e["text"], meta["ops"]))
        self.assertLessEqual(len(bad), 10, bad[:10])

    def test_window_and_layout_restrict_gold(self):
        for eid in sorted(GOLDS)[:30]:
            rec, g, meta = A.augment(RECS[eid], GOLDS[eid], random.Random(eid), POOLS, p_layout=1, p_window=1)
            for e in g["pii"] + spans(g):
                self.assertTrue(find_all(e["text"], rec.get(e["field"]) or ""), (eid, e))
            self.assertTrue(set(meta["sections"]) <= set(FIELDS))
            for k in meta["known"]:
                self.assertIn(k["type"], {"PATIENT_NAME", "CLINICIAN_NAME", "OTHER_PERSON_NAME", "DATE", "CONTACT",
                                          "ID", "AGE", "LOCATION", "ORGANIZATION", "OCCUPATION", "OTHER_QUASI"})


if __name__ == "__main__":
    unittest.main()
