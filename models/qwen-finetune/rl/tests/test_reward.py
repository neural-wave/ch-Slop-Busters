"""Reward tests of TRAINING_PLAN.md §4, using the eval reference labels as if they were silver.

  python3 -m unittest discover -s rl/tests        (or .venv-train/bin/python for the tokenizer test)
"""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parents[3] / "term-generalizer")]
from common import EVAL_DATA, GOLD_EVAL, read_jsonl  # noqa: E402

import augment as A  # noqa: E402
import reward as W  # noqa: E402
from data import silver_target  # noqa: E402
from spec import dump_output, dx_prefix  # noqa: E402

RECS = {r["encounter_id"]: r for r in read_jsonl(EVAL_DATA)}
GOLD = {g["encounter_id"]: g for g in read_jsonl(GOLD_EVAL)}
IDS = sorted(GOLD)[:8]
EMPTY = {"dx": [], "evidence": [], "entities": [], "keep": []}


def pii_free(rec: dict, gold: dict) -> tuple[dict, dict]:
    """The lines of the dialogue without any silver PII or regex span, note dropped."""
    import generalize as G
    lines = []
    for l in rec["dialogue"].split("\n"):
        if any(W.R.find_all(e["text"], l) for e in gold["pii"]) or G.find_spans(l):
            continue
        lines.append(l)
    r = {**rec, "dialogue": "\n".join(lines[:60]), "note": ""}
    return r, A.restrict_gold(r, gold)


def no_concepts(gold: dict) -> dict:
    return {**gold, "medical_concepts": [], "evidence": [], "ambiguous_terms": []}


def cells():
    for eid in IDS:
        rec, g = RECS[eid], GOLD[eid]
        clean_rec, clean_g = pii_free(rec, g)
        yield f"{eid}/pii+med", rec, g
        yield f"{eid}/pii-only", rec, no_concepts(g)
        yield f"{eid}/med-only", clean_rec, clean_g
        yield f"{eid}/neither", clean_rec, no_concepts(clean_g)


def words(text: str) -> list[str]:
    import re
    return sorted({w for w in re.findall(r"[a-z][a-z']+", text.lower()) if len(w) >= 2})


def candidates(rec: dict, g: dict) -> dict:
    silver = silver_target(rec, g)
    text = (rec.get("dialogue") or "") + "\n" + (rec.get("note") or "")
    med = {w for c in g.get("medical_concepts", []) for w in words(c["text"])}
    pii = {w for e in g.get("pii", []) for w in words(e["text"])}
    rng = random.Random(0)
    vocab = words(text)
    return {
        "identity": EMPTY,
        "silver": silver,
        "redact_every_word": {**EMPTY, "entities": [{"text": w, "type": "OTHER_QUASI"} for w in vocab]},
        "selective_over": {**EMPTY, "entities": [{"text": w, "type": "OTHER_QUASI"} for w in vocab
                                                 if w not in med and w not in pii]},
        "line_deletion": {**EMPTY, "entities": [{"text": l, "type": "OTHER_QUASI"} for l in text.split("\n") if l.strip()]},
        "random_200": {**EMPTY, "entities": [{"text": rng.choice(vocab), "type": "OTHER_QUASI"} for _ in range(200)]},
    }


def ret(rec, g, out, prefill=False, cfg=None) -> W.RewardResult:
    sv = W.prepare(rec, g)
    full = dump_output({**out, "dx": out.get("dx") or [{"code": c["code"], "desc": c["description"].lower()}
                                                       for c in rec["icd10_codes"]]}) if prefill else dump_output(out)
    prefix = dx_prefix(rec["icd10_codes"]) if prefill else ""
    if prefill:
        assert full.startswith(prefix)
    return W.compute(rec, sv, full[len(prefix):], prefix, cfg)


class TestCells(unittest.TestCase):
    def test_silver_is_best_and_hacks_lose(self):
        for name, rec, g in cells():
            has_pii = bool(W.prepare(rec, g).pii_rows)
            cands = candidates(rec, g)
            r = {k: ret(rec, g, v, prefill=True).total for k, v in cands.items()}
            with self.subTest(cell=name, returns={k: round(v, 3) for k, v in r.items()}):
                # with neither PII nor concepts the silver output is the identity: a tie at 0
                self.assertGreaterEqual(r["silver"], max(r.values()) - 1e-9)
                self.assertLessEqual(r["selective_over"], -1)
                self.assertLessEqual(r["line_deletion"], -1)
                if not has_pii:
                    self.assertAlmostEqual(r["identity"], 0.0, places=6)
                else:
                    res = ret(rec, g, EMPTY, prefill=True)
                    self.assertAlmostEqual(r["identity"], -(1 - res.parts["P0"]), places=6)
                    self.assertLess(r["identity"], 0)

    def test_invalid_json(self):
        rec, g = RECS[IDS[0]], GOLD[IDS[0]]
        sv = W.prepare(rec, g)
        for bad in ['{"entities": [{"text": "x"', "not json", '{"dx": []} trailing', ""]:
            res = W.compute(rec, sv, bad)
            self.assertFalse(res.valid)
            self.assertEqual(res.total, -1.0)


class TestDx(unittest.TestCase):
    def test_invented_code_penalized_prefill_scores_zero(self):
        rec, g = RECS[IDS[1]], GOLD[IDS[1]]
        silver = silver_target(rec, g)
        r_pre = ret(rec, g, silver, prefill=True)
        self.assertNotIn("dx", r_pre.parts)
        good = ret(rec, g, silver)
        bad = ret(rec, g, {**silver, "dx": silver["dx"] + [{"code": "Q99.99X", "desc": "invented disease of nothing"}]})
        self.assertGreater(good.parts["dx"], 0)
        self.assertLess(bad.parts["dx"], good.parts["dx"] - 0.4)   # -0.3 false positive, -0.2 invalid code
        wrong = ret(rec, g, {**silver, "dx": [{**silver["dx"][0], "code": "Z99.89"}] + silver["dx"][1:]})
        self.assertLess(wrong.parts["dx"], good.parts["dx"])


class TestDirections(unittest.TestCase):
    REC = {"encounter_id": "T1", "dialogue": "[patient] my friend martha said my knee hurts .\n[doctor] okay .",
           "note": "", "icd10_codes": [{"code": "M25.561", "description": "Pain in right knee"}]}
    GOLD = {"pii": [{"text": "martha", "field": "dialogue", "category": "OTHER_PERSON_NAME", "tier": "direct"}],
            "medical_concepts": [{"text": "knee hurts", "field": "dialogue", "type": "symptom", "icd_relevant": True}],
            "ambiguous_terms": [], "evidence": [{"code": "M25.561", "spans": [{"text": "knee hurts", "field": "dialogue"}]}]}
    A = {**EMPTY, "entities": [{"text": "martha said my knee", "type": "OTHER_QUASI"}]}   # one more PII removed
    B = EMPTY                                                                            # one more concept kept

    def diff(self, **kw) -> float:
        cfg = W.RewardConfig(**kw)
        return ret(self.REC, self.GOLD, self.A, cfg=cfg).total - ret(self.REC, self.GOLD, self.B, cfg=cfg).total

    def test_knobs(self):
        base = self.diff()
        self.assertGreater(self.diff(w_P=2.0), base)
        self.assertGreater(self.diff(w_miss=2.0), base)
        self.assertLess(self.diff(w_U=2.0), base)
        self.assertLess(self.diff(lam_fp=0.1), base)


class TestSumProperty(unittest.TestCase):
    def test_entity_rewards_telescope(self):
        for eid in IDS[:4]:
            rec, g = RECS[eid], GOLD[eid]
            silver = silver_target(rec, g)
            totals = []
            for seed in range(3):
                ents = silver["entities"][:]
                random.Random(seed).shuffle(ents)
                res = ret(rec, g, {**silver, "entities": ents})
                d = res.parts.get("dP", 0) + res.parts.get("dU", 0)
                self.assertAlmostEqual(d, (res.parts["P"] - res.parts["P0"]) + (res.parts["U"] - res.parts["U0"]), 9)
                totals.append(round(d, 9))
            self.assertEqual(len(set(totals)), 1, eid)


class TestPositions(unittest.TestCase):
    def test_item_ends(self):
        out = {"dx": [{"code": "I10", "desc": "x"}], "evidence": [], "entities": [{"text": "a b", "type": "AGE"}],
               "keep": [{"text": "k"}]}
        s = dump_output(out)
        obj, ends = W.parse_with_positions(s)
        self.assertEqual(obj, out)
        self.assertEqual(s[ends[("entities", 0)] - 1], "}")
        self.assertEqual(s[ends[("entities", "]")] - 1], "]")
        self.assertTrue(s[:ends[("dx", 0)]].endswith('"x"}'))

    def test_token_mapping(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers not installed (use .venv-train)")
        tok = AutoTokenizer.from_pretrained(str(Path(W.DATA_DIR).parent / "models" / "Qwen3.5-2B"))
        rec, g = RECS[IDS[0]], GOLD[IDS[0]]
        silver = silver_target(rec, g)
        prefix = dx_prefix(rec["icd10_codes"])
        resp = dump_output(silver)[len(prefix):]
        ids = tok(resp, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
        ends = W.token_char_ends(tok, ids)
        self.assertEqual(ends[-2], len(resp))
        self.assertEqual(ends[-1], len(resp))
        res = W.compute(rec, W.prepare(rec, g), resp, prefix)
        dense = W.to_token_rewards(res, ends, len(prefix))
        self.assertAlmostEqual(sum(dense), res.total, 9)
        # every entity reward lands on a token that ends with the item's closing brace
        for pos, r, label in res.items:
            if label in ("dP", "type"):
                j = next(i for i, e in enumerate(ends) if e >= pos - len(prefix))
                self.assertIn("}", tok.decode(ids[j:j + 1]))


if __name__ == "__main__":
    unittest.main()
