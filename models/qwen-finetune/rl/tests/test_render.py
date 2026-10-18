"""Renderer tests: linking, ctx, span limits, keep vetoes, category words.

  python3 -m unittest discover -s rl/tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parents[3] / "term-generalizer")]
import render as R  # noqa: E402

REC = {"encounter_id": "T", "dialogue": "[doctor] hi , martha . how is mr. parkinson ?\n"
                                        "[patient] fine . my parkinson's disease is worse since 4/24/2021 .\n"
                                        "[patient] i fell skiing on mount baker , pain 7/10 .",
       "note": "Martha Collins is a 62-year-old female.", "icd10_codes": [{"code": "G20", "description": "Parkinson's disease"}]}


def out(**kw):
    return {"dx": [], "evidence": [], "entities": kw.get("entities", []), "keep": kw.get("keep", [])}


class TestRender(unittest.TestCase):
    def test_linking_across_fields(self):
        r, log = R.render_record(REC, out(entities=[{"text": "martha", "type": "PATIENT_NAME"}]))
        self.assertNotIn("martha", r["dialogue"].lower())
        self.assertNotIn("martha", r["note"].lower())
        self.assertEqual(r["icd10_codes"], REC["icd10_codes"])
        self.assertEqual(log[-1]["occurrences"], 2)

    def test_eponym_needs_ctx(self):
        terms = R.terms_of(["Parkinson's disease"])
        r, log = R.render_record(REC, out(entities=[{"text": "parkinson", "type": "OTHER_PERSON_NAME"}]),
                                 protect_terms=terms)
        self.assertIn("parkinson", r["dialogue"])
        self.assertEqual(log[-1]["rule"], "rejected:needs_ctx")
        r, _ = R.render_record(REC, out(entities=[{"text": "parkinson", "ctx": "mr. parkinson",
                                                   "type": "OTHER_PERSON_NAME"}]), protect_terms=terms)
        self.assertIn("mr. [OTHER_PERSON_NAME]", r["dialogue"])
        self.assertIn("parkinson's disease", r["dialogue"])

    def test_span_limits(self):
        line = REC["dialogue"].split("\n")[1]
        p = R.plan(REC, out(entities=[{"text": line, "type": "OTHER_QUASI"}, {"text": "[doctor]", "type": "ID"},
                                      {"text": "not in the text", "type": "ID"}, {"text": "fine", "type": "BOGUS"},
                                      {"text": "fine\nmy", "type": "ID"}]))
        self.assertEqual([d.status for d in p.by_kind("entity")],
                         ["rejected:too_long", "rejected:speaker_tag", "rejected:not_found", "rejected:bad_type",
                          "rejected:multiline"])

    def test_regex_and_keep_veto(self):
        r, _ = R.render_record(REC, out())
        self.assertIn("early 2020s", r["dialogue"])
        self.assertIn("sixty-something", r["note"])
        self.assertIn("7/10", r["dialogue"])
        r, log = R.render_record(REC, out(keep=[{"text": "4/24/2021", "reason": "clinical date"}]))
        self.assertIn("4/24/2021", r["dialogue"])
        self.assertTrue(any(e["action"] == "retained" and e["kind"] == "regex" for e in log))

    def test_category_word(self):
        r, log = R.render_record(REC, out(entities=[{"text": "mount baker", "type": "LOCATION", "cat": "a mountain"}]))
        self.assertIn("skiing on a mountain", r["dialogue"])
        self.assertEqual(log[-1]["action"], "category")
        r, _ = R.render_record(REC, out(entities=[{"text": "mount baker", "type": "LOCATION", "cat": "a spaceship"}]))
        self.assertIn("skiing on [LOCATION]", r["dialogue"])

    def test_order_independent(self):
        ents = [{"text": "martha collins", "type": "PATIENT_NAME"}, {"text": "martha", "type": "PATIENT_NAME"},
                {"text": "collins", "type": "PATIENT_NAME"}]
        a, _ = R.render_record(REC, out(entities=ents))
        b, _ = R.render_record(REC, out(entities=ents[::-1]))
        self.assertEqual(a, b)
        self.assertTrue(a["note"].startswith("[PATIENT_NAME] is a sixty-something"))


if __name__ == "__main__":
    unittest.main()
