"""Unit tests for generalize.py (age/date ranges) and the generalize path of anonymize.apply_entities.

  python3 -m unittest harness/test_generalize.py      (or: python3 harness/test_generalize.py)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generalize as G  # noqa: E402
from anonymize import (anonymize_chunk_extract, anonymize_chunk_generalize, apply_entities,  # noqa: E402
                       icd_terms, is_icd_term)


def regex_pass(text: str) -> str:
    """What generalize mode does with no LLM entities at all."""
    return apply_entities(text, G.find_spans(text), render=G.render)[0]


# ---------------------------------------------------------------- bands

class TestBands(unittest.TestCase):
    def test_age_band(self):
        cases = {0.5: "newborn", 1: "infant", 11: "infant", 12: "toddler", 35: "toddler", 36: "preschool-age",
                 59: "preschool-age", 60: "school-age", 12 * 12: "school-age", 13 * 12: "teenage",
                 17 * 12: "teenage", 18 * 12: "young-adult", 19 * 12: "young-adult", 20 * 12: "twenty-something",
                 44 * 12: "forty-something", 80 * 12: "eighty-something", 89 * 12: "eighty-something",
                 90 * 12: "ninety-plus", 100 * 12: "ninety-plus"}
        for months, band in cases.items():
            self.assertEqual(G.age_band(months), band, months)

    def test_parse_num(self):
        for s, n in {"44": 44, "sixteen": 16, "thirteen": 13, "twenty one": 21, "twenty-one": 21, "eight": 8,
                     "forty": 40}.items():
            self.assertEqual(G.parse_num(s), n)
        self.assertIsNone(G.parse_num("eighth"))

    def test_decade_phrase(self):
        self.assertEqual(G.decade_phrase(2021), "early 2020s")
        self.assertEqual(G.decade_phrase(1949), "late 1940s")
        self.assertEqual(G.decade_phrase(2015), "mid 2010s")
        self.assertEqual(G.decade_phrase(1980), "early 1980s")


# ---------------------------------------------------------------- ages (regex pass)

class TestAges(unittest.TestCase):
    def check(self, cases):
        for src, want in cases.items():
            self.assertEqual(regex_pass(src), want, src)

    def test_adjectival(self):
        self.check({
            "a 62-year-old male": "a sixty-something male",
            "she is a 44-year-old female .": "she is a forty-something female .",
            "62 year old": "sixty-something",
            "a 34 -year-old woman": "a thirty-something woman",
            "44-years-old": "forty-something",
            "a thirteen -year-old": "a teenage",
            "my two four -year-old": "my two preschool-age",
            "a twenty one -year-old": "a twenty-something",
            "a 91-year-old female": "a ninety-plus female",
            "a 100-year-old female": "a ninety-plus female",
            "Sophia Brown is a 75 y.o. female": "Sophia Brown is a seventy-something female",
            "a 27 y/o female": "a twenty-something female",
        })

    def test_article_fix(self):
        self.check({
            "an 8-year-old boy": "a school-age boy",
            "An 8-year-old boy": "A school-age boy",
            "a 6-month-old": "an infant",
            "an 80-year-old man": "an eighty-something man",
            "a 18-year-old": "a young-adult",
        })

    def test_infants(self):
        self.check({
            "katherine is a 22-month-old twin": "katherine is a toddler twin",
            "my 9-month-old": "my infant",
            "a 3-week-old": "a newborn",
            "my 6-month-old": "my infant",
        })

    def test_predicative(self):
        self.check({
            "when she was 14 years old .": "when she was teenage .",
            "Brother: 9 years old, positive for autism.": "Brother: school-age, positive for autism.",
            "around eight years old i had": "around school-age i had",
            "she is 59 years old .": "she is fifty-something years old .",
            "typically 10 years of age": "typically school-age",
            "he is 45 years of age": "he is forty-something years of age",
            "a 93 years old": "a ninety-plus years old",
            "since she was 4 or 5 years old .": "since she was school-age .",
            "When the patient was 4 years old, she": "When the patient was preschool-age, she",
        })

    def test_age_phrase(self):
        self.check({
            "Mother, deceased age 45, from": "Mother, deceased age forty-something, from",
            "at the young age of 27 ,": "at the young age of twenty-something ,",
            "Appendectomy at age 7.": "Appendectomy at school age.",
            "reconstruction at the age of 14.": "reconstruction at the teenage years.",
            "since approximately age 12 and": "since approximately school age and",
            "a boy aged 7": "a boy of school age",
            "a woman aged 45": "a woman aged forty-something",
            "i'm happy at age 67 you're": "i'm happy at age sixty-something you're",
            "Age 45 at onset": "Age forty-something at onset",
        })

    def test_clauses(self):
        self.check({
            "since she was 4 .": "since she was preschool-age .",
            "when she was 14 ,": "when she was teenage ,",
            "when i was 18 .": "when i was young-adult .",
            "since i was 13 years old .": "since i was teenage .",
            "since i was around 12 , but": "since i was school-age , but",
            "when you were fourteen i'm gon na": "when you were teenage i'm gon na",
            "when i was seven .": "when i was school-age .",
            "when she was four years old she": "when she was preschool-age she",
            "when i was 25 years old .": "when i was twenty-something years old .",
            "i'm almost 40 and my mom was almost 45 when": "i'm thirty-something and my mom was forty-something when",
            "she is almost sixteen and": "she is teenage and",
            "i'm 44 now and": "i'm forty-something now and",
            "i'm 25 .": "i'm twenty-something .",
            "when you turn 50 .": "when you turn fifty-something .",
            "he turned 80 last week": "he turned eighty-something last week",
        })

    def test_age_traps(self):
        for src in [
            "pain is 7/10 today", "blood pressure 120/80", "it started 2 years ago", "about 30 years ago",
            "a 10-year history of diabetes", "a 3-month follow-up", "heart rate in the 80s", "he died in his 40s",
            "i started in my teens", "children under 1 years old", "dose for kids under a year of age",
            "under the age of 50 who have", "children between the ages of 9 to 11 years old and again between "
            "17 to 20 years old", "occur around five to ten years of age", "around 5 to 10 years of age",
            "did not sit up unassisted until 1 years old", "a former 34+3-week-old twin",
            "former 34 and 3-week-old , twin", "abrasion that appears to be 1 week old", "turn 90 degrees",
            "i'm 5 foot 4", "you're ninety seven . eight", "a1c for almost five months", "almost 200",
            "since i was like four five i've", "when i was younger", "when she was one of the", "age related",
            "given my age of all", "the second visit", "you may have", "march madness", "march 3 times",
            "it was 140 .", "2000 mg per day", "i weigh 180 now", "stage 3 cancer",
        ]:
            self.assertEqual(regex_pass(src), src, src)

    def test_case_preserved(self):
        self.assertEqual(G.render("Since she was 4", "AGE"), "Since she was preschool-age")
        self.assertEqual(G.render("July", "DATE"), "Summer")
        self.assertEqual(G.render("july", "DATE"), "summer")

    def test_no_digit_survives(self):
        for n in range(1, 121):
            for form in (f"a {n}-year-old woman", f"she is {n} years old .", f"age {n}", f"when i was {n} ."):
                out = regex_pass(form)
                self.assertNotIn(str(n), out, form)


# ---------------------------------------------------------------- dates

class TestDates(unittest.TestCase):
    def test_regex_dates(self):
        cases = {
            "born on january 13th , 1982 .": "born on early 1980s .",
            "seen on july 8th and": "seen on summer and",
            "DOB 1/12/71.": "DOB early 1970s.",
            "on 4/24/2021 she": "on early 2020s she",
            "since 03/2021.": "since early 2020s.",
            "since 04/2022": "since early 2020s",
            "on september 9th , 2019 ,": "on late 2010s ,",
            "on january 15th , '21 and": "on early 2020s and",
            "back in october 2019 ,": "back in late 2010s ,",
            "until april of 2022 .": "until early 2020s .",
            "born 09/18/1949 .": "born late 1940s .",
            "last seen 8-23-97 .": "last seen late 1990s .",
            "on the 28th of march": "on spring",
            "on march 26 we": "on spring we",
            "on august first": "on summer",
            "january , 2021": "early 2020s",
            "may 17th": "spring",
            "In March 2020, she": "In Early 2020s, she",
        }
        for src, want in cases.items():
            self.assertEqual(regex_pass(src), want, src)

    def test_date_traps(self):
        for src in ["7/10 pain", "bp 120/80 .", "2/6 strength", "vision 20/25", "the 28th", "in july", "2019",
                    "you may first want to", "may 17 .", "march 2000 mg", "10-10 20-20", "6421", "3/10 pain",
                    "a 3-month follow-up", "the second time", "may 2 weeks"]:
            self.assertEqual(regex_pass(src), src, src)

    def test_llm_date_spans(self):
        cases = {"january 13th , 1982": "early 1980s", "july 8th": "summer", "july": "summer",
                 "november and december": "autumn and winter", "december and january": "winter",
                 "1/12/71": "early 1970s", "4/24/2021": "early 2020s", "03/2021": "early 2020s",
                 "04/2022": "early 2020s", "september 9th , 2019": "late 2010s", "january 15th , '21": "early 2020s",
                 "october 2019": "late 2010s", "april of 2022": "early 2020s", "2019": "late 2010s",
                 "15th 1962": "early 1960s", "November 24th 2020": "Early 2020s", "twenty eighteen": "late 2010s",
                 "two thousand eighteen": "late 2010s", "march of this year": "spring of this year",
                 "end of june": "end of summer", "04/10": "spring", "the 28th": "[DATE]", "6421": "[DATE]",
                 "10-10 20-20": "[DATE]", "monday": "[DATE]", "my birthday": "[DATE]", "on the twenty first": "[DATE]",
                 "49-year-old": "forty-something", "75 y.o.": "seventy-something"}
        for src, want in cases.items():
            self.assertEqual(G.render(src, "DATE"), want, src)

    def test_no_calendar_kept(self):
        """Relative time and measurements mislabeled DATE carry no calendar date: kept for the ICD coder."""
        for src in ["yesterday", "two weeks", "about the last 48 hours", "2 years ago", "today's visit", "last year",
                    "180/95 mmHg", "120/80", "2000 mg", "40 milligrams", "ninety eight percent",
                    "one twenty to one forty", "five to ten years of age", "2:00 am"]:
            self.assertEqual(G.render(src, "DATE"), src, src)


# ---------------------------------------------------------------- LLM AGE spans and combination

class TestRender(unittest.TestCase):
    def test_llm_age_spans(self):
        cases = {"44-year-old female": "forty-something female", "61-year-old": "sixty-something",
                 "4 or 5 years old": "school-age", "59": "fifty-something", "sixteen": "teenage",
                 "thirteen": "teenage", "fourteen": "teenage", "four": "preschool-age", "four five": "school-age",
                 "almost 40": "thirty-something", "95": "ninety-plus", "in his 40s": "in his 40s",
                 "180": "[AGE]", "110 pounds": "[AGE]", "eight years": "[AGE]", "twenty eighth": "[AGE]",
                 "30 years ago": "[AGE]", "1 to 2": "[AGE]"}
        for src, want in cases.items():
            self.assertEqual(G.render(src, "AGE"), want, src)

    def test_other_types_keep_tags(self):
        for typ in ("PERSON", "LOCATION", "ORGANIZATION", "OCCUPATION", "CONTACT", "ID"):
            self.assertEqual(G.render("john", typ), f"[{typ}]")

    def test_combination(self):
        chunk = "[doctor] hi john , you are a 61-year-old male seen on 4/24/2021 , pain 7/10 .\nJohn is 61-year-old."
        llm = [{"text": "john", "type": "PERSON"}, {"text": "61-year-old male", "type": "AGE"}]
        out, kept = apply_entities(chunk, G.find_spans(chunk) + llm, render=G.render)
        self.assertEqual(out, "[doctor] hi [PERSON] , you are a sixty-something male seen on early 2020s , "
                              "pain 7/10 .\n[PERSON] is sixty-something.")
        self.assertEqual({e["text"] for e in kept}, {"a 61-year-old", "61-year-old", "4/24/2021", "john",
                                                       "61-year-old male"})

    def test_extract_default_unchanged(self):
        chunk = "hi john , a 61-year-old on 4/24/2021"
        out, _ = apply_entities(chunk, [{"text": "john", "type": "PERSON"}, {"text": "61-year-old", "type": "AGE"}])
        self.assertEqual(out, "hi [PERSON] , a [AGE] on 4/24/2021")


# ---------------------------------------------------------------- ICD-10 hints and explanations

class TestIcdHints(unittest.TestCase):
    REC = {"icd10_codes": [{"code": "E11.9", "description": "Type 2 diabetes mellitus without complications"},
                           {"code": "M17.11", "description": "Unilateral primary osteoarthritis, right knee"},
                           {"code": "G20", "description": "Parkinson's disease"}]}

    def test_terms(self):
        terms = icd_terms(self.REC)
        self.assertTrue({"diabetes", "mellitus", "knee", "parkinson", "osteoarthritis"} <= terms)
        self.assertNotIn("type", terms)
        self.assertNotIn("without", terms)

    def test_is_icd_term(self):
        terms = icd_terms(self.REC)
        for t in ("diabetes", "right knee", "Parkinson's", "knee", "left"):
            self.assertTrue(is_icd_term(t, terms), t)
        for t in ("martha", "Dr. Parkinson Smith", "boston", "type", ""):
            self.assertFalse(is_icd_term(t, terms), t)

    def test_generalize_keeps_diagnosis_words(self):
        chunk = "martha has diabetes and her right knee hurts since 4/24/2021 ."
        cached = {"entities": [{"text": "martha", "type": "PERSON"}, {"text": "diabetes", "type": "ORGANIZATION"},
                               {"text": "right knee", "type": "LOCATION"}]}
        out, log = anonymize_chunk_generalize(chunk, "m", 0, cached=cached, protect=icd_terms(self.REC))
        self.assertEqual(out, "[PERSON] has diabetes and her right knee hurts since early 2020s .")
        self.assertEqual({e["text"] for e in log["retained"]}, {"diabetes", "right knee"})
        self.assertEqual(len(log["entities"]), 3)  # the replay log keeps the raw LLM spans
        acts = {e["text"]: (e["action"], e["replacement"]) for e in log["applied"]}
        self.assertEqual(acts, {"martha": ("removed", "[PERSON]"), "4/24/2021": ("generalized", "early 2020s")})
        self.assertTrue(all(e["reason"] for e in log["applied"] + log["retained"]))
        out, log = anonymize_chunk_generalize(chunk, "m", 0, cached=cached, protect=None)  # --no-icd-hints
        self.assertEqual(out, "[PERSON] has [ORGANIZATION] and her [LOCATION] hurts since early 2020s .")
        self.assertEqual(log["retained"], [])

    def test_explain(self):
        self.assertIn("placeholder", G.explain("PERSON", "removed"))
        self.assertIn("age band", G.explain("AGE", "generalized"))


class TestTruncatedReplies(unittest.TestCase):
    RAW = ('{\n  "entities": [\n    {"text": "martha", "type": "PERSON"},\n    {"text": "4/24/2021", "type": "DATE"},\n'
           '    {"text": "knee", "type": "LOCATION"},\n    {"text": "knee", "type": "LOC')

    def test_salvaged_on_replay(self):
        """A chunk whose reply hit the token limit keeps its finished entities (old logs stored none)."""
        cached = {"entities": [], "parse_ok": False, "raw": self.RAW, "done_reason": "length"}
        out, log = anonymize_chunk_extract("martha , knee pain since 4/24/2021", "m", 0, cached=cached)
        self.assertEqual(out, "[PERSON] , [LOCATION] pain since [DATE]")
        self.assertEqual(log["salvaged"], 3)


if __name__ == "__main__":
    unittest.main()
