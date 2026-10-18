"""Layer 1: regex detectors for structured identifiers (HIPAA Safe Harbor style) and spoken-form ASR variants."""
from __future__ import annotations

import regex as re

from common import Span

_NUM_WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
              r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)")
_SPOKEN_NUM = rf"{_NUM_WORDS}(?:[\s-]+{_NUM_WORDS})*"
_NUM = rf"(?:\d{{1,3}}|{_SPOKEN_NUM})"
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
SAFE_MONTHS = r"(?:january|february|april|june|july|august|september|october|november|december)"  # 'may'/'march' are also verbs
ORD = r"(?:\d{1,2}(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty[\s-]+\w+|thirtieth|thirty[\s-]+first)"
YEAR = r"(?:19\d\d|20\d\d|'\d\d)"
SPOKEN_YEAR = r"(?:nineteen[\s-]+(?:hundred|" + _NUM_WORDS + r")(?:[\s-]+" + _NUM_WORDS + r")?|two[\s-]+thousand(?:[\s-]+and)?(?:[\s-]+" + _NUM_WORDS + r")*|twenty[\s-]+(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:[\s-]+" + _NUM_WORDS + r")?))"
WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
HOLIDAYS = (r"(?:christmas(?:\s+eve)?|thanksgiving|easter|halloween|new\s+year'?s(?:\s+eve|\s+day)?|fourth\s+of\s+july|"
            r"4th\s+of\s+july|independence\s+day|memorial\s+day|labor\s+day|valentine'?s(?:\s+day)?|hanukkah|"
            r"ramadan|passover|super\s*bowl|black\s+friday|st\.?\s+patrick'?s(?:\s+day)?|mother'?s\s+day|father'?s\s+day)")

PATTERNS = [
    # --- ages ---
    ("AGE", 0.97, "age expression 'N-year-old'",
     rf"\b{_NUM}\s*-?\s*(?:years?|yrs?|y/o|yo)(?:\s*-\s*|\s+)old\b"),
    ("AGE", 0.97, "age expression 'N-year-old'", rf"\b{_NUM}\s*-\s*year\s*-\s*old\b"),
    ("AGE", 0.95, "age expression 'age N'", rf"(?<=\bage\s)(?:of\s+)?{_NUM}\b|(?<=\baged\s){_NUM}\b|(?<=\bage:\s){_NUM}\b"),
    ("AGE", 0.9, "age expression 'N years of age'", rf"\b{_NUM}\s+years\s+of\s+age\b"),
    ("AGE", 0.9, "age expression 'in his/her Ns'", r"(?<=\b(?:his|her|their|my)\s)(?:early\s+|mid\s+|late\s+)?(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties|[2-9]0s)\b"),
    ("AGE", 0.85, "age expression 'I am N'", rf"(?<=\bi'm\s)(?:{_NUM})(?=\s*(?:\.|,|years|$))|(?<=\bi am\s)(?:{_NUM})(?=\s*(?:\.|,|years|$))"),
    ("AGE", 0.9, "age expression 'turned N'", rf"(?<=\bturned\s){_NUM}\b|(?<=\bturning\s){_NUM}\b"),
    # --- dates ---
    ("DATE", 0.98, "numeric date", r"\b\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*(?:\d{4}|\d{2})\b"),
    # month/day only with a date cue: bare "2/6", "5/5", "4/10" are murmur grades, strength and pain scores
    ("DATE", 0.9, "numeric month/day", r"(?<=\b(?:on|since|from|until|by|before|after|dated|date|of|seen|back on)\s)(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])\b(?![\d/.])"),
    ("DATE", 0.97, "month + day (+ year)", rf"\b{MONTHS}\.?\s+(?:the\s+)?{ORD}(?:\s*,?\s*(?:{YEAR}|{SPOKEN_YEAR}))?\b"),
    ("DATE", 0.97, "day + month", rf"\b(?:the\s+)?{ORD}\s+of\s+{MONTHS}(?:\s*,?\s*(?:{YEAR}|{SPOKEN_YEAR}))?\b"),
    ("DATE", 0.97, "month + year", rf"\b{MONTHS}\s*,?\s*(?:of\s+)?(?:{YEAR}|{SPOKEN_YEAR})\b"),
    ("DATE", 0.9, "calendar year", rf"\b(?:19[3-9]\d|20[0-3]\d)\b(?!\s*(?:mg|ml|mcg|milligrams?|micrograms?|units?|iu|international|cc|%|ft|feet|lbs?|pounds|steps|calories|miles|meters|dollars))"),
    ("DATE", 0.85, "spoken calendar year", rf"\b{SPOKEN_YEAR}\b(?=\s*(?:\.|,|\?|$|and|when|i|we|she|he|they|that|was|is))"),
    ("DATE", 0.8, "month name", rf"\b{SAFE_MONTHS}\b"),
    ("DATE", 0.75, "month name (may/march in date context)", r"(?<=\b(?:in|since|until|of|last|next|early|late|mid)\s)(?:may|march)\b"),
    ("DATE", 0.95, "date of birth", rf"(?<=\bdate of birth\s)(?:is\s+)?[^.\n]{{4,30}}?(?=\s*[.\n])|(?<=\bdob\s)[^.\n]{{4,30}}?(?=\s*[.\n])"),
    # --- temporal quasi-identifiers ---
    ("TEMPORAL", 0.8, "weekday pins down timing", rf"\b{WEEKDAYS}\b"),
    ("TEMPORAL", 0.85, "holiday / named event", rf"\b{HOLIDAYS}\b"),
    # --- contact / IDs ---
    ("CONTACT", 0.99, "phone number", r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
    ("CONTACT", 0.99, "email address", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("CONTACT", 0.95, "url", r"\bhttps?://\S+|\bwww\.\S+"),
    ("ID", 0.97, "medical record number", r"\b(?:mrn|mr#|medical record(?: number)?|record number|chart number)\s*[:#]?\s*[a-z]*\d[\w-]{3,}\b|\bmrn[a-z]?\d{4,}\b"),
    ("ID", 0.95, "social security number", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("ID", 0.9, "long numeric identifier", r"\b(?:id|account|policy|member|insurance)\s*(?:number|no\.?|#)?\s*[:#]?\s*\d{5,}\b"),
    ("LOCATION", 0.9, "zip code", r"(?<=\b(?:zip|zip code|postal code)\s)\d{5}(?:-\d{4})?\b"),
    ("LOCATION", 0.85, "street address", r"\b\d{1,5}\s+(?:[a-z]+\s){1,3}(?:street|st\.|avenue|ave\.?|road|rd\.|boulevard|blvd|lane|ln\.|drive|dr\.|court|ct\.|way)\b"),
]
_COMPILED = [(t, c, r, re.compile(p, re.IGNORECASE)) for t, c, r, p in PATTERNS]
# numbers that look like years but are clinical values (e.g. "glucose of 2000" is unlikely; "1980s" handled)
_CLINICAL_CONTEXT = re.compile(r"(mg|mcg|ml|units|count|level|score|calories|steps|cc)\W*$", re.IGNORECASE)


def detect(field: str, text: str) -> list[Span]:
    spans = []
    for typ, conf, reason, rx in _COMPILED:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            frag = text[s:e].strip()
            if not frag:
                continue
            if typ == "DATE" and reason == "calendar year" and _CLINICAL_CONTEXT.search(text[max(0, s - 25):s]):
                continue
            spans.append(Span(field, s, e, text[s:e], typ, "regex", conf, f"regex: {reason}"))
    return spans
