"""High-precision rule detectors (CPU, zero VRAM).

Each detector reads a `Doc` (normalized, speaker tags masked, lowercase view for dialogue) and
returns `Span`s with offsets into the original field. Rules come from the dataset analysis trap
catalogue: dictated headers, DOB in 6+ formats, MRNs glued/dashed/alphanumeric, spelled numbers in
`aci` dialogues, markerless ages (`you are 74`), titles with split dots (`mr . james`), etc.

Person-name rules here are *anchors*: they find a name with strong context. The identity graph
(`identity.py`) then propagates every anchored name through both fields of the encounter.
"""

from __future__ import annotations

import re

from . import lexicons as L
from .spans import Span
from .text import NUM, NUMWORDS, ORDINAL_WORDS, Doc, is_sentence_start, ordinal_to_int, words_to_int

# --------------------------------------------------------------------------- dates

_TT = r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_U = r"(?:one|two|three|four|five|six|seven|eight|nine)"
# Spoken years: "nineteen fifty seven", "twenty eighteen", "two thousand and one".
YEARWORDS = rf"(?:(?:nineteen|twenty) (?:oh {_U}|{_TT}(?:[ -]{_U})?)|two thousand(?: and)?(?: (?:{_TT}(?:[ -]{_U})?|{_U}))?)"
YEAR_RX = r"(?:(?:19|20)\d{2}|'\d{2}|" + YEARWORDS + r")"
DAY_RX = r"(?:\d{1,2}(?:st|nd|rd|th)?|" + ORDINAL_WORDS + r")"

DATE_PATTERNS = [
    ("date_numeric", rf"(?<!\d[-/.])\b\d{{1,2}}\s?([/\-.])\s?\d{{1,2}}\s?\1\s?(?:\d{{4}}|\d{{2}})\b(?!\s?[-/.]\s?\d)"),
    ("date_iso", r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b"),
    ("date_month_day_year", rf"\b{L.MONTHS}\.? (?:the )?{DAY_RX}(?: ?,? ?(?:of )?{YEAR_RX})?\b"),
    ("date_day_month_year", rf"\b(?:the )?{DAY_RX} (?:of )?{L.MONTHS}\b(?: ?,? ?{YEAR_RX})?"),
    ("date_month_year", rf"\b{L.MONTHS}\.? ?,? ?(?:of )?{YEAR_RX}\b"),
    ("date_month_year_numeric", r"\b(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b"),
    ("date_this_year", rf"\b{L.MONTHS_FULL} of (?:this|last) year\b"),
]
_DATE_RES = [(n, re.compile(p)) for n, p in DATE_PATTERNS]
_MD_RE = re.compile(r"(?<![\d/])(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])(?![\d/])")
_MD_CTX = re.compile(
    r"\b(?:on|dated|from|since|until|of|as of|through|by|date|visit|seen|done|performed|starting|started|last|completed|"
    r"scheduled|at|exam|review|appt|appointment|admitted|admission|discharged|discharge|extubated|intubated|transferred|"
    r"stopped|restarted|held|resumed|placed|removed|inserted|presented|operated|surgery|op|procedure|delivered|born|died|"
    r"collected|drawn|resulted|reported|signed|received|follow-up|f/u|until|till|assessment|scan|test|review|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)[\s:]*$"
)
_DM_RE = re.compile(r"(?<![\d/])(1[3-9]|2\d|3[01])/(0?[1-9]|1[0-2])(?![\d/])")  # unambiguous dd/mm (`30/08`)
_CLIN_AFTER = re.compile(r"^\s*(?:murmur|strength|pain|scale|score|reflex|systolic|diastolic|mmhg|dose|vision|grade|ratio|mg|mcg)")


def _valid_date(text: str) -> bool:
    """Reject things like `may 2` used as a verb + number, and impossible days."""
    m = re.search(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?!\d)", text)  # a day, never the digits of a year
    if text.startswith("may ") and not re.search(r"\d|first|second|third|th\b", text):
        return False
    if m and not re.search(r"\d{1,2}\s?[/\-.]\s?\d", text):
        d = int(m.group(1))
        if not 1 <= d <= 31:
            return False
    yw = re.search(YEARWORDS, text)
    if yw:
        v = words_to_int(yw.group(0))
        if v is None or not 1900 <= v <= 2039:
            return False
    return True


def dates(doc: Doc) -> list[Span]:
    t, out = doc.lower, []
    for name, rx in _DATE_RES:
        for m in rx.finditer(t):
            txt = m.group(0)
            if name.startswith("date_month") or name == "date_day_month_year":
                if not _valid_date(txt):
                    continue
            if name == "date_numeric":
                a_, b_ = (int(x) for x in re.findall(r"\d+", txt)[:2])
                if not ((1 <= a_ <= 12 and 1 <= b_ <= 31) or (1 <= b_ <= 12 and 1 <= a_ <= 31)):
                    continue
                # bare "march 2" style without year is fine; bare month+ordinal word must be a real ordinal
            out.append(Span(doc.field, m.start(), m.end(), "DATE", f"rule:{name}", 0.95, "absolute calendar date"))
    for rx in (_MD_RE, _DM_RE):  # mm/dd or dd/mm without year: only in date context, never in clinical context
        for m in rx.finditer(t):
            if _MD_CTX.search(t[max(0, m.start() - 14) : m.start()]) and not _CLIN_AFTER.search(t[m.end() : m.end() + 12]):
                out.append(Span(doc.field, m.start(), m.end(), "DATE", "rule:date_month_day_numeric", 0.85, "day/month date in date context"))
    # "on the 24th" (day of month anchored to an event)
    for m in re.finditer(r"\b(?:on|since|until|from|by|before|after) the (\d{1,2}(?:st|nd|rd|th))\b(?=\s?(?:[,.;!?)]|$|\s(?:of (?!the (?:month|year))|and|at|when|for|we|i|you|she|he|they|so|but|because|in)\b))", t):
        out.append(Span(doc.field, m.start(1), m.end(1), "DATE", "rule:day_of_month", 0.8, "day of month tied to an event"))
    return out


_DOB_CUE = re.compile(r"\b(?:date of birth|birth ?date|d\.?\s?o\.?\s?b\.?|born on|born in|birthday is|birthday)\b")


def dob(doc: Doc, date_spans: list[Span]) -> list[Span]:
    """Promote dates that follow a DOB cue to DOB; also catch DOB-cued numbers the date rules missed."""
    t, out = doc.lower, []
    for m in _DOB_CUE.finditer(t):
        window_end = m.end() + 45
        hit = None
        for s in date_spans:
            if m.end() <= s.start <= window_end and re.fullmatch(r"[\s,:.\-]*(?:is|of|was|would be|um|uh|n-|,|\.|:)?[\s,:.\-]*(?:(?:um|uh|is)[\s,]*)?", t[m.end() : s.start]):
                hit = s
                break
        if hit:
            out.append(Span(doc.field, hit.start, hit.end, "DOB", "rule:dob_cue+date", 1.0, "date following a date-of-birth cue"))
            continue
        mm = re.match(r"[\s,:.\-]*(?:is\s)?(\d[\d/\-. ]{4,12}\d)", t[m.end() : window_end])
        if mm:
            s = m.end() + mm.start(1)
            out.append(Span(doc.field, s, s + len(mm.group(1)), "DOB", "rule:dob_cue+digits", 0.95, "digits following a date-of-birth cue"))
    return out


_LAB_BEFORE = re.compile(
    r"\b(?:ck|cpk|anc|ldh|bnp|nt-?probnp|probnp|viral load|load|count|level|levels|output|intake|platelets?|wbc|alt|ast|"
    r"ferritin|tsh|psa|cd4|copies|glucose|sugar|troponin|d-dimer|creatinine|crp|esr|lipase|amylase|alk phos|"
    r"calorie|calories|diet|fluid restriction|restriction|limit|goal|target|dose|units|weight|weighs|grams|ml|cc)"
    r"\s?(?:is|was|of|at|about|around|were|=|:|went up to|came back|came back at|up to|down to|now)?\s?(?:a\s)?$"
)


def years(doc: Doc) -> list[Span]:
    t, out = doc.lower, []
    for m in re.finditer(r"(?<![\d$/.,-])\b((?:19[3-9]\d|20[0-3]\d))\b(?![\d/,.]\d|\s?(?:mg|mcg|ml|units?|milligrams?|calorie|calories|kcal|cal\b|steps|pounds?|lbs|dollars|grams?|cc|iu|%|copies|cells|u/l|iu/l|international|ml/|mosm|meq))", t):
        if _LAB_BEFORE.search(t[max(0, m.start() - 30) : m.start()]):
            continue  # `ck 2000`, `viral load 1990`, `nt-probnp was 2030`, `urine output 1950`
        out.append(Span(doc.field, m.start(1), m.end(1), "YEAR", "rule:year_digits", 0.9, "calendar year", value=int(m.group(1))))
    for m in re.finditer(r"\b" + YEARWORDS + r"\b", t):
        v = words_to_int(m.group(0))
        if v and 1930 <= v <= 2039 and not re.match(r"\s?(?:milligrams?|mg|units|calories|steps|pounds|dollars)", t[m.end() : m.end() + 12]):
            out.append(Span(doc.field, m.start(), m.end(), "YEAR", "rule:year_words", 0.85, "spoken calendar year", value=v))
    return out


# --------------------------------------------------------------------------- identifiers / contact

_ID_CUE = (
    r"(?:mrn|m\.r\.n\.|medical record(?: number| no\.?)?|record number|chart number|account number|acct(?: number)?|member id|"
    r"member number|policy number|patient id|id number|ssn|social security(?: number)?|npi|dea|insurance id|subscriber id|"
    r"case number|encounter(?: number| no\.?| #)?|visit number|reference number|ref(?: no\.?)?|accession(?: number| no\.?| #)?|"
    r"ur(?: number| no\.?)|hospital number|hospital no\.?|nhs(?: number| no\.?)?|chi(?: number| no\.?)|phn|ihi|medicare(?: number| no\.?)?|"
    r"health card(?: number| no\.?)?|specimen(?: number| no\.?| #)?|lab(?: number| no\.?| #)|order(?: number| no\.?| #)|"
    r"claim(?: number| no\.?| #)?|licen[cs]e(?: number| no\.?)?|badge(?: number| no\.?)?|crew id|unit id|employee id)"
)
_ID_TOKEN = r"(?:(?:\d\s){3,}\d|\d{3,5}(?:\s\d{3,5}){1,2}(?:\s\d{1,4})?|[a-z]{0,4}[-#]?\d[\d\-]{2,}[a-z0-9]*|[a-z]{1,4}\d{3,}[a-z0-9]*|(?:\d\s?-\s?){3,}\d|[a-z]{2,4}\d{2,})"


def identifiers(doc: Doc) -> list[Span]:
    t, out = doc.lower, []
    for m in re.finditer(rf"\b{_ID_CUE}\b\s?(?:#|no\.?|number)?\s?(?:is\s|was\s|of\s)?[:#]?\s?(?:-\s?)?(?P<id>{_ID_TOKEN})", t):
        if m.group(0).startswith("ref") and re.fullmatch(r"\d{1,4}\s?-\s?\d{1,4}", m.group("id")):
            continue  # `(ref 70-99)` is a reference range
        out.append(Span(doc.field, m.start("id"), m.end("id"), "ID", "rule:id_cue", 1.0, "identifier after a record-number cue"))
    digit = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double [a-z]+|triple [a-z]+|\d)"
    for m in re.finditer(rf"\b(?:{_ID_CUE}|phone(?: number)?|number|cell|mobile|call me at|reach me at|contact number)\b\s?(?:#|no\.?|number)?\s?(?:is\s|was\s|of\s|at\s)?[:#]?\s?(?P<id>(?:{digit}[\s,-]{{1,3}}){{3,}}{digit})\b", t):
        cat = "PHONE" if re.search(r"phone|cell|mobile|call|reach|contact", m.group(0)) else "ID"
        out.append(Span(doc.field, m.start("id"), m.end("id"), cat, "rule:spoken_digits", 0.9, "identifier spoken digit by digit"))
    for m in re.finditer(r"\bmrn(?P<id>[a-z]?\d{4,}[a-z0-9]*)\b", t):  # glued: mrnr472348
        out.append(Span(doc.field, m.start("id"), m.end("id"), "ID", "rule:mrn_glued", 1.0, "MRN glued to its cue (ASR)"))
    for m in re.finditer(r"(?<![\d.,/])\b\d{6,}\b(?![.,]\d)", t):  # long digit strings
        if _LAB_BEFORE.search(t[max(0, m.start() - 30) : m.start()]) or re.match(r"\s?(?:copies|cells|cfu|/|per|iu|units|mg|ml)", t[m.end() : m.end() + 8]):
            continue  # viral load 150000 copies/mL, >100,000 CFU
        out.append(Span(doc.field, m.start(), m.end(), "ID", "rule:long_number", 0.75, "long digit string (possible identifier)"))
    for m in re.finditer(r"\b[a-z]{2,5}-\d{5,}\b", t):  # MRN-3920184 style
        out.append(Span(doc.field, m.start(), m.end(), "ID", "rule:prefixed_number", 0.9, "prefixed record number"))
    return out


def contacts(doc: Doc) -> list[Span]:
    t, out = doc.view, []
    for m in re.finditer(r"(?<=, )(?P<town>[A-Z][a-z]+(?: [A-Z][a-z]+)?)\s\d{5,6}\b", t):  # `Springfield 62701`
        if not any(w.lower() in L.medical_vocab() or L.is_common_word(w.lower(), 30) for w in m.group("town").split()) and not _LAB_BEFORE.search(m.group("town").lower() + " "):
            out.append(Span(doc.field, m.start(), m.end(), "ADDRESS", "rule:town_zip", 0.9, "town + postal code"))
    for name, cat, rx in (
        ("email", "EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
        ("url", "URL", r"\b(?:https?://|www\.)\S+"),
        ("ssn", "ID", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("phone", "PHONE", r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)[\s.-]?|\b\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"),
        ("phone_in", "PHONE", r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"),
        ("street", "ADDRESS", r"\b\d{1,6}\s(?:[A-Za-z][A-Za-z.']+\s){1,4}(?:Street|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Court|Terrace|Circle|Highway|Hwy|Parkway|Pkwy|Marg|Nagar)\b\.?"),
        ("zip", "ADDRESS", r"\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\s\d{5}(?:-\d{4})?\b|\b(?:Tamil Nadu|Telangana|Karnataka|Kerala|Maharashtra|Delhi|West Bengal|Gujarat|Rajasthan|Punjab|Haryana|Uttar Pradesh|Andhra Pradesh|Odisha|Bihar|Assam)\s\d{6}\b"),
    ):
        for m in re.finditer(rx, t):
            out.append(Span(doc.field, m.start(), m.end(), cat, f"rule:{name}", 0.95, f"{cat.lower()} pattern"))
    return out


# --------------------------------------------------------------------------- ages

_AGE_UNITS_AFTER = re.compile(
    r"^\s?-?\s?(?:mg|milligrams?|mcg|percent|%|pounds?|lbs|units?|minutes?|hours?|days?|weeks?|months?|"
    r"times?|puffs?|tablets?|pills?|cm|mm|inch|inches|feet|foot|degrees?|refills?|out of|over|"
    r"(?:to|and|or|-)\s?(?:\d|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a half)|"
    r"more|less|point|\.\s?\d|/|kids|children|grand|cigarettes|drinks|beers|miles|steps|years? (?!old)|of (?!age)|"
    r"\s?\.\s?(?:zero|one|two|three|four|five|six|seven|eight|nine)\b|\s?,\s?\d|\s?,\s?(?:one|two|three|four|five|six|seven|eight|nine)\b|on room air)"
)
_VITAL_CTX = re.compile(
    r"\b(?:temp\w*|fever|febrile|oxygen|o2|sat\w*|room air|sugar|sugars|glucose|pulse|heart rate|pain|scale|tall|height|"
    r"weigh\w*|pounds|lost|gained|blood pressure|pressure|bp|score|rate|level|number|a1c|reading|units)\b"
)
_GESTATIONAL = re.compile(r"(?:gestational|corrected|bone|post-?menstrual|conceptional|gestation)\s?(?:age)?\s?:?\s*$")


def _age_value(num: str) -> int | None:
    return words_to_int(num)


def ages(doc: Doc) -> list[Span]:
    t, out = doc.lower, []

    def add(m, g, rule, conf, reason, unit="year"):
        v = _age_value(m.group(g))
        if v is None or v > 120:
            return
        # the span covers the number (and its `-year-old` unit), never the cue words before it, so a name
        # inside the cue (`my daughter sarah is 45`) is not swallowed by the age span
        out.append(Span(doc.field, m.start(g), m.end(), "AGE", f"rule:{rule}", conf, reason, value=v, meta={"unit": unit, "num_start": m.start(g), "num_end": m.end(g)}))

    for m in re.finditer(rf"\b(?P<num>{NUM})\s?-?\s?(?P<unit>year|yr|month|week|day)s?\s?-?\s?old\b", t):
        if re.search(r"\d{2}\s?(?:\+|and)\s?$", t[max(0, m.start() - 8) : m.start()]):
            continue  # `34+3-week-old`: gestational weeks+days, not an age
        if m.group("unit") in ("day", "week") and re.match(r"\s(?:wound|fracture|injury|burn|lesion|rash|ulcer|cut|laceration|bruise|incision|scar|bite|clot|bleed|hematoma|blister)", t[m.end() : m.end() + 14]):
            continue  # `a 5-day-old wound` dates the lesion, not the patient
        add(m, "num", "age_year_old", 1.0, "age expression 'N-year-old'", m.group("unit"))
    for m in re.finditer(r"\b(?P<num>\d{1,3})\s?(?:yo|y/o|y\.o\.?|yrs? old|years of age|year of age)(?![a-z])", t):
        add(m, "num", "age_yo", 1.0, "age expression 'N y.o.'")
    for m in re.finditer(rf"\b(?P<num>{NUM}) years of age\b", t):
        add(m, "num", "age_years_of_age", 1.0, "age expression 'N years of age'")
    for m in re.finditer(rf"\b(?:age|aged|at age|at the age of|the age of|age of|turned|turning|turns|just turned)\s?:?\s(?:about |almost |nearly |around |approximately )?(?P<num>{NUM})\b", t):
        if not _AGE_UNITS_AFTER.match(t[m.end() : m.end() + 14]) and not _GESTATIONAL.search(t[max(0, m.start() - 16) : m.start() + 4]):
            add(m, "num", "age_cue", 0.95, "number after an age cue")
    for m in re.finditer(rf"\b(?:you are|you're|i am|i'm|he is|he's|she is|she's|patient is|they're|he was|she was|you were|i was|is now|she'll be|he'll be|i'll be)\s(?:only |almost |nearly |about |over |just |now )?(?P<num>{NUM})\b", t):
        v = _age_value(m.group("num"))
        num_words = re.split(r"[ -]", m.group("num"))
        if len(num_words) == 2 and all(w in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine") for w in num_words):
            continue  # `five eight` is a height, not 13
        if _VITAL_CTX.search(t[max(0, m.start() - 60) : m.end() + 30]):
            continue  # spoken vitals, weights, heights and pain scores
        if v is not None and 1 <= v <= 110 and not _AGE_UNITS_AFTER.match(t[m.end() : m.end() + 14]):
            if re.match(r"\s?(?:-|years?|year-old)", t[m.end() : m.end() + 6]):
                continue  # covered by the year-old rule
            add(m, "num", "age_markerless", 0.85, "markerless age ('you are 74')")
    rel = "|".join(sorted(L.RELATIONS | {"oldest", "youngest", "other one", "middle one", "little one", "baby"}, key=len, reverse=True))
    for m in re.finditer(rf"\b(?:{rel})\s(?:[a-z]+\s)?(?:is|was|turned|just turned|will be|who is)\s(?:only |almost |nearly |about |just )?(?P<num>{NUM})\b", t):
        if not _AGE_UNITS_AFTER.match(t[m.end() : m.end() + 14]):
            add(m, "num", "age_relative", 0.85, "age of a relative")
    if doc.field == "note":
        for m in re.finditer(r"\b(?:age|Age|AGE)\s?:\s?(?P<num>\d{1,3})\b", doc.view):
            if not _GESTATIONAL.search(doc.lower[max(0, m.start() - 16) : m.start()]):
                add(m, "num", "age_label", 1.0, "age field")
    return out


# --------------------------------------------------------------------------- person-name anchors

TOK_RE = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]|[,.?!;:]|\d+")


def _tokens(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    return [(start + m.start(), start + m.end(), m.group(0)) for m in TOK_RE.finditer(text[start:end])]


def _clean(w: str) -> str:
    return w.lower().removesuffix("'s").removesuffix("s'").strip("-'")


def _word_after(t: str, i: int) -> str:
    m = re.match(r"\s*([a-z']+)", t[i : i + 30])
    return m.group(1) if m else ""


def name_ok(w: str, min_score: float = 0.0, allow_common: bool = False) -> bool:
    w = _clean(w)
    if len(w) < 2 or w in L.NAME_STOP or w in L.TITLES or w in L.SCRIBE_PRODUCTS or w in L.RELATIONS:
        return False
    if not re.fullmatch(r"[a-z][a-z\-]*[a-z]", w):
        return False
    if not allow_common and L.is_common_word(w):
        return False
    return L.name_score(w) >= min_score


def _name_span(doc: Doc, s: int, e: int, cat: str, rule: str, conf: float, reason: str, role_hint: str) -> Span:
    w = doc.lower[s:e]
    if w.endswith("'s"):
        e -= 2
    return Span(doc.field, s, e, cat, f"rule:{rule}", conf, reason, meta={"role": role_hint, "anchor": True})


ROLE_CAT = {"patient": "PATIENT_NAME", "clinician": "CLINICIAN_NAME", "staff": "STAFF_NAME", "relative": "RELATIVE_NAME", "person": "PERSON_NAME"}


def name_anchors(doc: Doc) -> list[Span]:
    t, out = doc.lower, []

    def emit(s, e, role, rule, conf, reason):
        out.append(_name_span(doc, s, e, ROLE_CAT[role], rule, conf, reason, role))

    NT = r"[a-z][a-z\-]*[a-z](?:'s)?"
    # 1. Title + name (+ surname): `mr . james`, `dr. jacob ward`, `Ms. Martinez`, `doctor isabella`.
    for m in re.finditer(rf"\b(?P<title>mr|mrs|ms|miss|mister|missus|dr|doctor)(?P<dot>\s?\.)?\s(?:(?:dr|doctor)\s?\.?\s)*(?P<n1>{NT})(?:\s(?P<n2>{NT}))?", t):
        title, n1, n2 = m.group("title"), m.group("n1"), m.group("n2")
        c1 = _clean(n1)
        if c1 in L.TITLE_STOP or c1 in L.SCRIBE_PRODUCTS:
            continue
        if c1 in L.NAME_STOP:  # `doctor ray castellano`, `doctor may okonkwo`: keep at least the surname
            c2 = _clean(n2) if n2 else ""
            if c2 in L.surnames() and not L.is_common_word(c2) and c2 not in L.NAME_STOP:
                role = "clinician" if title in ("dr", "doctor") else "patient"
                if c1 in L.first_names():
                    emit(m.start("n1"), m.end("n1"), role, "title_name", 0.85, f"first name after the title '{title}' followed by a surname")
                emit(m.start("n2"), m.end("n2"), role, "title_name_surname", 0.9, f"surname after the title '{title}' and a first name")
            continue
        ambiguous = title in ("miss", "ms") and not m.group("dot") or title == "doctor"
        score = L.name_score(c1)
        in_gaz = c1 in L.surnames() or c1 in L.first_names()
        if ambiguous and score < 0.5:
            if title == "doctor" and not L.is_common_word(c1) and c1 not in L.medical_vocab() and len(c1) >= 3:
                pass
            elif title == "doctor" and L.surnames().get(c1, 0) >= 2000 and L.lower_uses(c1) < 2000:
                pass  # `doctor taylor`, `doctor wood`
            elif title in ("doctor", "miss", "ms") and L.surnames().get(c1, 0) >= 2000 and _word_after(t, m.end("n1")) not in L.WORD_SENSE_NEXT:
                pass  # `doctor white said`: a surname after a title unless a collocation follows (`white blood cell`)
            elif title in ("miss", "ms") and in_gaz and L.lower_uses(c1) < 30:
                pass  # `miss taylor`: surname that is rarely an ordinary word
            else:
                continue
        if c1 in L.drug_tokens() and c1 not in L.surnames():
            continue
        role = "clinician" if title in ("dr", "doctor") else "patient"
        emit(m.start("n1"), m.end("n1"), role, "title_name", 0.97, f"name after the title '{title}'")
        if n2:
            c2 = _clean(n2)
            if c2 in L.surnames() and c2 not in L.TITLE_STOP and c2 not in L.NAME_STOP and (not L.is_common_word(c2) or c1 in L.first_names()):
                emit(m.start("n2"), m.end("n2"), role, "title_name_surname", 0.9, f"surname after the title '{title}' and a first name")
    # 2. Intro: `andrew campbell is a 59-year-old`, `so tyler is a 56 -year-old`, `Jerry Nguyen a 54-year-old`.
    NTX = rf"(?!(?:is|was|who|and)\b){NT}"
    for m in re.finditer(rf"\b(?P<names>(?:{NTX}\s){{0,2}}{NTX})\s?,?\s(?:is\s|was\s)?(?:an?|our|my)\s(?:very\s|pleasant\s|lovely\s|delightful\s|nice\s|healthy\s|sweet\s)*{NUM}\s?-?\s?(?:year|month|week|day)s?\s?-?\s?old", t):
        toks = list(re.finditer(NT, m.group("names")))
        cand = []
        for tk in reversed(toks):
            w = _clean(tk.group(0))
            if name_ok(w, 0.05, allow_common=True) or (w not in L.NAME_STOP and not L.is_common_word(w) and w not in L.medical_vocab() and len(w) > 2 and w.isalpha()):
                cand.append(tk)
            else:
                break
        for tk in cand:
            emit(m.start("names") + tk.start(), m.start("names") + tk.end(), "patient", "intro_age", 0.95, "name introduced with the patient's age")
    # 3. Dictated header: `next patient is emily hernandez .`, `kelly wood , date of birth`.
    for m in re.finditer(rf"\b(?:next patient is|next patient|patient's name is|patient name is|the patient is|patient is|patient ,|patient|today i(?:'m| am) seeing|i(?:'m| am) seeing|i'm here with|here with|seeing|this is|we have|i have|name is|name's)\s(?P<n1>{NT})\s(?P<n2>{NT})(?=\s?[,.]|\s(?:who|she|he|mrn|medical|date|dob|a\s|an\s|is\s|today|for|and|here)|$)", t, re.M):
        c1, c2 = _clean(m.group("n1")), _clean(m.group("n2"))
        if name_ok(c1, 0.2, allow_common=True) and (c2 in L.surnames() or name_ok(c2, 0.2, allow_common=True)) and c2 not in L.NAME_STOP:
            emit(m.start("n1"), m.end("n1"), "patient", "dictated_header", 0.97, "dictated header (first name)")
            emit(m.start("n2"), m.end("n2"), "patient", "dictated_header", 0.97, "dictated header (surname)")
    for m in re.finditer(rf"\b(?P<n1>{NT})\s(?P<n2>{NT})\s?[,.]?\s(?:date of birth|dob|medical record|mrn|birth date)", t):
        for g in ("n1", "n2"):
            if name_ok(m.group(g), 0.2, allow_common=True):
                emit(m.start(g), m.end(g), "patient", "header_line", 0.97, "name before a DOB/MRN cue")
    # 4. Self introduction: `my name is X`, `call me X`.
    for m in re.finditer(rf"\b(?:my name is|my name's|name is|call me|i go by|goes by)\s(?P<n1>{NT})(?:\s(?P<n2>{NT}))?", t):
        if name_ok(m.group("n1"), 0.3, allow_common=True):
            emit(m.start("n1"), m.end("n1"), "person", "self_intro", 0.95, "self introduction")
            if m.group("n2") and _clean(m.group("n2")) in L.surnames() and not L.is_common_word(_clean(m.group("n2"))):
                emit(m.start("n2"), m.end("n2"), "person", "self_intro", 0.9, "self introduction (surname)")
    # 5. Named relatives: `my son sean`, `my daughter , theresa`, `her friend diane`.
    rel = "|".join(sorted(L.RELATIONS, key=len, reverse=True))
    for m in re.finditer(rf"\b(?:my|his|her|your|our|their|the)\s(?:{rel})s?\s?,?\s(?:named\s|called\s)?(?P<n>{NT})", t):
        n = _clean(m.group("n"))
        if name_ok(n, 0.6) or (n in L.first_names() and L.first_names()[n] >= 0.02 and name_ok(n, 0.2, allow_common=True) and L.lower_uses(n) < 1000):
            emit(m.start("n"), m.end("n"), "relative", "relative_name", 0.9, "name after a relation word")
    # 6. Staff frames: `have helen come in`, `went through with madison`.
    for m in re.finditer(rf"\b(?:have|with|ask|see|tell|told|talk to|talked to|called|update(?:d)? it with|went through with|go over with|sat down with|spoke with|speak with|let|send|sending|get)\s(?P<n>{NT})\s(?:come in|will|'ll|is going|can|at the front|up front|in the front|in the back|the nurse|our nurse|my nurse|the ma|bring|draw|take|check|give|schedule|help|call|come|get|go over|do)\b", t):
        if name_ok(m.group("n"), 0.8):
            emit(m.start("n"), m.end("n"), "staff", "staff_frame", 0.85, "staff name in a care-team frame")
    for m in re.finditer(rf"\b(?P<n>{NT})\s(?:will|'ll|is going to|can|is gonna)\s(?:come in|be in|be right in|be back|bring|get you|get that|get things|take you|take your|draw|check you|go over|give you|schedule|set you|help you|print|call you|walk you|grab)\b", t):
        if name_ok(m.group("n"), 0.8):
            emit(m.start("n"), m.end("n"), "staff", "staff_will_come", 0.85, "staff name in a care-team frame")
    for m in re.finditer(rf"\b(?:talk|talked|speak|spoke|check|checked|work|worked|coordinate)\s(?:with|to)\s(?P<n>{NT})\s(?:and|about|to|,|\.|who|our|in|at)\b", t):
        if name_ok(m.group("n"), 0.8):
            emit(m.start("n"), m.end("n"), "staff", "staff_talk_with", 0.8, "named colleague / staff")
    staff = "|".join(sorted(L.STAFF_WORDS, key=len, reverse=True))
    for m in re.finditer(rf"\b(?:{staff})\s?,?\s(?:named\s)?(?P<n>{NT})\b(?=\s(?:will|is|can|here|come|comes|has|to|and|'ll|who|give|bring|take|draw|check|get|help|call|schedule|see|do|go|let|be|walk|show|set|put|print|send|grab|said|says|told|mentioned|did|does|was)|\s?[,.])", t):
        if name_ok(m.group("n"), 0.7):
            emit(m.start("n"), m.end("n"), "staff", "staff_title", 0.85, "name after a staff role")
    if doc.field == "dialogue":
        out += _dialogue_vocatives(doc)
    if doc.field == "note" or doc.cased:
        out += _note_names(doc)
        out += _cased_intros(doc)
    return out


def _cased_intros(doc: Doc) -> list[Span]:
    """Typed text: `my name is Will Price`, `I'm Chase`, `this is Grace White` (casing disambiguates)."""
    v, out = doc.view, []
    CAP = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"
    for m in re.finditer(rf"\b(?:[Mm]y name is|[Mm]y name's|I'm|I am|[Tt]his is|[Cc]all me|[Ii]t's)[ \t](?P<n1>{CAP})(?:[ \t](?P<n2>{CAP}))?", v):
        for g in ("n1", "n2"):
            w = m.group(g)
            if not w:
                continue
            c = w.lower()
            if c in L.NAME_STOP and c not in L.COMMON_EXTRA or c in L.TITLES or c in L.SCRIBE_PRODUCTS or (c not in L.first_names() and c not in L.surnames()):
                break
            if c in L.MONTHS_FULL or c in L.places() and c not in L.first_names():
                break
            out.append(_name_span(doc, m.start(g), m.end(g), "PATIENT_NAME", "cased_self_intro", 0.9, "capitalised name in a self introduction", "patient"))
    for m in re.finditer(rf"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss)\.?[ \t](?P<n1>{CAP})(?:[ \t](?P<n2>{CAP}))?", v):
        for g in ("n1", "n2"):
            w = m.group(g)
            if w and w.lower() not in L.TITLE_STOP and (w.lower() in L.first_names() or w.lower() in L.surnames()):
                role = "clinician" if m.group(0).lower().startswith(("dr", "doctor")) else "patient"
                out.append(_name_span(doc, m.start(g), m.end(g), ROLE_CAT[role], "cased_title_name", 0.95, "capitalised name after a title", role))
            elif w:
                break
    return out


_DET_PREV = set("a an the this that these those my your his her our their its no any some every each good bad great nice new old same whole of in on at by for to".split())
_GREET = {"hi", "hello", "hey", "hiya", "howdy", "morning", "afternoon", "evening", "welcome", "meet", "see"}
_GREET_SKIP = {"there", "again", "good", "morning", "afternoon", "evening", "um", "uh", "so", "and", ",", ".", "!", "back", "you", "to", "nice", "great", "oh", "yeah", "hi", "hey", "hello"}
_VOC_PRE = {"so", "okay", "ok", "alright", "well", "thanks", "thank", "bye", "goodbye", "right", "yes", "yeah", "now", "great", "good", "perfect", "sure", "oh", "listen", "look", "sorry", "and", "all"}
_VOC_POST = {",", ".", "?", "!", "so", "i", "i'm", "it", "it's", "how", "what", "you", "we", "let's", "let", "are", "do", "can", "the", "if", "your", "we're", "that", "this", "there", "just", "when", "um", "uh", "and", "is", "have", "did", "nice", "good", "great", "hi", "hey", "for", "come", "go", "please", "my", "thanks", "thank", "any", "were", "was", "had", "you're", "you've", "i've", "i'll", "i'd"}


def _dialogue_vocatives(doc: Doc) -> list[Span]:
    """Bare vocatives in lowercase dialogue (`alright scott so`, `okay , tyler , i'm examining`)."""
    out, t = [], doc.lower
    prev_text = ""
    for ln in doc.lines:
        toks = _tokens(t, ln.text_start, ln.end)
        words = [w for _, _, w in toks]
        for i, (s, e, w) in enumerate(toks):
            c = _clean(w)
            if not re.fullmatch(r"[a-z][a-z\-]*[a-z]", c):
                continue
            prev = words[i - 1] if i else "<s>"
            prev2 = words[i - 2] if i > 1 else "<s>"
            nxt = words[i + 1] if i + 1 < len(words) else "</s>"
            score = L.name_score(c)
            common = L.is_common_word(c)
            if score <= 0 or c in L.NAME_STOP:
                continue
            # greeting within 4 tokens before (skipping fillers)
            j, greeted = i - 1, False
            while j >= 0 and i - j <= 4:
                if words[j] in _GREET:
                    greeted = True
                    break
                if words[j] not in _GREET_SKIP:
                    break
                j -= 1
            boundary_after = nxt in _VOC_POST or nxt == "</s>"
            if greeted and ((boundary_after and (not common or ln.idx <= 3) and score >= 0.25) or (not common and score >= 0.8)):
                out.append(_name_span(doc, s, e, "PATIENT_NAME", "greeting_name", 0.9 if not common else 0.75, "name after a greeting", "patient"))
                # optional surname right after
                if nxt not in _VOC_POST and nxt != "</s>":
                    pass
                elif i + 1 < len(toks) and _clean(words[i + 1]) in L.surnames() and not L.is_common_word(_clean(words[i + 1])):
                    s2, e2, _ = toks[i + 1]
                    out.append(_name_span(doc, s2, e2, "PATIENT_NAME", "greeting_surname", 0.85, "surname after greeted first name", "patient"))
                continue
            if common:
                continue
            if (prev in _VOC_PRE or prev in (",", "<s>")) and boundary_after and score >= 0.7:
                conf = 0.85 if prev in _VOC_PRE or prev == "," else 0.75
                out.append(_name_span(doc, s, e, "PERSON_NAME", "vocative", conf, "vocative name (no punctuation anchor)", "patient"))
                continue
            if all(not re.match(r"[a-z0-9]", w2) for w2 in words[i + 1 :]) and prev not in ("a", "an", "the", "my", "your", "his", "her", "of", "to", "in", "on", "at") and score >= 0.9:
                out.append(_name_span(doc, s, e, "PERSON_NAME", "turn_final_vocative", 0.7, "name at the end of a turn", "person"))
                continue
            if score >= 0.9 and len(c) >= 3 and c not in L.medical_vocab() and c not in L.places() and L.lower_uses(c) <= 3 and prev not in _DET_PREV:
                out.append(_name_span(doc, s, e, "PERSON_NAME", "gazetteer_name", 0.6, "frequent person name not used as an ordinary word", "person"))
        # one-word answer after "how do you say/spell your last name"
        if re.search(r"\b(?:last name|your name|spell (?:it|that|your))\b", prev_text) and 1 <= len([w for w in words if re.fullmatch(r"[a-z\-]+", w)]) <= 2:
            for s, e, w in toks:
                if re.fullmatch(r"[a-z][a-z\-]+", w) and w not in L.NAME_STOP and not L.is_common_word(w):
                    out.append(_name_span(doc, s, e, "PATIENT_NAME", "name_answer", 0.85, "answer to a 'your name' question", "patient"))
        prev_text = t[ln.text_start : ln.end]
    return out


def _note_names(doc: Doc) -> list[Span]:
    """Cased note patterns: `Andrew Campbell is a ...`, labelled fields, capitalised First Last."""
    v, out = doc.view, []
    CAP = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:'s)?"
    for m in re.finditer(rf"\b(?:Patient(?: Name)?|Name|Pt|Patient name|Patient's name|Re)[ \t]?:[ \t]?(?P<names>{CAP}(?:[ \t](?:[A-Z]\.?[ \t])?{CAP}){{0,3}})", v):
        for tk in re.finditer(CAP, m.group("names")):
            out.append(_name_span(doc, m.start("names") + tk.start(), m.start("names") + tk.end(), "PATIENT_NAME", "name_field", 0.97, "patient name field", "patient"))
    for m in re.finditer(rf"\b(?:Attending|Attending Physician|Physician|Doctor|Provider|Surgeon|Signed|Signed by|Electronically signed by|Dictated by|Referring physician|Referring Physician|PCP|Consultant|Resident|Nurse|RN|Treating Physician|Discharging Physician|Consultant Physician)[ \t]?:[ \t]?(?:Dr\.?[ \t])?(?P<names>{CAP}(?:[ \t](?:[A-Z]\.?[ \t])?{CAP}){{0,2}})|(?:Yours sincerely|Yours faithfully|Sincerely|Regards|Best regards|Kind regards|Many thanks|Thanks),[ \t]?\n?[ \t]?(?:Dr\.?[ \t])?(?P<names2>{CAP}(?:[ \t](?:[A-Z]\.?[ \t])?{CAP}){{0,2}})", v):
        grp = "names" if m.group("names") else "names2"
        for tk in re.finditer(CAP, m.group(grp)):
            w = tk.group(0).lower().removesuffix("'s")
            if w in L.NAME_STOP or w in L.STAFF_WORDS or (w not in L.first_names() and w not in L.surnames() and (L.is_common_word(w) or w in L.medical_vocab())):
                break
            out.append(_name_span(doc, m.start(grp) + tk.start(), m.start(grp) + tk.end(), "CLINICIAN_NAME", "clinician_field", 0.95, "clinician name field", "clinician"))
    # Capitalised First Last bigrams (not sentence-initial requirement: both must look like names).
    for m in re.finditer(rf"\b(?P<f>[A-Z][a-z]+)[ \t](?P<l>[A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b", v):
        f, l_ = m.group("f").lower(), m.group("l").lower()
        if f in L.first_names() and l_ in L.surnames() and f not in L.NAME_STOP and l_ not in L.NAME_STOP and f not in L.medical_vocab() and l_ not in L.drug_tokens():
            if L.first_names()[f] >= 0.003 and not (L.is_common_word(f) and L.is_common_word(l_)):
                out.append(_name_span(doc, m.start("f"), m.end("f"), "PERSON_NAME", "cap_first_last", 0.85, "capitalised first + last name", "person"))
                out.append(_name_span(doc, m.start("l"), m.end("l"), "PERSON_NAME", "cap_first_last", 0.85, "capitalised first + last name", "person"))
    # Single capitalised name-like words not at sentence start (weak; corroborated by the identity graph).
    for m in re.finditer(r"\b[A-Z][a-z]{2,}(?:-[A-Z][a-z]+)?\b", v):
        w = m.group(0).lower()
        if is_sentence_start(v, m.start()) or w in L.NAME_STOP or L.is_common_word(w):
            continue
        if L.name_score(w) >= 0.8 and w not in L.medical_vocab() and w not in L.drug_tokens() and w not in L.MEDICAL_EPONYMS:
            out.append(_name_span(doc, m.start(), m.end(), "PERSON_NAME", "cap_name", 0.65, "capitalised name-like word mid-sentence", "person"))
    return out


def detect(doc: Doc) -> list[Span]:
    """All rule detectors for one field (quasi-identifiers are in `quasi.py`)."""
    d = dates(doc)
    return d + dob(doc, d) + years(doc) + identifiers(doc) + contacts(doc) + ages(doc) + name_anchors(doc)


__all__ = ["detect", "dates", "dob", "years", "identifiers", "contacts", "ages", "name_anchors", "name_ok", "ordinal_to_int"]
