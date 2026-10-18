"""Deterministic generalization of ages and dates (anonymize.py --mode generalize).

Instead of deleting an age or a date with an [AGE] / [DATE] tag, generalize mode replaces it with a
vague range. The exact value is gone, but the medically useful part (child vs adult vs elderly,
season, decade) stays readable for an ICD-10 coder.

  age (years)    band                  example
  < 1 month      newborn               "3-week-old"            -> "newborn"
  1-11 months    infant                "a 9-month-old"         -> "an infant"
  1-2 years      toddler               "22-month-old"          -> "toddler"
  3-4            preschool-age         "since she was 4"       -> "since she was preschool-age"
  5-12           school-age            "an 8-year-old boy"     -> "a school-age boy"
  13-17          teenage               "14 years old"          -> "teenage"
  18-19          young-adult
  20-89          <decade>-something    "44-year-old female"    -> "forty-something female"
  >= 90          ninety-plus           HIPAA Safe Harbor groups every age over 89 into "90 or older"

  date with a year     early/mid/late <decade>s   "4/24/2021" -> "early 2020s", "09/18/1949" -> "late 1940s"
                       (year % 10: 0-3 early, 4-6 mid, 7-9 late)
  month(s), no year    season(s)                  "july 8th" -> "summer", "november and december" -> "autumn and winter"
  other calendar date  [DATE]                     "the 28th", "monday", "my birthday", "6421"
  no calendar anchor   kept as is                 "two years ago", "48 hours", "today", "40 milligrams", "180 over 95"
                       (relative time and measurements the LLM mislabels as DATE: clinically relevant, not identifying)

The bands are words, not digit ranges such as "40-49", for two reasons: no digit of the true age
survives (a range endpoint can equal the true age), and the evaluator counts a gold string such as
"age 40" or "49-year-old" as leaked wherever it occurs as whole words, so "age 40-49" would score as a leak.

find_spans() is the regex pass. It finds the unambiguous age and date forms that the LLM tends to
miss ("N-year-old", "N years old", "age N", "since she was N", "turned N", "i'm almost N", numeric
dates, month + day, month + year). It leaves alone guideline ages ("under the age of 50",
"9 to 11 years old"), gestational ages ("34+3-week-old"), predicative week/day ages (a "1 week old"
abrasion), pain scales (7/10), blood pressure (120/80), durations ("2 years ago", "10-year history",
"3-month follow-up"), decades ("in his 40s", "in the 80s") and bare month names or years ("may",
"march", "2000 mg"); those are left to the LLM. render(text, type) is the replacement callable for
anonymize.apply_entities().
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- numbers

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
          "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
WORD_VALUE = {**{w: i for i, w in enumerate(_UNITS)}, **{w: 10 + i for i, w in enumerate(_TEENS)}, **_TENS}

_UNIT_RE = "|".join(_UNITS[1:])
_TEEN_RE = "|".join(_TEENS)
_TENS_RE = "|".join(_TENS)
# Longest alternatives first so "seventy" is not read as "seven", "fourteen" not as "four".
NUM_WORD = rf"(?:(?:{_TENS_RE})(?:[ -](?:{_UNIT_RE}))?|{_TEEN_RE}|{_UNIT_RE})"
NUM = rf"(?:(?<![\w.])(?:\d{{1,3}}|{NUM_WORD})(?![a-z\d]))"   # "40s", "1.5", "eighth" do not match


def parse_num(s: str) -> int | None:
    """'44' -> 44, 'sixteen' -> 16, 'twenty one' / 'twenty-one' -> 21; None if not a number."""
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    total = 0
    for w in re.split(r"[ -]+", s):
        if w not in WORD_VALUE:
            return None
        total += WORD_VALUE[w]
    return total


# ---------------------------------------------------------------- age bands

DECADE_WORD = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty"}
# Noun phrase used when "age N" / "the age of N" is a child's age ("at age 7" -> "at school age").
BAND_NOUN = {"newborn": "the newborn period", "infant": "infancy", "toddler": "toddlerhood",
             "preschool-age": "preschool age", "school-age": "school age", "teenage": "teenage years",
             "young-adult": "young adulthood"}
MONTHS_PER = {"year": 12.0, "yr": 12.0, "month": 1.0, "mo": 1.0, "week": 12 / 52, "wk": 12 / 52, "day": 12 / 365}


def age_band(months: float) -> str:
    """Band adjective for an age given in months."""
    if months < 1:
        return "newborn"
    if months < 12:
        return "infant"
    if months < 36:
        return "toddler"
    years = int(months // 12)
    if years <= 4:
        return "preschool-age"
    if years <= 12:
        return "school-age"
    if years <= 17:
        return "teenage"
    if years <= 19:
        return "young-adult"
    if years < 90:
        return f"{DECADE_WORD[years // 10]}-something"
    return "ninety-plus"


def is_adult_band(band: str) -> bool:
    return band.endswith("-something") or band == "ninety-plus"


# ---------------------------------------------------------------- age patterns

_ART = r"(?:(?P<art>\b(?:a|an))[ \t]+)?"
_N = rf"(?P<n1>{NUM})(?:[ \t]+or[ \t]+(?P<n2>{NUM}))?"                     # "4 or 5" -> max
_MOD = r"(?:(?P<mod>almost|nearly|around|about|approximately|like)[ \t]+)?"
_YO = r"(?P<yo>[ \t]+(?:years?|yrs?)[ \t]+old)?"
# A bare number only counts as an age when the clause ends after it ("when i was 18 .", "i'm 44 now").
_FOLLOW = (r"(?=[ \t]*(?:[.,;:!?)\]\n]|$|(?:now|and|but|so|when|then|or|because|in|at|last|this|next|"
           r"i|he|she|they|we|you|my|his|her)\b))")
# Guideline ages, ranges and gestational ages: "under the age of 50", "9 to 11 years old", "34+3-week-old".
_GUARD = re.compile(r"(?:\b(?:under|over|above|below|than|until|till|between|to|from|or|ages)|\d[ \t]*(?:\+|and|[-–]))"
                    r"[ \t]*(?:the[ \t]+)?$", re.IGNORECASE)

AGE_PATTERNS = [
    # "44-year-old", "34 -year-old", "62 year old", "14 years old", "10 years of age", "22-month-old", "3-week-old"
    ("unit", re.compile(rf"{_ART}{_N}(?P<sep1>[ \t]*-?[ \t]*)(?P<unit>years?|yrs?|months?|mos?|weeks?|wks?|days?)"
                        rf"(?P<sep2>[ \t]*-?[ \t]*)(?P<tail>old|of[ \t]+age)\b", re.IGNORECASE)),
    # "75 y.o.", "75 y/o"
    ("yo", re.compile(rf"{_ART}{_N}[ \t]*(?:y\.[ \t]?o\b\.?|y/o\b)", re.IGNORECASE)),
    # "age 45", "aged 45", "age of 27"
    ("phrase", re.compile(rf"\b(?P<pre>age(?:[ \t]+of)?|aged)[ \t]+{_N}(?![ \t]*(?:[-/%]|\.\d|to\b|years?\b|"
                          rf"months?\b|weeks?\b|days?\b|mg\b|percent\b))", re.IGNORECASE)),
    # "since she was 4", "when i was 18", "since i was around 12", "when she was four years old"
    ("clause", re.compile(rf"\b(?P<pre>(?:since|when)[ \t]+(?:i|he|she|you|they|we)[ \t]+(?:was|were)[ \t]+)"
                          rf"{_MOD}{_N}{_YO}{_FOLLOW}", re.IGNORECASE)),
    # "i'm 44 now", "she is almost sixteen", "my mom was almost 45 when"
    ("clause", re.compile(rf"\b(?P<pre>(?:i'm|i[ \t]+am|he's|she's|he[ \t]+is|she[ \t]+is|(?:mom|mother|dad|father|"
                          rf"brother|sister|son|daughter|wife|husband)[ \t]+(?:is|was))[ \t]+)"
                          rf"(?:(?P<mod>almost|nearly)[ \t]+)?{_N}{_YO}{_FOLLOW}", re.IGNORECASE)),
    # "when you turn 50", "turned 80 last week"
    ("clause", re.compile(rf"\b(?P<pre>turn(?:s|ed|ing)?[ \t]+){_N}{_YO}{_FOLLOW}", re.IGNORECASE)),
]


def _age_value(m: re.Match) -> tuple[str | None, str | None]:
    """(band, unit) for an age match, or (None, None) if the number is not a plausible age."""
    g = m.groupdict()
    nums = [parse_num(x) for x in (g.get("n1"), g.get("n2")) if x]
    if not nums or None in nums:
        return None, None
    n = max(nums)
    if (g.get("mod") or "").lower() in ("almost", "nearly"):
        n -= 1
    unit = (g.get("unit") or "year").lower().rstrip("s")
    if n <= 0 or (unit in ("year", "yr") and n > 120):
        return None, None
    return age_band(n * MONTHS_PER[unit]), unit


def _with_article(art: str | None, words: str) -> str:
    if not art:
        return words
    new = "an" if words[:1].lower() in "aeiou" else "a"
    return (new.capitalize() if art[:1].isupper() else new) + " " + words


def _age_replacement(kind: str, m: re.Match) -> str | None:
    band, unit = _age_value(m)
    if band is None:
        return None
    g = m.groupdict()
    adult = is_adult_band(band)
    if kind == "unit":
        hyphen = "-" in (g["sep1"] + g["sep2"])
        tail = re.sub(r"\s+", " ", g["tail"].lower())
        if unit in ("week", "wk", "day") and not hyphen:  # "appears to be 1 week old" (a wound, not a person)
            return None
        predicative = tail == "of age" or (not hyphen and g["unit"].lower().endswith("s"))
        if predicative and adult:
            return _with_article(g["art"], f"{band} years {tail}")
        return _with_article(g["art"], band)
    if kind == "yo":
        return _with_article(g["art"], band)
    if kind == "phrase":
        pre = g["pre"]
        if adult:
            return f"{pre} {band}"
        noun = BAND_NOUN[band]
        return f"of {noun}" if pre.lower() == "aged" else noun
    # clause: keep the lead-in, drop "almost"/"around", keep "years old" only for adults
    yo = (g.get("yo") or "") if adult else ""
    return f"{g['pre']}{band}{yo}"


def _resolve(hits: list[tuple]) -> list[tuple]:
    """Keep non-overlapping hits, earliest start first, longest first at the same start."""
    out, end = [], -1
    for h in sorted(hits, key=lambda h: (h[0], -h[1])):
        if h[0] >= end:
            out.append(h)
            end = h[1]
    return out


def _age_hits(text: str) -> list[tuple]:
    hits = []
    for kind, pat in AGE_PATTERNS:
        for m in pat.finditer(text):
            if _GUARD.search(text[:m.start()]):
                continue
            rep = _age_replacement(kind, m)
            if rep is not None:
                hits.append((m.start(), m.end(), "AGE", rep))
    return _resolve(hits)


# ---------------------------------------------------------------- dates

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
SEASON = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer",
          9: "autumn", 10: "autumn", 11: "autumn", 12: "winter"}
_MONTH_RE = "|".join(MONTHS)
_ORD_UNITS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
_ORD_WORD = (r"(?:(?:twenty|thirty)[ -](?:" + "|".join(_ORD_UNITS) + r")|twentieth|thirtieth|tenth|eleventh|"
             r"twelfth|(?:thir|four|fif|six|seven|eigh|nine)teenth|" + "|".join(_ORD_UNITS) + r")")
_DAY_ORD = rf"(?:\d{{1,2}}(?:st|nd|rd|th)|{_ORD_WORD})"
_YEAR4 = r"(?:19\d\d|20[0-4]\d)"
_NOT_DOSE = r"(?![\d%]|[ \t]*(?:mg|milligrams?|mcg|iu|units?|ml|cc|percent)\b)"
_SPOKEN_YEAR = (rf"(?:nineteen[ \t]+(?:oh[ \t]+(?:{_UNIT_RE})|(?:{_TENS_RE})(?:[ -](?:{_UNIT_RE}))?|{_TEEN_RE})"
                rf"|twenty[ \t]+(?:oh[ \t]+(?:{_UNIT_RE})|(?:twenty|thirty|forty)(?:[ -](?:{_UNIT_RE}))?|{_TEEN_RE})"
                rf"|two[ \t]+thousand(?:[ \t]+(?:and[ \t]+)?(?:{_TEEN_RE}|(?:twenty|thirty|forty)(?:[ -](?:{_UNIT_RE}))?"
                rf"|{_UNIT_RE}))?)(?![\w-])")

# Regex pass (unambiguous forms only).
DATE_NUMERIC = re.compile(r"(?<![\w/.])(?P<m>\d{1,2})(?P<s>[/-])(?P<d>\d{1,2})(?P=s)(?P<y>\d{4}|\d{2})(?![\w/])"
                          r"|(?<![\w/.])(?P<m2>\d{1,2})/(?P<y2>\d{4})(?![\w/])")
DATE_MONTH = re.compile(
    rf"(?:\b(?P<pord>(?:the[ \t]+)?{_DAY_ORD})[ \t]+of[ \t]+)?\b(?P<mon>{_MONTH_RE})\b"
    rf"(?:[ \t]+(?:the[ \t]+)?(?P<day>{_DAY_ORD}\b|\d{{1,2}}(?![\d.]\d|[ \t]*(?:[%/-]|mg\b|milligrams?\b|times\b|"
    rf"units?\b|percent\b|pounds?\b|minutes?\b|hours?\b|days?\b|weeks?\b|months?\b|years?\b|out\b|to\b|or\b))))?"
    rf"(?:[ \t]*,?[ \t]*(?:of[ \t]+)?(?P<y>{_YEAR4}{_NOT_DOSE}|'[ \t]?\d\d\b))?", re.IGNORECASE)

# Inside a span (render): anything carrying a year collapses to its decade bucket.
_YEARFUL = re.compile(
    rf"(?:\b(?:the[ \t]+)?{_DAY_ORD}[ \t]+of[ \t]+)?(?:\b(?:{_MONTH_RE})\b(?:[ \t]*,)?[ \t]+(?:of[ \t]+)?)?"
    rf"(?:\b(?:the[ \t]+)?(?:{_DAY_ORD}|\d{{1,2}})(?:[ \t]*,)?[ \t]+)?"
    rf"(?P<y>(?<![\d.,/$])\b{_YEAR4}{_NOT_DOSE}|'[ \t]?\d\d\b|\b{_SPOKEN_YEAR})", re.IGNORECASE)
_MONTH_DAY = re.compile(rf"(?:\b(?:the[ \t]+)?{_DAY_ORD}[ \t]+of[ \t]+)?\b(?P<mon>{_MONTH_RE})\b"
                        rf"(?:[ \t]+(?:the[ \t]+)?(?:{_DAY_ORD}|\d{{1,2}})\b)?", re.IGNORECASE)
_MONTH_DAY_NUM = re.compile(r"(?<![\w/.])(?P<m>\d{1,2})/(?P<d>\d{1,2})(?![\w/])")
_SEASON_DUP = re.compile(r"\b(winter|spring|summer|autumn)\b(?:[ \t]*(?:,|and|or|to|through|-)[ \t]*)\1\b",
                         re.IGNORECASE)


def spoken_year(s: str) -> int | None:
    """'twenty eighteen' -> 2018, 'nineteen oh five' -> 1905, 'two thousand eighteen' -> 2018."""
    w = re.split(r"[ \t]+", s.strip().lower(), maxsplit=1)
    if s.lower().startswith("two"):
        rest = re.sub(r"^two[ \t]+thousand(?:[ \t]+and)?[ \t]*", "", s.strip().lower())
        return 2000 + (parse_num(rest) or 0) if rest else 2000
    if len(w) != 2:
        return None
    rest = re.sub(r"^oh[ \t]+", "", w[1])
    n = parse_num(rest)
    return None if n is None else (1900 if w[0] == "nineteen" else 2000) + n


def year_of(token: str) -> int | None:
    t = token.strip().lower()
    if t.isdigit():
        y = int(t)
        return y if len(t) == 4 else (2000 + y if y <= 30 else 1900 + y)
    if t.startswith("'"):
        y = int(re.sub(r"\D", "", t))
        return 2000 + y if y <= 30 else 1900 + y
    return spoken_year(t)


def decade_phrase(year: int) -> str:
    r = year % 10
    return f"{'early' if r <= 3 else 'mid' if r <= 6 else 'late'} {year // 10 * 10}s"


def _numeric_year(m: re.Match) -> int | None:
    """Year of a DATE_NUMERIC match if month/day are valid, else None (so 7/10 or 120/80 never qualify)."""
    if m.group("m2"):
        mon, day, y = int(m.group("m2")), 1, int(m.group("y2"))
        if not 1900 <= y <= 2049:
            return None
    else:
        mon, day, y = int(m.group("m")), int(m.group("d")), year_of(m.group("y"))
        if len(m.group("y")) == 4 and not 1900 <= y <= 2049:
            return None
    return y if 1 <= mon <= 12 and 1 <= day <= 31 else None


def _date_hits(text: str) -> list[tuple]:
    hits = []
    for m in DATE_NUMERIC.finditer(text):
        y = _numeric_year(m)
        if y is not None:
            hits.append((m.start(), m.end(), "DATE", decade_phrase(y)))
    for m in DATE_MONTH.finditer(text):
        day, y, mon = m.group("day"), m.group("y"), m.group("mon").lower()
        if not (day or y or m.group("pord")):
            continue  # a bare month name is too ambiguous ("may", "march") for the regex pass
        if mon == "may" and not y and not re.match(r"\d+(?:st|nd|rd|th)$", day or "", re.IGNORECASE):
            continue  # "you may first ...", "may 2 ..."
        if day and day[:1].isdigit() and not 1 <= int(re.match(r"\d+", day).group()) <= 31:
            continue
        rep = decade_phrase(year_of(y)) if y else SEASON[MONTHS.index(mon) + 1]
        hits.append((m.start(), m.end(), "DATE", rep))
    return _resolve(hits)


# ---------------------------------------------------------------- span generalizers

def _splice(text: str, hits: list[tuple]) -> str:
    out, last = [], 0
    for s, e, _, rep in hits:
        out.append(text[last:s])
        out.append(rep)
        last = e
    out.append(text[last:])
    return "".join(out)


_DECADE_ALREADY = re.compile(r"(?:(?:in|at)[ \t]+(?:his|her|my|their|your|the)[ \t]+)?(?:(?:early|mid|late)[ \t-]+)?"
                             r"(?:\d0'?s|teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)",
                             re.IGNORECASE)
_BARE_AGE = re.compile(rf"(?:(?P<mod>about|around|almost|nearly|approximately|like)[ \t]+)?(?P<n1>{NUM})"
                       rf"(?:[ \t]+(?:or[ \t]+)?(?P<n2>{NUM}))?(?:[ \t]+now)?", re.IGNORECASE)


def generalize_age(span: str) -> str | None:
    """Generalize the ages inside an AGE span; None if nothing in it parses as an age."""
    hits = _age_hits(span)
    if hits:
        return _splice(span, hits)
    if _DECADE_ALREADY.fullmatch(span.strip()):
        return span  # "in his 40s" is already a decade
    m = _BARE_AGE.fullmatch(span.strip())
    if m:
        band, _ = _age_value(m)
        return band
    return None


def generalize_date(span: str) -> str | None:
    """Generalize a DATE span: a year -> decade bucket, months -> seasons; None if neither is present."""
    hits = [(m.start(), m.end(), "DATE", decade_phrase(y)) for m in DATE_NUMERIC.finditer(span)
            if (y := _numeric_year(m)) is not None]
    hits += [(m.start(), m.end(), "DATE", decade_phrase(year_of(m.group("y")))) for m in _YEARFUL.finditer(span)
             if year_of(m.group("y")) is not None]
    hits = _resolve(hits)
    out = _splice(span, hits)
    n_month = 0

    def season(m):
        nonlocal n_month
        n_month += 1
        return SEASON[MONTHS.index(m.group("mon").lower()) + 1]

    out = _MONTH_DAY.sub(season, out)
    if not hits and not n_month:
        mds = [m for m in _MONTH_DAY_NUM.finditer(out)
               if 1 <= int(m.group("m")) <= 12 and 1 <= int(m.group("d")) <= 31]
        if not mds:
            return None
        out = _splice(out, [(m.start(), m.end(), "DATE", SEASON[int(m.group("m"))]) for m in mds])
    while _SEASON_DUP.search(out):
        out = _SEASON_DUP.sub(r"\1", out)
    return out


_CALENDAR = re.compile(
    rf"\b(?:{_MONTH_RE}|(?:jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?(?![a-z])"
    rf"|(?:mon|tues|wednes|thurs|fri|satur|sun)days?|christmas|thanksgiving|easter|halloween|new[ \t]+year'?s?"
    rf"|birthdays?|anniversary)\b"
    rf"|(?<!\d)\d{{1,2}}(?:st|nd|rd|th)\b|\b(?:the|on)[ \t]+{_ORD_WORD}\b"     # "the 28th", "on the twenty first"
    rf"|(?<![\d.])\d{{1,2}}[/-]\d{{1,2}}(?![\d.])"                                 # "4/24", "10-10" (not 180/95)
    rf"|(?<![\d.,/$])\b{_YEAR4}{_NOT_DOSE}|'[ \t]?\d\d\b|\b{_SPOKEN_YEAR}"
    rf"|^\W*\d{{3,8}}\W*$",                                          # a bare "6421" may be a garbled date
    re.IGNORECASE)


def has_calendar(span: str) -> bool:
    """True if a DATE span names a point in the calendar (month, year, day, weekday, holiday, numeric date).

    Relative time ("two years ago", "the last 48 hours", "today") and the measurements a small LLM
    mislabels as DATE ("40 milligrams", "ninety eight percent", "180 over 95") are not: they describe
    onset, duration and findings, which ICD-10 coding needs, and do not identify anyone.
    """
    return _CALENDAR.search(span) is not None


def _match_case(src: str, out: str) -> str:
    if src[:1].isupper() and out[:1].islower():
        return out[0].upper() + out[1:]
    return out


def render(text: str, typ: str) -> str:
    """Replacement for one occurrence of an entity: generalized AGE/DATE, [TYPE] tag for everything else.

    A DATE span without any calendar anchor comes back unchanged (see has_calendar).
    """
    out = None
    if typ == "AGE":
        out = generalize_age(text)
    elif typ == "DATE":
        out = generalize_date(text)
        if out is None:  # the LLM sometimes types an age ("75 y.o.", "49-year-old") as DATE
            hits = _age_hits(text)
            out = _splice(text, hits) if hits else None
        if out is None and not has_calendar(text):
            return text
    return f"[{typ}]" if out is None else _match_case(text, out)


REASONS = {
    "PERSON": "name of a person: a direct identifier",
    "CONTACT": "phone number, email or URL: a direct identifier",
    "ID": "record or ID number: a direct identifier",
    "DATE": "exact date: temporal information that helps re-identify the patient",
    "AGE": "exact age: a quasi-identifier",
    "LOCATION": "place: a quasi-identifier (non-medical context)",
    "ORGANIZATION": "employer, school, clinic or business: a quasi-identifier (non-medical context)",
    "OCCUPATION": "job: a quasi-identifier (non-medical context)",
}
KEPT_PART = {"DATE": "only the decade or season is kept, which is enough for disease course and onset",
             "AGE": "only the age band is kept (child / teenager / decade of adult life), which ICD-10 coding needs"}


def explain(typ: str, action: str) -> str:
    """Reason for one anonymization decision (logged per span in generalize mode)."""
    if action == "retained" and typ == "DATE":
        return ("no calendar date (relative time such as 'two years ago', or a measurement the model labelled DATE): "
                "onset, duration and findings matter for ICD-10 coding and do not identify the patient")
    why = REASONS.get(typ, "personal information")
    if action == "generalized":
        return f"{why}; {KEPT_PART.get(typ, 'generalized')}"
    return f"{why}; replaced by the [{typ}] placeholder"


def find_spans(chunk: str) -> list[dict]:
    """Regex pass: [{"text", "type"}] for every unambiguous age or date in the chunk, in order."""
    hits = _resolve(_age_hits(chunk) + _date_hits(chunk))
    return [{"text": chunk[s:e], "type": typ} for s, e, typ, _ in hits]
