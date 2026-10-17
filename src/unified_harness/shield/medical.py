"""Medical shield: lock what must survive *before* anything is anonymized.

The shield inverts the usual de-identification order. It marks clinical spans that no
anonymization edit may touch:

- clinical numbers (BP, doses, labs, scores, grades, murmurs, spinal levels, durations),
- eponyms used in medical syntax (`lyme titer`, `murphy's sign`, `thompson test`, `baker's cyst`),
- code evidence: words of the encounter's own ICD-10 descriptions and of the ICD-10-CM index entries
  that lead to those codes (`Y93.53 Activity, golf` locks `golfing`; `L21.0` locks `dandruff`),
- drug names (FDA NDC + ICD drug table),
- without codes: activity words inside an injury context (`hurt my back while golfing`).

A detection that overlaps a hard shield is dropped and logged as `KEEP (shielded)`, which also gives
an explanation for every retained look-alike.
"""

from __future__ import annotations

import re
from functools import cache

from .. import lexicons as L
from ..spans import Shield
from ..text.normalize import NUM, NUMWORDS, Doc

UNITS = (
    r"(?:mg|milligrams?|mcg|micrograms?|ml|milliliters?|cc|units?|iu|%|percent|mmhg|cm|centimeters?|mm|"
    r"millimeters?|kg|kilograms?|kilos?|lbs?|pounds?|grams?|oz|ounces?|inch(?:es)?|feet|foot|ft|years?|yrs?|"
    r"degrees?|bpm|beats|breaths|liters?|meq|mmol|ng|pg|mg/dl|g/dl|times|puffs?|tablets?|tabs?|pills?|"
    r"capsules?|caps|drops?|sprays?|hours?|hrs?|days?|weeks?|wks?|months?|mos?|minutes?|mins?|seconds?|secs?|"
    r"steps|miles|blocks|calories|cigarettes|packs?|drinks|beers|glasses|cups|servings|shots|refills?|"
    r"doses?|injections?|sessions?|visits|stitches|sutures|centimeter|millimeter|meters?|nights?|"
    r"teaspoons?|tablespoons?|cans|bottles|gallons|fl oz|out of)"
)
LABS = (
    r"(?:a1c|hba1c|hemoglobin a1c|hemoglobin|hgb|hematocrit|hct|creatinine|egfr|gfr|bnp|pro-bnp|ldl|hdl|"
    r"cholesterol|triglycerides?|glucose|sugars?|blood sugar|potassium|sodium|chloride|bicarb|tsh|t4|t3|psa|"
    r"wbc|white count|white blood cell count|platelets?|plt|ejection fraction|ef|inr|ptt|troponin|"
    r"vitamin d|b12|ferritin|iron|calcium|magnesium|phosphorus|alt|ast|alk phos|bilirubin|albumin|protein|"
    r"bun|lactate|lipase|amylase|saturation|sat|sats|o2|oxygen|spo2|pulse ox|temperature|temp|fever|"
    r"heart rate|pulse|hr|respiratory rate|rr|weight|weighs|bmi|blood pressure|bp|pressure|systolic|diastolic|"
    r"score|grade|stage|level|class|type|act score|moca|mmse|phq-?9|phq|gad-?7|gad|gleason|apgar|breslow|"
    r"clark level|thickness|size|measur\w*|dose|dosage|strength|reading|readings|count|titer|level of|"
    r"urine|microalbumin|esr|crp|d-dimer|uric acid|testosterone|estradiol|fsh|lh|cortisol|hcg|vitals?)"
)

_SLASH = re.compile(r"(?<![\d/])(\d{1,3})\s?/\s?(\d{1,3})(?![\d/])")
_SCORE_CTX = re.compile(
    r"\b(?:murmurs?|strength|pain|scale|score|reflex(?:es)?|grade|vision|acuity|out of|motor|power|systolic|"
    r"diastolic|pressure|bp|over|mmhg|dose|inhaler|puffs?|mcg|mg|wixela|advair|symbicort|ratio|moca|mmse|"
    r"sensation|pulses|dtr|tender|swelling|edema)\b"
)
_DATE_BEFORE = re.compile(r"\b(?:on|dated|from|since|until|through|as of|by|before|after|date)\s*$")


def _add(out: list[Shield], doc: Doc, s: int, e: int, kind: str, reason: str) -> None:
    out.append(Shield(doc.field, s, e, kind, reason))


def clinical_numbers(doc: Doc) -> list[Shield]:
    """Numbers whose clinical meaning must not change (never perturbed, never treated as PII)."""
    t, out = doc.lower, []
    for m in _SLASH.finditer(t):
        a, b = int(m.group(1)), int(m.group(2))
        ctx = t[max(0, m.start() - 40) : m.end() + 30]
        is_bp = 60 <= a <= 260 and 30 <= b <= 160 and a > b
        is_score = b in (4, 5, 6, 10, 15, 20, 24, 28, 30) and a <= b
        if _DATE_BEFORE.search(t[max(0, m.start() - 10) : m.start()]) and not is_bp:
            continue  # `on 05/11` is a date, not a score
        if a < b <= 4:
            _add(out, doc, m.start(), m.end(), "clinical_number", "fraction (`2/3 maintenance`)")
            continue
        if is_bp or (is_score and _SCORE_CTX.search(ctx)) or _SCORE_CTX.search(t[m.end() : m.end() + 25]):
            _add(out, doc, m.start(), m.end(), "clinical_number", "score/grade/blood-pressure value")
    # Number followed by a unit (doses, measurements, durations) -- not ages ("59 years old").
    for m in re.finditer(
        rf"(?<![\d\-/.(])\b({NUM}|\d+(?:[.,]\d+)*)(?:\s?\.\s?\d+)?\s?-?\s?{UNITS}\b(?!\s?-?\s?old|\s(?:of age))", t
    ):
        if re.search(r"\bmarried for\s$", t[max(0, m.start() - 13) : m.start()]):
            continue  # marital duration is a quasi-identifier, not a clinical quantity
        if re.search(r"(?:years?|yrs?)\s*$", m.group(0)) and re.search(
            r"\b(?:age|aged|ages|turned|turning|i'?m|i am|he'?s|she'?s|is now)\s?:?\s*$",
            t[max(0, m.start() - 10) : m.start()],
        ):
            continue  # `Age: 77 yrs`, `aged 77 years` is an age, not a duration
        _add(out, doc, m.start(), m.end(), "clinical_number", "quantity with unit (dose/measurement/duration)")
    # Lab/vital name followed by its value within a few tokens.
    for m in re.finditer(
        rf"\b{LABS}\b(?:\s(?:is|was|of|at|came back|came back at|level|value|reading|about|around|like|now|today|of about|has been|were|are|=|:|was at|been))*\s?:?\s?((?:\d+(?:[.,]\d+)?)|{NUMWORDS}(?: (?:point|\.) {NUMWORDS})?)",
        t,
    ):
        _add(out, doc, m.start(1), m.end(1), "clinical_number", "lab/vital value")
    # Spoken blood pressure "one twenty over eighty".
    for m in re.finditer(rf"\b{NUM}\s(?:over)\s{NUM}\b", t):
        _add(out, doc, m.start(), m.end(), "clinical_number", "spoken blood pressure")
    # Spinal levels, stages, types, grades.
    for m in re.finditer(
        r"\b(?:[ctls]\d{1,2}(?:\s?-\s?[ctls]?\d{1,2})?|(?:type|stage|grade|class|level)\s(?:[0-9]{1,2}[a-c]?|[ivx]{1,4}|"
        + NUMWORDS
        + r"))\b",
        t,
    ):
        _add(out, doc, m.start(), m.end(), "clinical_number", "type/stage/grade/spinal level")
    # Tumour markers and lab panels with numbers in their name (CA-125, CA 19-9, HbA1c, B12, T4).
    for m in re.finditer(r"\b(?:ca|cea|afp|cd|her|hla|il|tnf|vitamin|b|t|g)[- ]?\d{1,3}(?:-\d)?\b", t):
        _add(out, doc, m.start(), m.end(), "clinical_number", "lab marker / panel name")
    # Decimal values (labs) and percentages.
    for m in re.finditer(r"(?<![\d.])\b\d+\.\d+\b(?!\.\d)|\b\d+\s?%", t):
        _add(out, doc, m.start(), m.end(), "clinical_number", "decimal/percentage value")
    return out


@cache
def _eponym_re() -> re.Pattern:
    alts = sorted({e.rstrip("'").replace("'s", "") for e in L.MEDICAL_EPONYMS}, key=len, reverse=True)
    return re.compile(
        r"\b("
        + "|".join(re.escape(a) for a in alts)
        + r")(?=(?:'|’)?"
        + L.EPONYM_NEXT.replace("(?:'s|s'|’s)?", "(?:'s|s'|’s|')?")
        + r")"
    )


def eponyms(doc: Doc) -> list[Shield]:
    out = []
    for m in _eponym_re().finditer(doc.lower):
        if _after_title(doc.lower, m.start()):
            continue
        _add(out, doc, m.start(), m.end(), "eponym", "eponym in medical syntax (followed by disease/sign/test wording)")
    for m in re.finditer(
        r"\b(?:hill-sachs|hawkins-kennedy|guillain[- ]barr[eé]|smith[- ]magenis|cheyne-stokes|osgood-schlatter|wolff-parkinson-white|sturge-weber|ehlers-danlos|epstein-barr|legg-calve-perthes|charcot-marie-tooth|austin flint|reed-sternberg|montreal cognitive assessment|montreal (?:classification|a\d)|pancoast tumou?r|horner(?:'s|’s)? syndrome|argyll robertson|kussmaul breathing|cheyne stokes)\b",
        doc.lower,
    ):
        _add(out, doc, m.start(), m.end(), "eponym", "compound eponym")
    return out


ANATOMIC_OF = re.compile(
    r"\b(?:foramen|foramina|duct|canal|space|triangle|ligament|circle|line|node|pouch|ampulla|fossa|tubercle|sphincter|crypts?|islets?|loop|plexus|bursa|fascia|ganglion) of\s+[a-z]+\b"
)


def anatomic_eponyms(doc: Doc) -> list[Shield]:
    """`foramen of monro`, `circle of willis`, `ampulla of vater`, `sphincter of oddi`."""
    return [
        Shield(doc.field, m.start(), m.end(), "eponym", "anatomical eponym") for m in ANATOMIC_OF.finditer(doc.lower)
    ]


def _after_title(t: str, i: int) -> bool:
    return bool(re.search(r"\b(?:mr|mrs|ms|miss|mister|missus|dr|doctor)\s?\.?\s*$", t[max(0, i - 12) : i]))


GENERIC_ICD_WORDS = frozenset(
    """unspecified other specified without with due and of the in or to for not elsewhere classified initial
    subsequent sequela encounter type site part region multiple unsp nos nec than more less by from at as on
    one two both all any an a is are this that history personal family status condition disorder disorders new
    disease diseases syndrome left right bilateral side upper lower region organ""".split()
)

# External-cause code words that carry the activity/mechanism/place (`Y93.53 Activity, golf`).
EXTERNAL_PREFIX = ("V", "W", "X", "Y")


def _base_code(code: str) -> str:
    """`W00.0XXA` -> `W00.0`; `S63.501A` -> `S63.501`; used to look up index paths."""
    c = code.upper().strip()
    head, _, tail = c.partition(".")
    tail = re.sub(r"X+[A-Z]?$", "", tail)
    if len(tail) == 4 and tail[-1].isalpha():
        tail = tail[:-1]
    return f"{head}.{tail}" if tail else head


# Lay variants of Y93 activity words (`Y93.02 Activity, running` is evidenced by `i like to run`, `marathon`).
ACTIVITY_SYNONYMS = {
    "running": "run runs runner runners jog jogging jogger marathon marathons 5k 10k treadmill sprint sprinting",
    "golf": "golfing golfer golf course backswing",
    "tennis": "racquet racket serve serving",
    "skiing": "ski skis skier slopes snowboarding",
    "soccer": "football futbol",
    "swimming": "swim swimmer laps pool",
    "cycling": "bike biking bicycle bicycling cyclist",
    "walking": "walk walks hike hiking",
    "weight": "weightlifting lifting weights barbell barbells deadlift squats",
    "gardening": "garden yard yard work landscaping mowing raking weeding",
    "basketball": "hoops",
    "volleyball": "beach volleyball",
    "dancing": "dance ballet",
    "bowling": "bowl bowler",
}


def code_evidence_terms(codes: list[dict]) -> dict[str, str]:
    """Content words (lowercase) that support the encounter's codes -> the code they support."""
    terms: dict[str, str] = {}
    index = L.icd_index_paths()
    for c in codes or []:
        if not isinstance(c, dict):
            continue
        code, desc = str(c.get("code") or ""), str(c.get("description") or "")
        texts = [desc]
        base = _base_code(code)
        desc_low = desc.lower()
        keys = {base}
        if code.upper().startswith(("V", "W", "X")):  # mechanism codes: category-level index words too
            keys.add(code.split(".")[0])
        for key in keys:
            texts += index.get(key, [])[:80]
        for txt in texts:
            # drop free-text index notes ("should not be confused with the New York Heart Association ...")
            txt = " > ".join(
                seg for seg in txt.split(" > ") if not seg.strip().lower().startswith(("note", "see ", "should not"))
            )
            for w in re.findall(r"[a-z][a-z'\-]{2,}", txt.lower()):
                w = w.strip("'-")
                if w in GENERIC_ICD_WORDS or len(w) < 3:
                    continue
                if (
                    (w in L.places() or w in L.first_names() or w in L.surnames())
                    and w not in desc_low
                    and not code.upper().startswith(EXTERNAL_PREFIX)
                ):
                    continue  # a place or person word mined from an index path must not shield a real name
                terms.setdefault(w, code)
        if code.upper().startswith("Y93"):
            for key, syn in ACTIVITY_SYNONYMS.items():
                if key in " ".join(texts).lower():
                    for w in syn.split():
                        terms.setdefault(w, code)
    return terms


def _variants(w: str) -> str:
    """Regex matching simple morphological variants (`golf` -> golf, golfing, golfer)."""
    stem = re.sub(r"(?:ing|ed|er|ers|es|s)$", "", w) if len(w) > 5 else w
    stem = stem[:-1] if stem.endswith("e") and len(stem) > 4 else stem
    return re.escape(stem) + r"(?:s|es|ed|ing|er|ers|ers'|'s|e|ies|y)?"


def code_evidence(doc: Doc, codes: list[dict]) -> list[Shield]:
    out = []
    terms = code_evidence_terms(codes)
    if not terms:
        return out
    by_code: dict[str, list[str]] = {}
    for w, code in terms.items():
        by_code.setdefault(code, []).append(w)
    for code, words in by_code.items():
        ext = code.upper().startswith(EXTERNAL_PREFIX)
        pat = re.compile(r"\b(?:" + "|".join(_variants(w) for w in sorted(words, key=len, reverse=True)) + r")\b")
        for m in pat.finditer(doc.lower):
            if _after_title(doc.lower, m.start()):
                continue
            _add(
                out, doc, m.start(), m.end(), "code_evidence", f"supports {code}" + (" (external cause)" if ext else "")
            )
    return out


def drugs(doc: Doc) -> list[Shield]:
    """Medication names (set lookup per token; a 17k-way regex alternation would be slow)."""
    out, toks = [], L.drug_tokens()
    for m in re.finditer(r"[a-z][a-z\-]{3,}", doc.lower):
        if m.group(0) in toks and not _after_title(doc.lower, m.start()):
            _add(out, doc, m.start(), m.end(), "drug", "medication name")
    return out


_INJURY = re.compile(
    r"\b(?:hurt|hurting|injur\w*|fell|fall(?:ing)?|twist\w*|sprain\w*|strain\w*|pain\w*|accident|hit|struck|"
    r"slipp\w*|tripp\w*|crash\w*|collision|pop(?:ped)?|landed|tore|torn|broke|broken|fractur\w*|swell\w*|"
    r"while|playing|during|caught|jumped|rolled|tweak\w*|pulled|dropped|bitten|bite|tick|ticks|exposure|"
    r"overuse|repetitive|sore|soreness|ache|aching)\b"
)
INJURY_CHAPTERS = ("S", "T", "V", "W", "X", "Y")


@cache
def _activity_re() -> re.Pattern:
    acts = sorted(L.ACTIVITY_WORDS, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(a) for a in acts) + r")\b")


# F0: verbs that tell an injury mechanism. Pain and soreness words, "while" and "playing" alone, and the
# noun "injury" are not enough ("my favorite is soccer and i've had a lot of injury").
_MECHANISM = re.compile(
    r"\b(?:hurt(?!s)|injured|injuring|sustained|fell|twist(?:ed|ing)|sprain(?:ed|ing)|pop(?:ped)?|landed|tore|"
    r"broke|fractured|slipped|tripped|crash(?:ed)?|collided|caught|jammed|hyperextend\w*|rolled|struck|hit|bite|bitten|stung|sting)\b"
)
_SENTENCE_END = re.compile(r"[.?!\n]")


def _sentence(t: str, s: int, e: int) -> str:
    a = max((m.end() for m in _SENTENCE_END.finditer(t, 0, s)), default=0)
    b = _SENTENCE_END.search(t, e)
    return t[a : b.start() if b else len(t)]


def activity_context(doc: Doc, injury_encounter: bool = False, sentence_level: bool = False) -> list[Shield]:
    """Keep activities told as the injury/exposure mechanism (`hurt my knee playing soccer`).

    In an encounter coded with an injury or external-cause code (chapters S-T, V-Y), every activity
    mention may be the mechanism, so all are kept. Otherwise an injury word within 150 characters is
    enough; with `sentence_level` (feature F0) a mechanism verb must be in the same sentence.
    """
    out, t = [], doc.lower
    for m in _activity_re().finditer(t):
        if sentence_level:
            told = bool(_MECHANISM.search(_sentence(t, m.start(), m.end())))
        else:
            told = bool(_INJURY.search(t[max(0, m.start() - 150) : m.end() + 150]))
        if injury_encounter or told:
            _add(
                out,
                doc,
                m.start(),
                m.end(),
                "activity_context",
                "activity in injury context (possible external-cause evidence)",
            )
    return out


# Codes whose evidence is an occupational exposure (pneumoconioses, hypersensitivity pneumonitis, other
# lung diseases due to external agents, occupational exposure Z57): the occupation is code evidence.
EXPOSURE_CODES = ("J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J68", "J69", "J70", "Z57", "T65", "L24")


def occupational_exposure(doc: Doc, codes: list[dict]) -> list[Shield]:
    if not any(isinstance(c, dict) and str(c.get("code") or "").upper().startswith(EXPOSURE_CODES) for c in codes):
        return []
    occ = "|".join(sorted((re.escape(o) for o in L.OCCUPATIONS | L.OCCUPATION_PLACES), key=len, reverse=True))
    out = []
    for m in re.finditer(rf"\b(?:{occ})(?:s|'s)?\b", doc.lower):
        _add(out, doc, m.start(), m.end(), "code_evidence", "occupational exposure supports an exposure code")
    return out


def evidence_consistency(docs: dict[str, Doc], shields: list[Shield]) -> list[Shield]:
    """F0: every mention of an activity kept as injury evidence is kept, in both fields.

    Generalizing one mention while the same word stays visible elsewhere in the record removes
    nothing an attacker could not read, and it loses code evidence.
    """
    words = {docs[s.field].lower[s.start : s.end] for s in shields if s.kind == "activity_context"}
    out: list[Shield] = []
    for w in sorted(words):
        rx = re.compile(r"\b" + re.escape(w) + r"\b")
        for f, d in docs.items():
            for m in rx.finditer(d.lower):
                if not any(s.field == f and s.start <= m.start() and m.end() <= s.end for s in shields):
                    _add(out, d, m.start(), m.end(), "activity_context", "same activity kept as injury evidence")
    return out


def build(
    doc: Doc, codes: list[dict] | None, injury_context: bool = True, sentence_level: bool = False
) -> list[Shield]:
    """All shields for one field. `codes=None` means the without-codes mode.

    Activities told as an injury mechanism are kept in both modes (the plan's policy: keep hobbies
    when they are the mechanism or external-cause evidence, generalize them otherwise).
    """
    out = clinical_numbers(doc) + eponyms(doc) + anatomic_eponyms(doc) + drugs(doc)
    if codes:
        out += code_evidence(doc, codes) + occupational_exposure(doc, codes)
    if injury_context or not codes:
        injury_encounter = bool(codes) and any(
            isinstance(c, dict) and str(c.get("code") or "")[:1].upper() in INJURY_CHAPTERS for c in codes
        )
        out += activity_context(doc, injury_encounter, sentence_level)
    return out


def overlapping(shields: list[Shield], field: str, s: int, e: int) -> Shield | None:
    for sh in shields:
        if sh.field == field and sh.start < e and s < sh.end:
            return sh
    return None
