"""PII category regexes shared by phases 3, 10 and 11 (extracted from 03_pii_inventory.py)."""
from namelib import *

recs = load_all()
N = len(recs)
STOP = corpus_stopwords(recs) | GREET_STOP

MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"
MONTHS_FULL = r"(?:january|february|march|april|june|july|august|september|october|november|december)"
NUMWORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|a hundred)"
DR_STOP = sorted({"put", "said", "told", "is", "was", "will", "who", "and", "has", "wants", "thinks", "that", "to", "of", "for", "about", "can",
                  "did", "if", "here", "says", "does", "would", "should", "could", "or", "in", "on", "at", "with", "may", "might", "also",
                  "just", "then", "there", "they", "you", "i", "we", "he", "she", "it", "a", "an", "the", "my", "your", "his", "her",
                  "their", "our", "visit", "appointment", "recommended", "mentioned", "wanted", "thought", "called", "ordered",
                  "prescribed", "referred", "gave", "sent", "wrote", "looked", "checked", "saw", "sees", "see", "seen", "asked", "know",
                  "think", "feel", "believe", "yet", "already", "still", "today", "last", "next", "first", "right", "okay", "yeah", "um",
                  "uh", "so", "when", "what", "how", "where", "why", "because", "but", "since", "before", "after", "once", "while", "not",
                  "no", "yes", "please", "sorry", "let", "look", "looks", "do", "done", "doing", "going", "comes", "come", "came", "get",
                  "gets", "got", "give", "had", "have", "be", "been", "am", "are", "were", "us", "them", "me", "him", "this", "any", "some",
                  "every", "all", "both", "more", "most", "much", "many", "few", "other", "another", "very", "too", "really", "pretty",
                  "well", "good", "new", "old", "little", "day", "time", "week", "month", "year", "again", "back", "up", "down", "out",
                  "over", "like", "from", "by", "nurse", "office", "white coat", "arthrogram", "as", "than", "visits", "wants", "want",
                  "needs", "need", "say", "says", "tells", "tell", "ask", "asks", "recommend", "recommends", "order", "orders", "write",
                  "writes", "call", "calls", "which", "whose", "whom", "either", "neither", "yourself", "himself", "herself", "myself",
                  "actually", "basically", "probably", "maybe", "definitely", "certainly", "obviously", "apparently", "hopefully"})

def gaz_re(words):
    words = sorted(words, key=len, reverse=True)
    return r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"

def dr_re(prefix):
    return prefix + r"\s?\.?\s(?!(?:" + "|".join(re.escape(w) for w in DR_STOP) + r")\b)(?P<n>[a-z]+)\b"

CATEGORIES = {
    "patient_first_name_greeting": [DLG_GREET_RE.pattern],
    "patient_name_intro": [DLG_INTRO_RE.pattern, DLG_HEADER_RE.pattern, DLG_HEADER2_RE.pattern, NOTE_FULLNAME_RE.pattern],
    "title_plus_name": [DLG_TITLE_RE.pattern.replace("(?P<name>", "(?P<n>"), NOTE_TITLE_RE.pattern.replace("(?P<last>", "(?P<n>")],
    "doctor_staff_name": [dr_re(r"\b(?:dr|doctor)"), r"\bDr\.? (?P<n>[A-Z][a-z]+)",
                          r"\b(?:nurse|medical assistant|my assistant|the assistant|front desk|receptionist|scheduler|therapist|pharmacist|dietitian|my pa|my np)\s?,?\s(?:named )?(?P<n>[a-z]+)\b(?= will| is| can| here| come| has| to| and|'ll| who| ,| \.)",
                          r"\b(?:with|and|have|ask|see|tell|told|talk to|talked to|spoke with|speak with|call|called|update(?:d)? it with|went through with|go over with|sat down with) (?P<n>" + "|".join(sorted(FIRST_NAMES)) + r") (?:come in|will|'ll|is going to|can|at the front|up front|in the front|in the back|the nurse|our nurse|my nurse|the ma|our ma|my ma|the assistant|our assistant|my assistant|the receptionist|our receptionist)\b"],
    "relative_or_friend": [gaz_re(RELATIVES)],
    "named_relative_or_friend": [r"\b(?:my|his|her|their|your|our) (?:" + "|".join(sorted(RELATIVES)) + r")\s?,?\s(?:named )?(?P<n>[a-z]+)\b(?= (?:is|was|has|had|who|and|said|told|got|does|did|came|lives|works|,|\.))",
                                 r"\b(?:my|his|her|their|your|our) (?:" + "|".join(sorted(RELATIVES)) + r")\s?,?\s(?P<n>" + "|".join(sorted(FIRST_NAMES)) + r")\b"],
    "dob": [r"\b(?:date of birth|dob|d\.o\.b\.|birth ?date|born on|born in|birthday is)\b"],
    "birthday_mention": [r"\bbirthday\b"],
    "exact_age": [r"\b(?:\d{1,3}|" + NUMWORD + r"(?:[ -]" + NUMWORD + r")?)\s?-?\s?(?:year|month|week|day)s?\s?-?\s?old\b",
                  r"\b(?:age|aged|at age|at the age of|turning|turned|i'm|i am|he's|she's|you are|you're) (?:\d{1,3}|" + NUMWORD + r"(?:[ -]" + NUMWORD + r")?)\b(?! (?:mg|milligrams?|mcg|percent|%|pounds?|lbs|units?|minutes?|hours?|days?|weeks?|months?|years?|times?|puffs?|tablets?|pills?|cm|mm|inches?|feet|foot|degrees?|refills?|out of|over|to|and|or|more|less|point))",
                  r"\bin (?:his|her|their|my) (?:early |late |mid[- ])?(?:\d0s|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
                  r"\b(?:\d{1,3}|" + NUMWORD + r")[- ]?(?:yo|y/o|y\.o\.)\b"],
    "absolute_date": [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b",
                      r"\b" + MONTHS + r"\.? \d{1,2}(?:st|nd|rd|th)?(?: ?,? ?(?:19|20)\d{2})?\b(?! ?(?:mg|mcg|ml|percent|%|days|weeks|months|years|times|puffs|tablets))",
                      r"\b\d{1,2}(?:st|nd|rd|th)? (?:of )?" + MONTHS + r"\b(?: ?,? ?(?:19|20)\d{2})?",
                      r"\b" + MONTHS + r"\.? ?,? ?(?:of )?(?:19|20)\d{2}\b", r"\b\d{1,2}/(?:19|20)\d{2}\b",
                      r"\b(?:in|since|of|back in|until|from|through|by|around|about) (?:19|20)\d{2}\b",
                      r"\b(?:19|20)\d{2}\b(?! ?(?:mg|mcg|ml|units?|milligrams?|calories|steps|pounds?|lbs|dollars|\$))",
                      r"\b(?:nineteen|two thousand) (?:" + NUMWORD + r"|and)(?: " + NUMWORD + r")?\b",
                      r"\b(?:on|since|until|from) the \d{1,2}(?:st|nd|rd|th)\b"],
    "month_or_season_or_holiday": [r"\b" + MONTHS_FULL + r"\b", r"\b(?:in|last|this|next|early|late|of|since|around|during) may\b",
                                   r"\b(?:spring|summer|winter|autumn)\b|\b(?:in the|this|last|next|every|the) fall\b|\bfall of\b",
                                   gaz_re({"christmas", "thanksgiving", "easter", "halloween", "new year", "new year's", "new years",
                                           "fourth of july", "4th of july", "memorial day", "labor day", "mother's day", "father's day",
                                           "valentine's", "valentines", "super bowl", "spring break", "the holidays", "holiday season"})],
    "weekday": [r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b"],
    "relative_date": [r"\b(?:\w+[- ])?(?:\w+ )?(?:days?|weeks?|months?|years?|decades?|hours?|minutes?|nights?) ago\b",
                      r"\b(?:last|this past|past|next|this coming|the other|previous|prior|following|coming|upcoming) (?:week|month|year|spring|summer|fall|winter|night|morning|evening|weekend|visit|appointment|monday|tuesday|wednesday|thursday|friday|saturday|sunday|few (?:days|weeks|months|years)|couple (?:of )?(?:days|weeks|months|years)|\d+ (?:days|weeks|months|years))\b",
                      r"\b(?:yesterday|tomorrow|tonight|last night|this morning|this afternoon|this evening|earlier today|later today|the weekend|over the weekend|a while ago|a while back|recently|lately|a few (?:days|weeks|months|years) (?:ago|back))\b",
                      r"\b(?:in|within|for|over|after|about|around|almost|nearly|roughly) (?:the (?:last|past|next) )?(?:\d+|a|an|" + NUMWORD + r"|a couple(?: of)?|a few|several|many|\d+ to \d+|\d+-\d+|\d+ or \d+) (?:more )?(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
                      r"\b(?:since|during|before|after|with|through|post|pre|around|when|at the start of|at the beginning of|at the end of|in the middle of) (?:the )?(?:covid|pandemic|lockdown|quarantine|the shutdown|the holidays|college|high school|the surgery|the accident|the fall|my (?:surgery|accident|fall|injury|divorce|wedding|retirement|deployment|pregnancy)|her (?:surgery|pregnancy)|his (?:surgery|accident))\b",
                      r"\b(?:when i was|since i was|when she was|when he was|since he was|since she was|as a|since) (?:a )?(?:kid|child|teenager|teen|young|little|baby|in (?:high school|college|school|the military|the army|the navy)|\d+|" + NUMWORD + r"(?: " + NUMWORD + r")?)\b"],
    "clock_time": [r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[ap]\.?m\.?)?\b", r"\b\d{1,2}\s?(?:[ap]\.?m\.?)\b", r"\b\d{1,2} o'?clock\b"],
    "time_of_day_phrase": [r"\b(?:noon|midnight|at night|in the morning|in the afternoon|in the evening|bedtime|at bedtime|supper time|dinner ?time|lunch ?time|breakfast)\b"],
    "location": [gaz_re(PLACES)] + PLACE_PHRASES,
    "employer_occupation_school": [gaz_re(OCCUPATIONS)] + OCC_PHRASES,
    "hobby": [gaz_re(HOBBIES)] + HOBBY_PHRASES,
    "event": [gaz_re(EVENTS)],
    "sports_team": [gaz_re(SPORTS_TEAMS)],
    "religion_ethnicity_nationality": [gaz_re(RELIGION_ETHNICITY)],
    "phone_email_mrn_id_insurance": [r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", r"\(\d{3}\) ?\d{3}-\d{4}", r"\S+@\S+\.\S+", r"\b(?:www\.|http)\S+",
                                      r"\bmrn\w*\b", r"\b(?:medical record|record number|chart number|account number|member id|policy number|social security|ssn|driver'?s license|license number|passport)\b",
                                      r"\b\d{5,}\b", r"\b\d(?:-\d){4,}\b", r"\b(?:\d{3}-\d{2}-\d{4})\b",
                                      gaz_re(INSURERS), r"\b(?:insurance|insurer|co-?pay|deductible|premium|out of pocket|prior auth\w*|preapprov\w*|pharmacy benefit)\b"],
    "vehicle": [gaz_re(VEHICLES)],
    "pet": [gaz_re(PETS), r"\b(?:my|our|his|her) (?:dog|cat|puppy|kitten|horse|pet|bird)\s?,?\s(?:named )?(?P<n>[a-z]+)\b"],
    "product_brand": [gaz_re(BRANDS)],
    "other_quasi_identifier": [r"\b(?:twins?|triplets?|adopt\w*|foster (?:kid|child|care|parent)|widow\w*|divorc\w*|homeless|shelter|prison|jail|incarcerat\w*|parole|probation|arrest\w*|dui|lawsuit|custody|food stamps|section 8|welfare|tattoo\w*|piercing\w*|vegan|vegetarian|kosher|halal|sober|aa meetings|methadone|suboxone|celebrit\w+|famous|lottery|million\w*|wealthy|bankrupt\w*)\b",
                               r"\b(?:lives? alone|live with|lives with|living with|stays with|moved in with|roommates?|apartment|condo|trailer|mobile home|farm|ranch|basement|upstairs|downstairs|stairs at home|two[- ]story|cul-?de-?sac|culdesac|driveway)\b",
                               r"\b(?:speak|speaks|speaking) (?:spanish|english|french|chinese|mandarin|cantonese|tagalog|vietnamese|arabic|russian|portuguese|korean|german|italian|polish|hindi|urdu)\b|\binterpreter\b|\btranslat\w+\b"],
}
NAME_CATS = {"patient_first_name_greeting", "patient_name_intro", "title_plus_name", "doctor_staff_name", "named_relative_or_friend"}


def accept(cat, m):
    gd = m.groupdict()
    if cat in NAME_CATS:
        if cat == "patient_name_intro":
            pass
        else:
            for g in ("name", "first", "last", "n"):
                v = gd.get(g)
                if v and v.lower() in STOP:
                    return False
        if cat == "patient_name_intro":
            # "the patient is a 58-year-old": no name
            if (gd.get("first") or "").lower() in {"patient", "the", "she", "he", "this", "who"}:
                return False
            # "so tyler is a 56 -year-old": greedy regex took first="so"; accept on the last token
            f, l = (gd.get("first") or "").lower(), (gd.get("last") or "").lower()
            if f in STOP and l and l not in STOP:
                return True
            if f in STOP:
                return False
    if cat == "location":
        v = (gd.get("n") or "").lower()
        if v and (v in PLACE_HEAD_STOP or v.split()[0] in PLACE_HEAD_STOP or v in STOP):
            return False
    if cat in ("hobby", "employer_occupation_school"):
        v = (gd.get("n") or "").lower()
        if v and (v in STOP or v.split()[0] in STOP - {"work"}):
            return False
    return True


def merge(hits):
    """Drop hits fully inside another hit of the same category (longest wins)."""
    hits.sort(key=lambda h: (h[0], h[3], h[4], -h[5]))
    out, last = [], None
    for h in hits:
        if last and h[0] == last[0] and h[3] == last[3] and h[4] < last[5]:
            continue
        out.append(h); last = h
    return out


