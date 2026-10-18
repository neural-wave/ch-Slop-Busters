"""Quasi-identifier detectors: places, facilities, organisations, occupations, events, activities.

These carry the re-identification risk the EDA measured (706 corpus-unique details), but several
are also code evidence (activities for Y93.x, places for Y92.x). Detection is broad here; the
medical shield (code evidence, injury context) and the preset policy decide what is removed.
"""

from __future__ import annotations

import re
from functools import cache

from .. import lexicons as L
from ..spans import Span
from ..text.normalize import Doc, is_sentence_start

PLACE_CUE = re.compile(
    r"\b(?:in|from|to|near|at|visit|visiting|visited|moved|move|moving|live|lives|living|lived|trip|vacation|"
    r"drove|flew|fly|went|go|going|back|out|down|up|over|outside|around|of|born|raised|grew up|based|"
    r"stationed|relocated|traveled|travelled|travel|cruise|hometown|state|city|county)\s(?:the\s)?$"
)
GENERIC_HEAD = frozenset(
    """the a an this that our your my his her their local nearest closest emergency er urgent same another other
    new old big small different cardiac heart cancer eye dental sleep pain wound women's walk-in any outpatient
    inpatient rehab surgical surgery orthopedic primary care medical health a family children's general main
    to at in from of for with and or private public veterans va state university teaching community nursing
    some which what whatever every each mental behavioral physical therapy specialty specialist gi kidney
    lung liver skin bone joint spine sports imaging radiology lab x-ray mri ct cat ultrasound infusion dialysis
    cath echo stress allergy diabetes weight vein vascular foot hand fertility pediatric womens mens senior
    assisted skilled living memory hospice home day overnight later first second next last only own whole
    university's here there their""".split()
)
WEAK_FACILITY_TAIL = r"(?:regional|medical|surgical services|medical services|health services|healthcare|physicians|associates|orthopedics|orthopaedics)"
FACILITY_TAIL = r"(?:regional hospital|regional medical center|medical group|hospital|medical center|medical centre|clinic|health center|health centre|health system|urgent care|orthopedics|orthopaedics|medical group|medical college|institute|infirmary|hospice|pharmacy|surgical center|surgery center|cancer center|heart center|rehab center|rehabilitation center|nursing home|healthcare|physicians|associates)"


@cache
def _place_index() -> dict[str, tuple[str, int, str]]:
    return L.places()


def _phrase_scan(t: str, lexicon: frozenset[str] | dict, max_words: int = 4):
    """Yield (start, end, phrase) for lexicon phrases of up to `max_words` words (longest first)."""
    words = [(m.start(), m.end()) for m in re.finditer(r"[a-z0-9][a-z0-9&'\-]*", t)]
    i = 0
    while i < len(words):
        hit = None
        for n in range(min(max_words, len(words) - i), 0, -1):
            s, e = words[i][0], words[i + n - 1][1]
            phrase = re.sub(r"\s+", " ", t[s:e])
            if phrase in lexicon:
                hit = (s, e, phrase, n)
                break
        if hit:
            yield hit[:3]
            i += hit[3]
        else:
            i += 1


RESIDENCE_CUE = re.compile(
    r"\b(?:live|lives|living|lived|moved|move|moving|trip|vacation|visit|visiting|visited|flew|drove|from|born in|raised in|grew up in|relocated)\b[a-z ]{0,12}$"
)


def places(doc: Doc) -> list[Span]:
    t, v, out = doc.lower, doc.view, []
    idx = _place_index()
    for s, e, ph in _phrase_scan(t, idx, 4):
        typ, pop, cc = idx[ph]
        nwords = len(ph.split())
        common = any(L.is_common_word(w) for w in ph.split()) and nwords == 1
        cue = bool(PLACE_CUE.search(t[max(0, s - 25) : s]))
        cased = (doc.field == "note" or doc.cased) and v[s].isupper() and (not is_sentence_start(v, s) or nwords > 1)
        after_place = bool(re.search(r"([a-z][a-z .'\-]+?)\s?,\s?$", t[max(0, s - 30) : s])) and any(
            t[max(0, s - 30) : s].rstrip(" ,").endswith(p)
            for p in (t[max(0, s - 30) : s].rstrip(" ,").split()[-1:] or [""])
            if p in idx
        )
        followed_state = bool(re.match(r"\s?,\s?[a-z]+", t[e : e + 20])) and typ == "city"
        name_like = ph in L.first_names() or (ph in L.surnames() and L.surnames()[ph] > 5000)
        if nwords >= 2 and not all(L.is_common_word(w) for w in ph.split()):
            ok = cue or cased or typ != "city" or pop >= 100000
        elif typ in ("state", "country"):
            ok = (cue or cased or after_place) and not common and not (name_like and not (cue or after_place))
        else:  # single-word city: needs a cue (or an address context in cased text)
            ok = (
                (cue or followed_state or (cased and pop >= 100000))
                and not common
                and not (name_like and not (cue or followed_state))
                and pop >= 30000
            )
        if ok and nwords == 1 and (ph in L.medical_vocab() or ph in L.MEDICAL_EPONYMS or ph in L.drug_tokens()):
            ok = followed_state or bool(
                RESIDENCE_CUE.search(t[max(0, s - 30) : s])
            )  # `fracture of the tarsus`, `history of kawasaki`
        if ok:
            out.append(
                Span(
                    doc.field, s, e, "LOCATION", f"gazetteer:place_{typ}", 0.85 if cue or cased else 0.7, f"{typ} name"
                )
            )
    for m in re.finditer(
        r"\b(?:east|west|north|south) coast\b|\bmidwest\b|\bnew england\b|\bthe bay area\b|\bpacific northwest\b|\bdown south\b|\bout west\b|\bback east\b|\bup north\b",
        t,
    ):
        out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "rule:region", 0.8, "region"))
    # Named streets: `main street`, `fifth street`, `kaiser permanente broadway`.
    for m in re.finditer(
        r"\b(?P<head>[a-z]+)\s(?:street|avenue|road|boulevard|lane|drive|highway|parkway|st\.|ave\.)\b", t
    ):
        h = m.group("head")
        if (h == "main" or h not in GENERIC_HEAD) and h not in (
            "rail",
            "dirt",
            "gravel",
            "private",
            "frontage",
            "service",
            "access",
            "across",
            "down",
            "up",
            "cross",
            "end",
            "side",
            "busy",
            "dirt",
            "back",
            "rough",
            "long",
            "open",
            "wrong",
            "right",
            "left",
            "next",
            "on",
            "off",
            "high",
            "toll",
            "country",
            "two",
            "one",
            "side",
            "that",
            "this",
        ):
            out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "rule:street", 0.8, "named street"))
    for m in re.finditer(
        r"\b(?:[A-Z][a-z]+ ){1,2}(?:Hills|Heights|Nagar|Colony|Layout|Estate|Gardens|Village|Beach|Island|Springs|Valley|Reclamation|Junction|Cross|Circle|Park)\b",
        v,
    ):
        if not is_sentence_start(v, m.start()) or "," in v[m.end() : m.end() + 3]:
            out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "rule:locality_cased", 0.8, "named locality"))
    for m in re.finditer(r"\b(?P<h>[a-z]+) (?:beach|island|lake|springs|valley|heights|hills)\b", t):
        if (
            m.group("h") not in GENERIC_HEAD
            and not L.is_common_word(m.group("h"), 30)
            and m.group("h") not in L.medical_vocab()
        ):
            out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "rule:locality", 0.75, "named locality"))
    for m in re.finditer(r"\bbroadway\b", t):
        out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "rule:street", 0.75, "named street"))
    return out


FAC_NAME_WORDS = frozenset(
    "district community county memorial mercy methodist baptist presbyterian regional saint st sacred providence valley metro east west north south central lakeside riverside university general".split()
)


FAC_HEAD_STOP = frozenset(
    """she he they we i you it denies reports report call so and or but no not well well-healed healed
    any some this that her his their our your my the a an to at in of for with from by""".split()
)


def _fac_head_ok(w: str, cased: bool) -> bool:
    if w in GENERIC_HEAD and w not in FAC_NAME_WORDS or not re.fullmatch(r"[a-z]+", w) or w in FAC_HEAD_STOP:
        return False
    if w in L.medical_vocab() or w in L.drug_tokens():
        return False
    return (
        cased
        or w in FAC_NAME_WORDS
        or w in L.surnames()
        or w in L.first_names()
        or w in L.places()
        or not L.is_common_word(w, 30)
    )


def facilities(doc: Doc) -> list[Span]:
    out = []
    for weak, rx in ((False, FACILITY_TAIL), (True, WEAK_FACILITY_TAIL)):
        out += _facility_matches(doc, rx, weak)
    if doc.field == "note":  # Capitalised institution names: "Christian Medical College (CMC)", "Apollo Health City"
        out += _cased_institutions(doc)
    return out


def _facility_matches(doc: Doc, tail: str, weak: bool) -> list[Span]:
    t, v, out = doc.lower, doc.view, []
    for m in re.finditer(rf"\b(?P<head>(?:[a-z][a-z'\-]*[ \t]){{1,3}}){tail}\b", t):
        heads = m.group("head").split()
        keep = []
        for w in reversed(heads):
            pos = m.start("head") + m.group("head").rfind(w)
            cased = (
                (doc.field == "note" or doc.cased)
                and v[pos].isupper()
                and not v[pos : pos + len(w)].isupper()
                and not is_sentence_start(v, pos)
            )
            ok = (
                _fac_head_ok(w, cased)
                if not weak
                else (
                    w not in FAC_HEAD_STOP
                    and w not in L.medical_vocab()
                    and w not in L.drug_tokens()
                    and re.fullmatch(r"[a-z]+", w) is not None
                    and (
                        (w in FAC_NAME_WORDS and w not in ("general", "university", "central", "community"))
                        or ((w in L.surnames() or w in L.first_names() or w in L.places()) and L.lower_uses(w) < 30)
                        or (cased and not L.is_common_word(w, 30))
                    )
                )
            )
            if not ok:
                break
            keep.append(w)
        if (
            not keep
            or all(w in FAC_NAME_WORDS for w in keep)
            and len(keep) == 1
            and keep[0] in ("general", "university", "central")
        ):
            continue
        start = m.end("head") - len(" ".join(reversed(keep))) - 1
        out.append(
            Span(
                doc.field,
                start,
                m.end(),
                "FACILITY",
                "rule:facility" + ("_weak" if weak else ""),
                0.8 if weak else 0.85,
                "named facility",
            )
        )
    return out


def _cased_institutions(doc: Doc) -> list[Span]:
    """Capitalised institution names: `Christian Medical College (CMC)`, `Apollo Health City`."""
    v, out = doc.view, []
    for m in re.finditer(
        r"\b(?:[A-Z][a-zA-Z'\.&]+[ \t]){1,4}(?:Hospital|Hospitals|Medical Center|Medical College|Clinic|Health City|Health System|Institute|Infirmary|Healthcare|Medical Institute|Multispecialty Hospital|Medicity|Heart Institute|Cancer Centre|Cancer Center)\b(?:\s\([A-Z]{2,6}\))?",
        v,
    ):
        words = m.group(0).split()
        if words[0].lower() in GENERIC_HEAD and len(words) <= 2:
            continue
        name_words = [w.lower().strip(".'") for w in words[:-1]]
        if any(w in L.medical_vocab() or w in L.drug_tokens() for w in name_words) and not any(
            (w in L.places() or w in L.surnames()) and w not in L.medical_vocab() for w in name_words
        ):
            continue  # `Heart Failure Clinic`, `Methadone Clinic`
        if is_sentence_start(v, m.start()) and all(L.is_common_word(w, 30) for w in name_words):
            continue
        out.append(Span(doc.field, m.start(), m.end(), "FACILITY", "rule:facility_cased", 0.9, "named institution"))
    return out


def organizations(doc: Doc) -> list[Span]:
    t, v, out = doc.lower, doc.view, []
    work_cue = re.compile(
        r"\b(?:work|works|worked|working|job|employ\w*|at|from|for|through|pharmacy|store|shop|insurance|company|through the|bought|buy|get|got|fill|filled|pick up|picked up)\s(?:the\s|my\s|a\s)?$"
    )
    clinical_after = re.compile(
        r"\s?(?::|bp\b|blood pressure|organ|ldl|a1c|hba1c|range|dose|junction|reflux|vaccine|booster|shot|juice|level|goal|heart rate|weight|inr|lesion)"
    )
    for s, e, ph in _phrase_scan(t, L.EMPLOYERS, 4):
        single = " " not in ph
        if single and (
            clinical_after.match(t[e : e + 15])
            or not (
                work_cue.search(t[max(0, s - 20) : s]) or re.match(r"\s(?:pharmacy|store|shop|company)", t[e : e + 12])
            )
        ):
            continue  # `Target BP`, `CVS:` exam header, `GE junction`, `Pfizer vaccine`
        out.append(Span(doc.field, s, e, "ORGANIZATION", "gazetteer:employer", 0.85, "named company / employer"))
    for s, e, ph in _phrase_scan(t, L.INSURERS, 6):
        single = " " not in ph
        if single and ph in ("ppo", "hmo"):
            continue
        out.append(Span(doc.field, s, e, "INSURER", "gazetteer:insurer", 0.9, "insurer"))
    team_cue = re.compile(
        r"\b(?:fan|fans|game|games|play|plays|playing|played|watch|watching|watched|season|team|root|rooting|cheer|tickets|stadium|win|won|lost|beat|playoffs)\b"
    )
    tight = re.compile(r"^\s?(?:'s)?\s?(?:fans?|games?|season|playoffs?|tickets|won|lost|beat)\b")
    tight_before = re.compile(
        r"(?:fan of the|fans? of|watch(?:ing|ed)? the|root(?:ing)? for the|cheer(?:ing)? for the|go|the)\s$"
    )
    for s, e, ph in _phrase_scan(t, L.SPORTS_TEAMS, 3):
        common = L.is_common_word(ph.split()[-1], 100) or ph in ("athletics", "the dogs")
        if common:
            ok = tight.match(t[e : e + 15]) or (
                tight_before.search(t[max(0, s - 22) : s]) and team_cue.search(t[max(0, s - 60) : e + 60])
            )
        else:
            ok = " " in ph or team_cue.search(t[max(0, s - 60) : e + 60])
        if ok:
            out.append(Span(doc.field, s, e, "SPORTS_TEAM", "gazetteer:team", 0.8, "sports team (regional hint)"))
    for m in re.finditer(r"\b(?P<x>[a-z][a-z'\-]+(?:\s[a-z][a-z'\-]+)?)\sfans?\b", t):  # `yankie's fan` (ASR spellings)
        words = m.group("x").split()
        if words[0] in (
            "a",
            "the",
            "big",
            "huge",
            "great",
            "any",
            "no",
            "not",
            "ceiling",
            "box",
            "electric",
            "exhaust",
        ):
            words = words[1:]
        if words and all(
            not L.is_common_word(w.removesuffix("'s"), 30) and w.removesuffix("'s") not in L.medical_vocab()
            for w in words
        ):
            s0 = m.end("x") - len(" ".join(words))
            out.append(
                Span(doc.field, s0, m.end("x"), "SPORTS_TEAM", "rule:fan_of", 0.7, "team / fandom (regional hint)")
            )
    # Employer frames: `works at <X>`, `retired from <X>`.
    for m in re.finditer(
        r"\b(?:work(?:s|ed|ing)?|employed|job|retired)\s(?:at|for|from)\s(?:a\s|an\s|the\s)?(?P<x>[a-z][a-z&'\-]+(?:\s[a-z&'\-]+){0,2})",
        t,
    ):
        words = m.group("x").split()
        for k, w in enumerate(words):
            if w in ("in", "on", "at", "for", "with", "and", "as", "since", "near", "downtown"):
                words = words[:k]
                break
        while words and (
            words[-1] in GENERIC_HEAD
            or words[-1] in WORK_TRAIL
            or L.is_common_word(words[-1], 300)
            and words[-1] not in L.OCCUPATION_PLACES
        ):
            words.pop()
        if not words or words[0] in WORK_GENERIC or words[0] in GENERIC_HEAD:
            continue
        nxt = re.match(r"\s([a-z]+)", t[m.start("x") + len(" ".join(words)) : m.start("x") + len(" ".join(words)) + 20])
        if any(w in L.medical_vocab() for w in words) or (nxt and nxt.group(1) in L.medical_vocab()):
            continue  # `it only works for the barky cough`
        phrase = " ".join(words)
        s0 = m.start("x")
        e0 = s0 + len(phrase)
        if phrase in L.OCCUPATION_PLACES or words[-1] in L.OCCUPATION_PLACES:
            out.append(
                Span(doc.field, s0, e0, "OCCUPATION", "rule:work_frame", 0.75, "workplace in a 'works at' frame")
            )
        elif (
            phrase in L.EMPLOYERS
            or (doc.field == "note" and v[s0].isupper())
            or all(not L.is_common_word(w, 30) and w not in L.medical_vocab() for w in words)
        ):
            out.append(
                Span(doc.field, s0, e0, "ORGANIZATION", "rule:work_frame", 0.75, "employer in a 'works at' frame")
            )
    for m in re.finditer(
        r"\b(?:i|we|he|she|they|family|parents|husband|wife|patient|who)\s(?:also\s|still\s)?(?:own|owns|owned|run|runs|ran|manage|manages)\s(?:a|an|the|our|their|his|her)\s(?P<x>(?:[a-z\-]+\s){0,2}(?:shop|store|business|company|restaurant|bakery|farm|ranch|salon|firm|practice|garage|bar|cafe|diner|franchise|dealership))\b",
        t,
    ):
        out.append(
            Span(
                doc.field,
                m.start("x"),
                m.end("x"),
                "OCCUPATION",
                "rule:owns_business",
                0.85,
                "owned business (occupation)",
            )
        )
    for m in re.finditer(
        r"\b(?:insurance|coverage|benefits|job|work|working|employed)\s(?:\w+\s){0,2}?through\s(?P<x>(?:the|my|his|her|our)\s(?:plant|factory|mill|company|employer|union|job|work|county|state|school district|district|railroad|refinery|mine))\b",
        t,
    ):
        out.append(
            Span(
                doc.field,
                m.start("x"),
                m.end("x"),
                "OCCUPATION",
                "rule:through_employer",
                0.8,
                "employer behind insurance/benefits",
            )
        )
    if doc.field == "note":
        for m in re.finditer(
            r"\b(?:works|worked|employed|working)\s(?:at|for|with)\s(?P<x>(?:[A-Z][\w&'\-]+\s?){1,4})", v
        ):
            out.append(
                Span(
                    doc.field, m.start("x"), m.end("x"), "ORGANIZATION", "rule:work_frame_cased", 0.85, "named employer"
                )
            )
    return out


WORK_GENERIC = frozenset(
    """home night nights work least all it this that them him her me you us my your his desk computer
    computers office job jobs the a an same different another other least first last full part both""".split()
)
WORK_TRAIL = frozenset(
    """and but so now for as since about because until when where all day home night nights years year
    time part full i you we he she they it in on at a the to of with""".split()
)

_OCC_RX = None


def occupations(doc: Doc) -> list[Span]:
    global _OCC_RX
    if _OCC_RX is None:
        occ = "|".join(sorted((re.escape(o) for o in L.OCCUPATIONS), key=len, reverse=True))
        _OCC_RX = re.compile(
            rf"\b(?:i'?m|i am|he'?s|she'?s|he is|she is|was|works? as|worked as|working as|job as|career as|"
            rf"retired|former|as an?|is an?|was an?|am an?|being an?|i'm an?|she's an?|he's an?|patient is an?|"
            rf"occupation\s?:|profession\s?:|employment\s?:)\s(?:an?\s)?(?:retired\s|former\s|part-time\s|full-time\s|"
            rf"high school\s|middle school\s|elementary school\s|school\s|college\s|registered\s|licensed\s|senior\s|head\s|"
            rf"(?!(?:big|huge|great|good|bad|little|very|real|pretty)\s)[a-z]+\s)?"
            rf"(?P<occ>{occ})s?\b"
        )
    t, out = doc.lower, []
    for m in _OCC_RX.finditer(t):
        out.append(
            Span(doc.field, m.start("occ"), m.end("occ"), "OCCUPATION", "rule:occupation_frame", 0.85, "occupation")
        )
    occ = "|".join(sorted((re.escape(o) for o in L.OCCUPATIONS), key=len, reverse=True))
    for m in re.finditer(
        rf"\b(?:an?|the)\s(?:\d{{1,3}}|[a-z]+(?:[ -][a-z]+)?)[ -]year[ -]old\s(?:[a-z]+\s)?(?P<occ>{occ})\b", t
    ):
        out.append(
            Span(doc.field, m.start("occ"), m.end("occ"), "OCCUPATION", "rule:occupation_after_age", 0.8, "occupation")
        )
    for m in re.finditer(
        r"\b(?:the\s|in the\s|join(?:ed|ing)? the\s|enlist(?:ed|ing)? in the\s|served in the\s)?(?P<x>military|army|navy|marines|marine corps|air force|national guard|coast guard)\b",
        t,
    ):
        out.append(Span(doc.field, m.start("x"), m.end("x"), "OCCUPATION", "rule:military", 0.8, "military service"))
    for m in re.finditer(r"\b(?:a\s)?(?P<x>veteran|vet)\b(?!\s?(?:inary|erinarian|clinic))", t):
        if m.group("x") == "vet" and not re.search(
            r"\b(?:military|army|navy|war|va)\b", t[max(0, m.start() - 60) : m.end() + 60]
        ):
            continue
        out.append(Span(doc.field, m.start("x"), m.end("x"), "OCCUPATION", "rule:veteran", 0.8, "veteran status"))
    if doc.field == "note":
        for m in re.finditer(
            r"(?:^|\n)\s*(?:Occupation|Employment|Profession|Works as)\s?:\s?(?P<x>[^\n.;]{2,60})", doc.view
        ):
            out.append(
                Span(
                    doc.field, m.start("x"), m.end("x"), "OCCUPATION", "rule:occupation_field", 0.9, "occupation field"
                )
            )
    return out


def holidays(doc: Doc) -> list[Span]:
    out = []
    for s, e, _ in _phrase_scan(doc.lower, L.HOLIDAYS, 4):
        out.append(Span(doc.field, s, e, "HOLIDAY", "gazetteer:holiday", 0.8, "named holiday (dates the visit)"))
    return out


def activities(doc: Doc) -> list[Span]:
    out, t = [], doc.lower
    for s, e, ph in _phrase_scan(t, L.ACTIVITY_WORDS, 3):
        before, after = t[max(0, s - 25) : s], t[e : e + 30]
        if ph == "cycling" and (
            re.search(r"(?:rapid(?:ly)?|ultra-?rapid|cell|menstrual)\s$", before)
            or re.search(r"\b(?:bipolar|mood|menstrual|disorder)\b", after)
        ):
            continue  # `rapid cycling bipolar disorder`
        if ph in ("swimming", "swim") and (
            re.search(r"\b(?:head|vision|eyes|room|everything)\s(?:is|was|keeps|feels?|starts?)\s$", before)
            or after.startswith((" sensation", " feeling"))
        ):
            continue  # `my head is swimming` (dizziness)
        if ph in ("barre", "barre classes") and re.search(r"guillain[\s-]$", before):
            continue
        out.append(Span(doc.field, s, e, "ACTIVITY", "gazetteer:activity", 0.6, "hobby / activity"))
    return out


OTHER_QI = [
    (
        r"\b(?:in|entering|starting|finished|repeating) (?:the )?(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth) grade\b|\b(?:freshman|sophomore|junior|senior) (?:in|at) (?:high school|college)\b",
        "school grade",
    ),
    (r"\b[a-z]+'s (?:restaurant|bar|diner|grill|pub|cafe|bakery)\b", "named venue"),
    (
        r"\b(?:house part(?:y|ies)|birthday part(?:y|ies)|wedding|funeral|family reunion|reunion|graduation|bar mitzvah|baby shower|cruise|road trip|camping trip|vacation|concert|tournament|convention|trade show)\b",
        "personal event",
    ),
    (
        r"\b(?:twins?|triplets?|adopted|foster (?:kid|child|care|parent)s?|widow(?:ed|er)?|divorced|homeless|(?:was|been|being|is|got|recently) incarcerated|prison|jail|parole|probation)\b(?! (?:hernia|bowel|stone|uterus|placenta))",
        "family / legal status",
    ),
    (r"\bmarried for (?:\d+|[a-z]+(?: [a-z]+)?) years\b", "marital duration"),
    (r"\b(?:my|our|his|her) (?:dog|dogs|cat|cats|puppy|kitten|horse|horses|bird|parrot|rabbit|bunny|bunnies)\b", "pet"),
    (r"\b(?:birthday)\b", "birthday reference"),
    (
        r"\b(?:big family|ten of us|\d+ (?:kids|children|grandchildren)|(?:two|three|four|five|six|seven|eight|nine|ten) (?:kids|children|grandchildren))\b",
        "family size",
    ),
]


def other_qi(doc: Doc) -> list[Span]:
    out = []
    for rx, why in OTHER_QI:
        for m in re.finditer(rx, doc.lower):
            out.append(Span(doc.field, m.start(), m.end(), "OTHER_QI", "rule:other_qi", 0.6, why))
    for s, e, ph in _phrase_scan(doc.lower, L.BRANDS | frozenset(b for b in L.EMPLOYERS if " " in b or len(b) > 4), 4):
        if " " not in ph and not re.search(
            r"\b(?:at|from|to|the|bought|buy|shop|shopping|store|eat|ate|drink|drank|use|uses|used|wear|wears)\s$",
            doc.lower[max(0, s - 12) : s],
        ):
            continue  # single brand words need a shopping/use cue (`target` the verb, `subway` the train)
        out.append(Span(doc.field, s, e, "OTHER_QI", "gazetteer:brand", 0.55, "brand / company mention"))
    return out


def detect(doc: Doc) -> list[Span]:
    return (
        places(doc)
        + facilities(doc)
        + organizations(doc)
        + occupations(doc)
        + holidays(doc)
        + activities(doc)
        + other_qi(doc)
    )
