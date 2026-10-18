"""Detectors for structured / hospital document formats (cased text).

The challenge data are lowercase dialogues and SOAP-like notes, but the robustness bonus (and real
hospital data) brings other layouts: labelled header fields (`Pt: OYELARAN-KAHN, Tomás`, `Surname:`),
role-prefixed staff (`Off-going RN: Destiny Coleman`), clinical shorthand ages (`33M`, `91F`, `38 y`),
`12-Apr-2026` dates, accession / hospital / NHS / Medicare numbers, pager-bleep-extension numbers,
UK/AU/US phone formats, postcodes, street addresses, wards/rooms/beds and institution names.
"""

from __future__ import annotations

import re

from .. import lexicons as L
from ..spans import Span
from ..text.normalize import Doc, is_sentence_start

NAME_TOK = r"[A-ZÀ-Þ][A-Za-zÀ-ÿ'’]+(?:-[A-ZÀ-Þ][A-Za-zÀ-ÿ'’]+)?"
NAME_SEQ = rf"{NAME_TOK}(?:,?[ \t]+(?:[A-Z]\.[ \t]?)?(?:\([A-Z][a-z]+\)[ \t]+)?{NAME_TOK}){{0,3}}"
TITLES = r"(?:(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Master|Sr|Sister|Nurse)\.?[ \t]+)?"

PATIENT_LABELS = r"(?:Pt|PT|Patient|PATIENT|Patient name|Patient Name|PATIENT NAME|Name|NAME|Re|RE|Surname|SURNAME|Forename\(s\)|Forenames?|First name|Last name|Client|Resident|Child|Infant|Caller|Subject)"
OTHER_LABELS = r"(?:Guardian|Parent|Mother|Father|NOK|Next of kin|Next of Kin|Emergency contact|Contact|Carer|Caregiver|Informant|Witness|Interpreter|Bystander|Landlord|Neighbour|Neighbor|Accompanied by|Brought in by|Spouse|Partner)"
CLIN_LABELS = (
    r"(?:Attending|ATTENDING|Consultant|CONSULTANT|Referring(?: physician| clinician| doctor)?|REFERRING(?: PHYSICIAN)?|Surgeon|SURGEON|"
    r"Assistant|Anaesthetist|Anesthetist|Anesthesiologist|Radiologist|RADIOLOGIST|Pathologist|PATHOLOGIST|Resident|Fellow|Registrar|"
    r"Hospitalist|GP|PCP|Physician|Provider|Prescriber|Pharmacist|Dictated by|Signed|Signed by|Verified by|Reported by|Reviewed by|"
    r"Electronically signed by|Off-going RN|On-coming RN|Off-going|On-coming|Charge nurse|Charge RN|RN|LPN|NP|PA|CNA|Nurse|Paramedic|EMT|"
    r"Crew|Midwife|Therapist|Physio|Physiotherapist|Social worker|Chaplain|Sitter|Scribe|Technologist|Sonographer|From|To|Cc|CC|cc)"
)
STOP_TOK = frozenset(
    """dob mrn sex age nhs chi ward room bed date time md do rn phd frcp frcs mbbs mrcp facs ma msc bsc dnp aprn lpn
    the and of with for a an in on at to from re subject urgent referral male female m f yo unit dept department
    hospital clinic ed er icu patient pt sex: dob: mrn: age: nok hcp poa tel ph fax mobile cell email cpt mdt sn hcpcs
    icd drg frcpath forename forenames surname postcode postal zip address addr physician consultant surgeon nurse registrar resident fellow attending
    hospitalist doctor provider therapist pharmacist paramedic technologist radiologist pathologist anaesthetist
    anesthetist anesthesiologist midwife chaplain sitter interpreter emergency medicine surgery department""".split()
)
REL = (
    r"(?i:son|daughter|wife|husband|mother|father|mum|mom|dad|granddaughter|grandson|grandmother|grandfather|sister|brother|"
    r"friend|partner|neighbou?r|carer|caregiver|niece|nephew|aunt|uncle|cousin|fianc[ée]e?|guardian|girlfriend|boyfriend|"
    r"landlord|landlady|stepmother|stepfather|stepson|stepdaughter|twin)"
)


def _ok_token(tok: str) -> bool:
    w = tok.lower().strip("'’").removesuffix("'s")
    if tok.isupper() and len(tok) <= 4 and not (L.first_names().get(w, 0) >= 0.01 or L.surnames().get(w, 0) >= 20000):
        return False  # CPT, MDT, SN: abbreviations, not names
    if w in STOP_TOK or w in L.NAME_STOP and w not in L.COMMON_EXTRA or w in L.TITLES or w in L.SCRIBE_PRODUCTS:
        return False
    if w in L.first_names() or w in L.surnames():
        return True
    plain = re.sub(
        r"[^a-z]", "", w.translate(str.maketrans("àáâãäåçèéêëìíîïñòóôõöùúûüý", "aaaaaaceeeeiiiinooooouuuuy"))
    )
    return (
        len(plain) >= 2
        and not L.is_common_word(plain, 30)
        and plain not in L.medical_vocab()
        and plain not in L.drug_tokens()
    )


def _names(doc: Doc, s0: int, seq: str, cat: str, role: str, rule: str, conf: float, out: list[Span]) -> None:
    for m in re.finditer(NAME_TOK, seq):
        tok = m.group(0)
        if not _ok_token(tok):
            break
        out.append(
            Span(
                doc.field,
                s0 + m.start(),
                s0 + m.end(),
                cat,
                f"format:{rule}",
                conf,
                f"name in a {rule.replace('_', ' ')}",
                meta={"role": role, "anchor": True},
            )
        )


def names(doc: Doc) -> list[Span]:
    v, out = doc.view, []
    for labels, cat, role, rule in ((PATIENT_LABELS, "PATIENT_NAME", "patient", "patient_field"), (OTHER_LABELS, "PERSON_NAME", "relative", "contact_field"),
                                    (CLIN_LABELS, "CLINICIAN_NAME", "clinician", "staff_field")):  # fmt: skip
        for m in re.finditer(
            rf"(?<![A-Za-z])(?:{labels})(?:\s?\([A-Za-z ]{{1,20}}\))?[ \t]{{0,3}}[:\-][ \t]{{0,3}}{TITLES}(?P<names>{NAME_SEQ})",
            v,
        ):
            _names(doc, m.start("names"), m.group("names"), cat, role, rule, 0.92, out)
        if labels is CLIN_LABELS:  # `RN Keisha Barnwell`, `Dr. Priya Natarajan` without a colon
            for m in re.finditer(
                rf"(?<![A-Za-z])(?:RN|LPN|NP|PA|EMT|Paramedic|Sitter|Interpreter|Dr\.?|Doctor|Nurse|Sister)[ \t]+\(?(?P<names>{NAME_SEQ})",
                v,
            ):
                _names(doc, m.start("names"), m.group("names"), cat, role, "staff_role", 0.9, out)
    for m in re.finditer(rf"\b{REL}\b,?[ \t]+(?:\(|named[ \t]+|called[ \t]+)?(?P<names>{NAME_SEQ})", v):
        _names(doc, m.start("names"), m.group("names"), "RELATIVE_NAME", "relative", "relative_cased", 0.88, out)
    # `Margarethe (Greta) Szabo`: nickname in parentheses
    for m in re.finditer(rf"{NAME_TOK}[ \t]+\((?P<nick>[A-Z][a-z]+)\)[ \t]+{NAME_TOK}", v):
        if _ok_token(m.group("nick")):
            out.append(
                Span(
                    doc.field,
                    m.start("nick"),
                    m.end("nick"),
                    "PERSON_NAME",
                    "format:nickname",
                    0.85,
                    "nickname in parentheses",
                    meta={"role": "person", "anchor": True},
                )
            )
    # `"Harold" (late husband)`: a quoted name explained as a relation
    for m in re.finditer(rf"[\"“](?P<n>{NAME_TOK})[\"”]\s?\((?:late\s|her\s|his\s)?{REL}", v):
        out.append(
            Span(
                doc.field,
                m.start("n"),
                m.end("n"),
                "RELATIVE_NAME",
                "format:quoted_relative",
                0.85,
                "named relative",
                meta={"role": "relative", "anchor": True},
            )
        )
    # e-mail headers: `From: Helena Voss <h.voss@...>`
    for m in re.finditer(rf"(?P<names>{NAME_SEQ})[ \t]*<[\w.+-]+@", v):
        _names(doc, m.start("names"), m.group("names"), "PERSON_NAME", "person", "email_header", 0.9, out)
    return out


def ages(doc: Doc) -> list[Span]:
    v, out = doc.view, []

    def add(m, rule, reason):
        n = int(m.group("num"))
        if n <= 110:
            out.append(
                Span(
                    doc.field,
                    m.start("num"),
                    m.end("num"),
                    "AGE",
                    f"format:{rule}",
                    0.9,
                    reason,
                    value=n,
                    meta={"unit": "year", "num_start": m.start("num"), "num_end": m.end("num")},
                )
            )

    for m in re.finditer(r"(?<![\d.,/])\b(?P<num>\d{1,3})\s?(?P<sex>[MF])\b(?![a-z/])", v):
        before, after = v[max(0, m.start() - 30) : m.start()], v[m.end() : m.end() + 15]
        if re.search(
            r"(?i)(?:\bT\s?(?:max)?|temp\w*|°|fever\w*|febrile|pyrexi\w*|x|\d)\b[^.\n]{0,25}$|(?:than|to|of|above|over|at)\s*$",
            before,
        ):
            continue  # `fever to 102 F`, `more than 100F`
        if re.match(
            r"(?i)\s?(?:Fr\b|French|Foley|catheter|sheath|pigtail|drain|tube|introducer|NG\b|cannula|gauge|g\b)", after
        ):
            continue  # `16 F Foley` is a catheter size
        add(m, "age_sex", "age with sex shorthand ('33M')")
    for m in re.finditer(
        r"(?<![\d.,/])\b(?P<num>\d{1,3})\s?(?:y|Y|yr|yrs|y/o|yo)\b(?=\s?(?:[)\],;]|[MF]\b|old\b|male|female|man|woman|boy|girl))",
        v,
    ):
        add(m, "age_y", "age shorthand ('38 y')")
    for m in re.finditer(r"(?i)\b(?:age(?:/sex)?|aged|age/gender)\s?:?\s?(?P<num>\d{1,3})\b(?!\s?[+])", v):
        if not re.search(
            r"(?i)(?:gestational|corrected|bone|post-?menstrual|conceptional)\s$", v[max(0, m.start() - 16) : m.start()]
        ):
            add(m, "age_field", "age field")
    for m in re.finditer(rf"\b{REL}[ \t]+{NAME_TOK}[ \t]+\((?P<num>\d{{1,2}})\)", v):
        add(m, "relative_age", "relative's age in parentheses")
    return out


MONTH_ABBR = r"(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"


def dates(doc: Doc) -> list[Span]:
    v, out = doc.view, []
    for m in re.finditer(rf"\b\d{{1,2}}[-/ ]{MONTH_ABBR}[-/ ,]+(?:19|20)?\d{{2}}\b", v):
        out.append(
            Span(
                doc.field, m.start(), m.end(), "DATE", "format:date_dd_mon_yyyy", 0.95, "absolute date ('12-Apr-2026')"
            )
        )
    for m in re.finditer(r"\b\d{1,2}(?i:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?:19|20)?\d{2}\b", v):
        out.append(
            Span(doc.field, m.start(), m.end(), "DATE", "format:date_compact", 0.95, "compact date ('15OCT2026')")
        )
    return out


ID_RX = [
    ("alnum_id", r"\b[A-Z]{2,5}-?\d{5,}[A-Z]?\b"),
    ("prefixed_id", r"\b[A-Z]{1,4}-[A-Z]?\d{4,}(?:-\d{1,6})?\b"),
    ("accession", r"\b[A-Z]\d{2}-\d{4,6}(?:-\d{3,6})?\b"),
    ("pager", r"(?i:\b(?:pager|bleep|beeper|ext\.?|extn|extension|badge|crew id|unit id)\s?#?:?\s?)(?P<id>\d{2,6})\b"),
]
PHONE_RX = [
    r"\+\d{1,3}[\s-]?(?:\(?\d{1,4}\)?[\s-]?){2,4}\d{2,4}\b",
    r"\(0\d{1,4}\)\s?\d{3,4}\s?\d{3,4}\b",
    r"\b0\d{2,4}\s\d{3,4}\s\d{3,4}\b",
    r"\b0\d{4}\s?\d{6}\b",
    r"\b04\d{2}\s?\d{3}\s?\d{3}\b",
    r"\(\d{3}\)\s?\d{3}-\d{4}\b",
]


def identifiers(doc: Doc) -> list[Span]:
    v, out = doc.view, []
    for name, rx in ID_RX:
        for m in re.finditer(rx, v):
            g = "id" if "id" in m.groupdict() else 0
            out.append(
                Span(
                    doc.field,
                    m.start(g),
                    m.end(g),
                    "ID",
                    f"format:{name}",
                    0.9,
                    f"identifier ({name.replace('_', ' ')})",
                )
            )
    for rx in PHONE_RX:
        for m in re.finditer(rx, v):
            out.append(
                Span(
                    doc.field,
                    m.start(),
                    m.end(),
                    "PHONE",
                    "format:phone_intl",
                    0.9,
                    "phone number (international formats)",
                )
            )
    return out


STREET_SUFFIX = r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Close|Approach|Row|Way|Parade|Crescent|Cres|Mews|Place|Pl|Square|Sq|Grove|Gardens|Terrace|Walk|Rise|Green|Trail|Path|Loop|Pike|Highway|Hwy|Parkway|Circle|Ridge|View|Heights|Hill|Marg|Nagar)"
POSTCODES = r"(?:\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b|\b(?:NSW|VIC|QLD|SA|WA|TAS|ACT|NT)\s\d{4}\b|\b(?!G\dP\d|S\dQ\dT\d|T\d[NM]\d|P\d[A-Z]\d)[A-Z]\d[A-Z]\s?\d[A-Z]\d\b|\b[A-Z]{2}\s\d{5}(?:-\d{4})?\b)"


def addresses(doc: Doc) -> list[Span]:
    v, out = doc.view, []
    for m in re.finditer(rf"\b\d{{1,5}}[A-Za-z]?[ \t]+(?:[A-Z][A-Za-z'’]+[ \t]+){{1,3}}{STREET_SUFFIX}\b\.?", v):
        out.append(Span(doc.field, m.start(), m.end(), "ADDRESS", "format:street_address", 0.95, "street address"))
    for m in re.finditer(r"(?<![A-Za-z])(?:Apt|Apartment|Unit|Flat|Suite|Ste)\.?[ \t]?#?[ \t]?\d{1,4}[A-Z]?\b", v):
        out.append(Span(doc.field, m.start(), m.end(), "ADDRESS", "format:unit", 0.9, "apartment / unit"))
    for m in re.finditer(POSTCODES, v):
        out.append(Span(doc.field, m.start(), m.end(), "ADDRESS", "format:postcode", 0.9, "postcode / ZIP"))
    for m in re.finditer(rf"\b(?P<town>(?:[A-Z][a-z]+[ \t]){{0,2}}[A-Z][a-z]+),?[ \t]{POSTCODES}", v):
        if _ok_place(m.group("town")):
            out.append(
                Span(
                    doc.field,
                    m.start("town"),
                    m.end("town"),
                    "LOCATION",
                    "format:town_before_postcode",
                    0.9,
                    "town in an address",
                )
            )
    for m in re.finditer(
        r"\b(?:[A-Z][a-z]+[ \t]){1,3}(?:Trail|Road|Street|Avenue|Lane|Drive|Way|Parade|Crescent|Highway|Path|Park)\b", v
    ):
        if not is_sentence_start(v, m.start()) and any(_ok_place(w) for w in m.group(0).split()[:-1]):
            out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "format:named_road", 0.8, "named road / trail"))
    # wards, rooms, beds
    for m in re.finditer(
        r"(?<![A-Za-z])(?i:room|rm|bed|bay|ward|cubicle|pod|stall|resus)[ \t]?#?[ \t]?\d{1,4}[A-Z]?(?:-[A-Z0-9]{1,3})?\b",
        v,
    ):
        out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "format:ward_room_bed", 0.85, "ward / room / bed"))
    for m in re.finditer(r"\b(?P<w>[A-Z][a-z]+)[ \t]Ward\b", v):
        if m.group("w").lower() not in L.first_names() and m.group("w").lower() not in (
            "the",
            "this",
            "medical",
            "surgical",
        ):
            out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "format:named_ward", 0.85, "named ward"))
    for m in re.finditer(r"\b\d{1,2}[ \t]?(?:North|South|East|West)\b(?:[ \t][A-Z][a-z]+)?", v):
        out.append(Span(doc.field, m.start(), m.end(), "LOCATION", "format:unit_wing", 0.8, "hospital unit"))
    # `lives in Brookhollow`, `from Millbrook`: unknown capitalised place after a locative
    for m in re.finditer(
        r"\b(?:in|from|near|outside|to|at|of)[ \t](?P<x>[A-Z][a-z]{3,}(?:[ \t][A-Z][a-z]+)?)\b(?![-\d])", v
    ):
        words = m.group("x").split()
        lead = doc.lower[max(0, m.start() - 30) : m.start("x")]
        residence = re.search(
            r"\b(?:live|lives|living|lived|moved|moving|relocated|born|raised|grew up|from|trip|vacation|visiting|visited|drove|flew|travel\w*|resides|resident|home)\b",
            lead,
        )
        if residence and all(_ok_place(w) and w.lower() not in L.first_names() for w in words):
            out.append(
                Span(
                    doc.field,
                    m.start("x"),
                    m.end("x"),
                    "LOCATION",
                    "format:locative_place",
                    0.7,
                    "capitalised place after a locative",
                )
            )
    return out


def _ok_place(w: str) -> bool:
    lw = w.lower()
    return (lw not in L.medical_vocab() and lw not in L.drug_tokens() and lw not in L.MEDICAL_EPONYMS and not L.is_common_word(lw, 10)
            and lw not in L.NAME_STOP and not re.fullmatch(L.MONTHS, lw) and lw not in STOP_TOK)  # fmt: skip


FAC_HEADS = (
    r"(?:Hospital|Medical Center|Medical Centre|Health Center|Health Centre|Clinic|Practice|Family Practice|Family Medicine|"
    r"Medical Group|Imaging(?: Center| Centre| Services)?|Radiology|Pediatrics|Paediatrics|Diagnostics|Diagnostic Centre|"
    r"Diagnostic Center|Care Home|Nursing Home|Assisted Living|Rehabilitation(?: Centre| Center)?|Hospice|Pharmacy|Chemist|"
    r"Urgent Care|Sports Medicine|Orthopaedics|Orthopedics|Infirmary|Institute|Heart Institute|Medical Practice|Surgery)"
)
ORG_HEADS = (
    r"(?:School|Primary School|High School|Academy|University|College|Nursery|Daycare|Day Care|Preschool|Kindergarten|Church|"
    r"Mosque|Temple|Synagogue|Lutheran|Baptist|Methodist|Catholic|Symphony|Orchestra|Solutions|Logistics|Energy|Marine|"
    r"Freight|Group|Inc|LLC|Ltd|Limited|Company|Corporation|Corp|Bank|Insurance|Partners|Associates|Foundation|Laboratories|"
    r"Labs|Services|Systems|Technologies|Manufacturing|Construction|Motors|Foods|Bakery|Restaurant|Cafe|Hotel|Equestrian Centre|"
    r"Centre|Center|Club|Stores|Market|Farm|Ranch|Council|Authority|Police|Fire Department|Army|Navy)"
)


def institutions(doc: Doc) -> list[Span]:
    v, out = doc.view, []
    for heads, cat in ((FAC_HEADS, "FACILITY"), (ORG_HEADS, "ORGANIZATION")):
        for upper in (False, True):
            head_rx = heads.upper() if upper else heads
            word = r"[A-Z][A-Z'’&\.\-]+" if upper else r"[A-Z][A-Za-z'’&\.\-]+"
            for m in re.finditer(rf"\b(?P<name>(?:{word}[ \t]){{1,4}}){head_rx}\b", v):
                words = [w for w in m.group("name").split()]
                if words and words[0].lower() in (
                    "the",
                    "a",
                    "an",
                    "our",
                    "your",
                    "his",
                    "her",
                    "at",
                    "to",
                    "in",
                    "and",
                ):
                    words = words[1:]
                if not words or all(
                    L.is_common_word(w.lower().strip(".'’"), 30) or w.lower() in L.medical_vocab() for w in words
                ):
                    continue  # "Emergency Department", "Primary Care Clinic"
                s = m.end("name") - len(" ".join(words)) - 1
                out.append(
                    Span(
                        doc.field,
                        s,
                        m.end(),
                        cat,
                        "format:institution" + ("_caps" if upper else ""),
                        0.85,
                        f"named {cat.lower()}",
                    )
                )
    return out


def detect(doc: Doc) -> list[Span]:
    if not doc.cased:
        return dates(doc) + identifiers(doc)
    return names(doc) + ages(doc) + dates(doc) + identifiers(doc) + addresses(doc) + institutions(doc)
