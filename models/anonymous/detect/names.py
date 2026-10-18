"""Layer 1b: structural name detection + cross-field propagation.

Names are found from strong structural cues (note opening sentence, honorifics, greetings, self-introductions,
relation words), assigned a role (patient / clinician / relative), and then propagated to every occurrence in BOTH
fields so the note never leaks what the dialogue hid (first-name <-> full-name linking).
"""
from __future__ import annotations

import functools
import re

from common import NEVER_PHI, Span, find_all

try:
    from spacy.lang.en.stop_words import STOP_WORDS
except Exception:  # noqa: BLE001
    STOP_WORDS = set()

NOT_NAMES = set(STOP_WORDS) | NEVER_PHI | {
    "um", "uh", "mm", "hmm", "okay", "ok", "yeah", "yes", "no", "so", "well", "good", "great", "sure", "thanks",
    "thank", "how", "doing", "today", "morning", "afternoon", "evening", "there", "said", "says", "wanted", "put",
    "told", "thinks", "thought", "recommended", "gave", "prescribed", "office", "visit", "appointment", "sir",
    "ma'am", "maam", "everyone", "everybody", "guys", "dear", "honey", "buddy", "friend", "nice", "again", "back",
    "here", "now", "then", "just", "also", "i'm", "it's", "that's", "you're", "we're", "let's", "is", "was", "will",
    "can", "could", "would", "should", "and", "but", "or", "the", "a", "an", "to", "of", "in", "on", "at", "for",
}
EPONYM_CONTEXT = re.compile(
    r"^\s*(?:'s|’s|s)?\s*(?:disease|syndrome|palsy|dementia|sign|test|maneuver|reflex|phenomenon|fracture|"
    r"tendon|tendinitis|tendonitis|cyst|ulcer|esophagus|thyroiditis|lymphoma|sarcoma|chorea|neuroma|"
    r"contracture|node|nodes|law|score|criteria|scale|classification|procedure|operation)\b", re.IGNORECASE)
RELATIONS = (r"wife|husband|son|daughter|mother|mom|mum|father|dad|brother|sister|grandson|granddaughter|grandma|"
             r"grandmother|grandpa|grandfather|partner|boyfriend|girlfriend|fiance|fiancee|friend|aunt|uncle|cousin|"
             r"niece|nephew|kid|child|baby|stepson|stepdaughter|neighbor|roommate|coworker|boss|caregiver")
HONOR = r"(?:mr|ms|mrs|miss|mister|missus)"
DOC = r"(?:dr|doctor)"
W = r"([a-z][a-z'\-]+)"


@functools.lru_cache(maxsize=1)
def _dictionary():
    """Lower-case common words (a capitalized-only entry like 'James' is a proper name, not a common word)."""
    lower, proper = set(), set()
    try:
        for w in open("/usr/share/dict/american-english-large", encoding="utf-8", errors="ignore"):
            w = w.strip()
            if w and "'" not in w:
                (lower if w[0].islower() else proper).add(w.lower())
    except FileNotFoundError:
        pass
    return lower, proper


def is_common_word(tok: str) -> bool:
    return tok.lower() in _dictionary()[0]


def _ok(tok: str | None) -> bool:
    return bool(tok) and len(tok) > 1 and tok.lower() not in NOT_NAMES and not tok.isdigit()


class NameRegistry:
    """Per-encounter registry of discovered names with roles; shared by dialogue and note."""

    def __init__(self):
        self.names: dict[str, dict] = {}   # lowercase surface -> {"role", "reason", "strong"}

    def add(self, surface: str, role: str, reason: str, strong: bool = True):
        surface = re.sub(r"\s+", " ", surface.strip().lower())
        toks = [t for t in surface.split(" ") if _ok(t)]
        if not toks:
            return
        surface = " ".join(toks)
        cur = self.names.get(surface)
        # patient role wins over generic; keep the first reason
        if cur is None or (role == "NAME_PATIENT" and cur["role"] != "NAME_PATIENT"):
            self.names[surface] = {"role": role, "reason": reason, "strong": strong}
        if len(toks) > 1:  # register parts so "andrew" in the dialogue links to "Andrew Campbell" in the note
            for t in toks:
                if t not in self.names:
                    self.names[t] = {"role": role, "reason": f"part of '{surface}' ({reason})", "strong": strong,
                                     "part_of": surface}


def harvest(rec: dict, reg: NameRegistry):
    d, n = rec["dialogue"], rec["note"]
    nl = n
    # note opening: "Andrew Campbell is a 59-year-old" / "Ms. Martinez is a 43-year-old"
    for m in re.finditer(r"(?:^|\n)\s*(?:(?:Mr|Ms|Mrs|Miss|Dr)\.?\s+)?([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,2})\s*,?\s+(?:is|was)?\s*an?\s+(?:\d|[a-z]+-year)", nl):
        reg.add(m.group(1), "NAME_PATIENT", "note: subject of the opening 'X is a N-year-old' sentence")
    for m in re.finditer(r"\b(?:Patient(?:'s)?\s+name|Name|Patient)\s*:\s*([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,2})", nl):
        reg.add(m.group(1), "NAME_PATIENT", "note: 'Name:' header")
    for fld, text in (("dialogue", d), ("note", n)):
        low = text.lower()
        for m in re.finditer(rf"\b{HONOR}\s*\.?\s+{W}(?:\s+{W})?", low):
            first, second = m.group(1), m.group(2)
            if m.group(0).startswith("miss") and not re.search(r"(^|\]|,|\.|hi|hello|hey)\s*$", low[max(0, m.start() - 12):m.start()]):
                continue
            if _ok(first):
                reg.add(first, "NAME_PATIENT", f"{fld}: follows honorific", strong=True)
                if _ok(second) and not is_common_word(second) and fld == "note":
                    reg.add(f"{first} {second}", "NAME_PATIENT", f"{fld}: follows honorific")
        for m in re.finditer(rf"\b{DOC}\s*\.?\s+{W}(?:\s+{W})?", low):
            first, second = m.group(1), m.group(2)
            if not _ok(first) or first in {"office", "visit", "appointment", "note", "orders", "said"}:
                continue
            ctx_ok = bool(re.match(r"dr\b", m.group(0))) or not is_common_word(first)
            if ctx_ok:
                reg.add(first, "NAME_CLINICIAN", f"{fld}: follows 'dr'/'doctor'")
                if _ok(second) and not is_common_word(second) and not re.match(r"(and|or|is|was|said)$", second):
                    reg.add(f"{first} {second}", "NAME_CLINICIAN", f"{fld}: follows 'dr'/'doctor'")
        for m in re.finditer(rf"\b(?:i'm|i am|this is|my name is|name's)\s+(?:{DOC}\s*\.?\s+){W}", low):
            if _ok(m.group(1)):
                reg.add(m.group(1), "NAME_CLINICIAN", f"{fld}: clinician self-introduction")
        for m in re.finditer(rf"(?:^|\n)\[doctor\]\s*(?:hi|hey|hello|good morning|good afternoon|good evening|morning|afternoon)\s*[,.]?\s*(?:there\s*[,.]?\s*)?(?:{HONOR}\s*\.?\s*)?{W}", low):
            if _ok(m.group(1)) and not is_common_word(m.group(1)):
                reg.add(m.group(1), "NAME_PATIENT", f"{fld}: addressed in the doctor's greeting")
        for m in re.finditer(rf"\b(?:seeing|see|meet|meeting)\s+(?:{HONOR}\s*\.?\s+)?{W}(?:\s+{W})?(?=\s*(?:\.|,|today|who|she|he|is|today))", low):
            if _ok(m.group(1)) and not is_common_word(m.group(1)):
                reg.add(m.group(1), "NAME_PATIENT", f"{fld}: 'today I am seeing X'")
                if _ok(m.group(2)) and not is_common_word(m.group(2)):
                    reg.add(f"{m.group(1)} {m.group(2)}", "NAME_PATIENT", f"{fld}: 'today I am seeing X Y'")
        for m in re.finditer(rf"(?:^|\]\s*|\.\s+|so\s*,?\s*){W}(?:\s+{W})?\s+is\s+an?\s+(?:\d+|[a-z]+(?:[\s-][a-z]+)?)\s*-?\s*year", low):
            for g in (1, 2):
                if _ok(m.group(g)) and not is_common_word(m.group(g)):
                    reg.add(m.group(g), "NAME_PATIENT", f"{fld}: subject of 'X is a N-year-old'")
        for m in re.finditer(rf"\bmy name is\s+{W}(?:\s+{W})?", low):
            if _ok(m.group(1)):
                reg.add(m.group(1), "NAME_PATIENT", f"{fld}: 'my name is X'")
        for m in re.finditer(rf"\b(?:my|your|his|her|their|our)\s+(?:{RELATIONS})s?\s*,?\s*{W}", low):
            tok = m.group(1)
            if _ok(tok) and not is_common_word(tok):
                reg.add(tok, "NAME_RELATIVE", f"{fld}: follows a relation word ('{m.group(0).split()[1]}')")


def propagate(rec: dict, reg: NameRegistry) -> list[Span]:
    spans = []
    for surface, info in sorted(reg.names.items(), key=lambda kv: -len(kv[0])):
        common = all(is_common_word(t) for t in surface.split())
        for fld in ("dialogue", "note"):
            text = rec[fld]
            for s, e in find_all(text, surface):
                after = text[e:e + 30]
                if EPONYM_CONTEXT.match(after):
                    continue  # "parkinson's disease", "baker's cyst" -> medical eponym, not a person
                if common:
                    # a common-word surname ("wood", "hill") only counts in honorific / name context
                    before = text[max(0, s - 12):s].lower()
                    cap = text[s].isupper()
                    if not (re.search(rf"\b(?:{HONOR}|{DOC})\s*\.?\s*$", before) or (fld == "note" and cap and s > 0 and text[s - 1] != "\n")):
                        continue
                spans.append(Span(fld, s, e, text[s:e], info["role"], "names", 0.95 if info["strong"] else 0.8,
                                  f"name '{surface}' — {info['reason']}; propagated to all occurrences in both fields"))
    return spans


def detect(rec: dict) -> tuple[list[Span], NameRegistry]:
    reg = NameRegistry()
    harvest(rec, reg)
    return propagate(rec, reg), reg
