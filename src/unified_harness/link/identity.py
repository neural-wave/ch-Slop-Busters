"""Per-encounter identity graph.

Collect every person-name anchor found in either field, group anchors into entities (first + last
names that occur side by side, the same token across fields), then sweep *both* fields for every
mention of every entity token: bare vocatives late in the dialogue (`bye otto`), the surname that
only the note states (`Mr. Ellison`), possessives, and typo variants (`katharyn` vs `Kathryn`).

Tokens that are also ordinary words (`white`, `cook`, `mark`) are only swept where a name cue is
present (title before, the entity's other token next to it, capitalised mid-sentence in the note,
vocative position in the dialogue), so `white blood cell` survives while `Dr. White` does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import lexicons as L
from ..detect.rules import _GREET, _VOC_POST, _VOC_PRE, ROLE_CAT
from ..spans import Span
from ..text.normalize import Doc, is_sentence_start

ROLE_WEIGHT = {"clinician": 1.3, "patient": 1.0, "staff": 1.1, "relative": 1.1, "person": 0.5}


@dataclass
class Entity:
    key: str
    role: str
    tokens: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    conf: float = 0.0
    label: str = ""  # placeholder text, filled by the policy


def _tok(s: Span, docs: dict[str, Doc]) -> str:
    return docs[s.field].lower[s.start : s.end].removesuffix("'s").strip("-'")


def build_entities(anchors: list[Span], docs: dict[str, Doc], weak_threshold: float) -> list[Entity]:
    """Group anchor spans into entities (union-find on adjacency and shared tokens)."""
    anchors = [a for a in anchors if a.conf >= weak_threshold]
    parent = list(range(len(anchors)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    toks = [_tok(a, docs) for a in anchors]
    roles = [a.meta.get("role", "person") for a in anchors]
    by_pos = sorted(range(len(anchors)), key=lambda i: (anchors[i].field, anchors[i].start))
    for x, y in zip(by_pos, by_pos[1:], strict=False):
        a, b = anchors[x], anchors[y]
        gap = docs[a.field].lower[a.end : b.start]
        side_by_side = gap.strip() == "" or (gap.strip() == "," and roles[x] == roles[y])  # `DUFRESNE, Jerome`
        if a.field == b.field and 0 < b.start - a.end <= 3 and side_by_side and toks[x] != toks[y]:
            union(x, y)  # first + last name side by side
    first_seen: dict[tuple[str, str], int] = {}
    for i, t in enumerate(toks):
        cls = "clin" if roles[i] == "clinician" else "other"
        key = (t, cls)
        if key in first_seen:
            union(i, first_seen[key])
        else:
            first_seen[key] = i
    # A token seen only as a non-clinician anchor and also as clinician: keep separate (two people).
    groups: dict[int, list[int]] = {}
    for i in range(len(anchors)):
        groups.setdefault(find(i), []).append(i)
    ents: list[Entity] = []
    for n, idxs in enumerate(groups.values()):
        score: dict[str, float] = {}
        for i in idxs:
            score[roles[i]] = score.get(roles[i], 0.0) + anchors[i].conf * ROLE_WEIGHT.get(roles[i], 0.5)
        role = max(score, key=score.get)
        if role == "person" and "patient" in score:
            role = "patient"
        ents.append(
            Entity(
                key=f"E{n + 1}",
                role=role,
                tokens={toks[i] for i in idxs if len(toks[i]) >= 2},
                sources={anchors[i].source for i in idxs},
                conf=max(anchors[i].conf for i in idxs),
            )
        )
    # Merge person-role singletons into an existing entity sharing a token across classes.
    return ents


def _name_context(doc: Doc, s: int, e: int, ent: Entity) -> bool:
    """Is a common-word token at [s,e) used as this entity's name here?"""
    t = doc.lower
    if re.search(r"\b(?:mr|mrs|ms|miss|mister|missus|dr|doctor)\s?\.?\s$", t[max(0, s - 12) : s]):
        return True
    before = re.findall(r"[a-z][a-z'\-]*|[,.?!]", t[max(0, s - 40) : s])
    after = re.findall(r"[a-z][a-z'\-]*|[,.?!]", t[e : e + 40])
    prev = before[-1] if before else "<s>"
    nxt = after[0] if after else "</s>"
    if prev.removesuffix("'s") in ent.tokens or nxt.removesuffix("'s") in ent.tokens:
        return True
    if doc.field == "note" or doc.cased:
        if not doc.view[s].isupper():
            return False
        if not is_sentence_start(doc.view, s) or re.search(r"\b[A-Z]\.\s*$", doc.view[max(0, s - 4) : s]):
            return True
        # sentence-initial `Mark reports ...` / `Grace Hill, 70-year-old`: a linked name unless the word sense follows
        return t[s:e] not in VERY_COMMON and nxt not in WORD_SENSE_NEXT
    line = doc.line_at(s)
    at_line_start = t[line.text_start : s].strip(" \x00") == ""
    at_line_end = t[e : line.end].strip(" .,?!") == ""
    if (prev in _VOC_PRE or prev in _GREET or prev == "," or at_line_start) and (nxt in _VOC_POST or at_line_end):
        return True
    # Moderately common words (`mark`, `cook`, `walker`, `price`): once the encounter has anchored the
    # token as a name, accept it unless the local syntax shows the ordinary-word sense.
    tok = t[s:e]
    if L.lower_uses(tok) < 1000 and tok not in VERY_COMMON:
        return prev not in WORD_SENSE_PREV and nxt not in WORD_SENSE_NEXT
    return False


VERY_COMMON = frozenset(
    "will may white black brown green young long short hope grace joy faith summer fall spring winter".split()
)
WORD_SENSE_PREV = L.WORD_SENSE_PREV
WORD_SENSE_NEXT = L.WORD_SENSE_NEXT


def _edit_distance(a: str, b: str, limit: int) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            best = min(best, cur[j])
        if best > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def sweep(ents: list[Entity], docs: dict[str, Doc]) -> list[Span]:
    """Find every mention of every entity token in both fields (+ fuzzy typo variants)."""
    out: list[Span] = []
    for ent in ents:
        cat = ROLE_CAT.get(ent.role, "PERSON_NAME")
        for tok in sorted(ent.tokens):
            if tok in L.NAME_STOP or tok in L.SCRIBE_PRODUCTS:
                continue
            common = L.is_common_word(tok)
            rx = re.compile(rf"(?<![a-z\-]){re.escape(tok)}(?![a-z])")  # doc.lower is accent-folded and 1:1
            for f, doc in docs.items():
                for m in rx.finditer(doc.lower):
                    if common and not _name_context(doc, m.start(), m.end(), ent):
                        continue
                    out.append(
                        Span(
                            f,
                            m.start(),
                            m.end(),
                            cat,
                            "identity:sweep",
                            min(0.95, ent.conf),
                            f"mention of {ent.role} name linked across fields",
                            entity=ent.key,
                        )
                    )
        # Initials (`J.D.`) and letter-by-letter spellings (`L-I-N-D-A-H-L`) of a linked name.
        toks = sorted(ent.tokens, key=len, reverse=True)
        for f, doc in docs.items():
            for tok in toks:
                if len(tok) >= 3:
                    spelled = r"\b" + r"[- ]".join(re.escape(ch) for ch in tok) + r"\b"
                    for m in re.finditer(spelled, doc.lower):
                        out.append(
                            Span(
                                f,
                                m.start(),
                                m.end(),
                                cat,
                                "identity:spelled",
                                0.9,
                                f"{ent.role} name spelled out",
                                entity=ent.key,
                            )
                        )
            pat_toks = {t for e2 in ents if e2.role == "patient" for t in e2.tokens} if ent.role == "patient" else set()
            if ent.role == "patient" and len(pat_toks) >= 2:
                firsts = {t[0].upper() for t in pat_toks}
                for m in re.finditer(r"(?<![A-Za-z.])([A-Z])\.\s?([A-Z])\.(?![A-Za-z])", doc.view):
                    if {m.group(1), m.group(2)} <= firsts and m.group(1) != m.group(2):
                        out.append(
                            Span(
                                f,
                                m.start(),
                                m.end(),
                                cat,
                                "identity:initials",
                                0.85,
                                "initials of the linked patient name",
                                entity=ent.key,
                            )
                        )
        # Fuzzy variants (ASR/typo): same first letter, edit distance 1 (2 for long names).
        for tok in [t for t in ent.tokens if len(t) >= 5 and not L.is_common_word(t)]:
            limit = 2 if len(tok) >= 8 else 1
            for f, doc in docs.items():
                for m in re.finditer(r"[a-z][a-z\-]{3,}", doc.lower):
                    w = m.group(0)
                    if w == tok or w[0] != tok[0] or w in ent.tokens or L.lower_uses(w) >= 5:
                        continue
                    if w in L.medical_vocab() or w in L.drug_tokens() or w in L.NAME_STOP:
                        continue
                    if _edit_distance(w, tok, limit) <= limit:
                        out.append(
                            Span(
                                f,
                                m.start(),
                                m.end(),
                                cat,
                                "identity:fuzzy",
                                0.85,
                                f"spelling variant of a linked {ent.role} name",
                                entity=ent.key,
                            )
                        )
    return out
