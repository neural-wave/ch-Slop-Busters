"""Offset-preserving text utilities: normalization, dialogue/note parsing, number words.

Every function here keeps character offsets identical to the original text, so a span found on
the normalized/masked view can be applied to the original string without any mapping.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cache, cached_property

# 1:1 character map (each replacement has length 1, so offsets never shift).
_CHAR_MAP = str.maketrans(
    {
        " ": " ",  # no-break space
        " ": " ",  # narrow no-break space (used as a word separator in some notes)
        " ": " ",
        "​": " ",  # zero-width space: replaced, not deleted, to keep offsets
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "‐": "-",  # U+2010 hyphen
        "‑": "-",  # U+2011 non-breaking hyphen
        "‒": "-",
        "−": "-",
        "﹣": "-",
        "－": "-",
        "²": "2",
        "İ": "I",  # U+0130 lowercases to 2 code points; map first so .lower() stays 1:1
    }
)
_FOLD_EXTRA = {"ı": "i", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "s", "æ": "a", "Æ": "A", "œ": "o", "Œ": "O", "ð": "d", "þ": "t"}


@cache
def _fold_char(c: str) -> str:
    """Accented letter -> its ASCII base letter (one character), so ASCII rules see `jose`, `muller`."""
    if c in _FOLD_EXTRA:
        return _FOLD_EXTRA[c]
    base = unicodedata.normalize("NFD", c)[0]
    return base if base.isascii() and base.isalpha() else c

SPEAKER_RE = re.compile(r"^\[(doctor|patient|patient_guest|[A-Za-z_]+)\][ \t]?", re.M)


def normalize(text: str) -> str:
    """Same-length copy: Unicode spaces/quotes/dashes mapped to ASCII, accented letters folded.

    Folding keeps every character position (`José Núñez` -> `Jose Nunez`), so spans found on the folded
    view apply to the original text, which keeps its accents everywhere except where PII is replaced.
    """
    out = text.translate(_CHAR_MAP)
    if not out.isascii():
        out = "".join(_fold_char(c) if not c.isascii() and c.isalpha() else c for c in out)
    assert len(out) == len(text)
    return out


def mask_speaker_tags(text: str) -> str:
    """Replace `[doctor]`-style speaker tags by same-length filler so rules never read them.

    `[doctor] andrew is ...` must not look like the title rule `doctor <name>`.
    """
    return SPEAKER_RE.sub(lambda m: "\x00" * (m.end() - m.start() - 1) + " " if m.group(0).endswith(" ") else "\x00" * (m.end() - m.start()), text)


@dataclass
class Line:
    idx: int
    start: int  # offset of the first character of the line (including the tag)
    end: int  # offset of the newline (exclusive)
    text_start: int  # offset after the speaker tag
    speaker: str | None  # tag, or inherited from the previous line for continuations
    tagged: bool


class Doc:
    """One field of an encounter with cheap structural indexes."""

    def __init__(self, field: str, text: str):
        self.field = field
        self.raw = text
        self.norm = normalize(text)
        self.view = mask_speaker_tags(self.norm) if field == "dialogue" else self.norm
        self.lower = "".join(c.lower() if len(c.lower()) == 1 else c for c in self.view)
        assert len(self.lower) == len(text) == len(self.view), "offset-preserving views required"
        letters = [c for c in self.view[:4000] if c.isalpha()]  # speaker tags are masked here
        # Cased text (notes, typed chat) carries a capitalisation signal; ASR dialogue is all lowercase.
        self.cased = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.01

    @cached_property
    def lines(self) -> list[Line]:
        out, pos, prev = [], 0, None
        for i, ln in enumerate(self.norm.split("\n")):
            m = SPEAKER_RE.match(ln)
            if m:
                spk, ts, tagged = m.group(1).lower(), pos + m.end(), True
            else:
                spk, ts, tagged = (prev if prev else ("doctor" if self.field == "dialogue" else None)), pos, False
            out.append(Line(i, pos, pos + len(ln), ts, spk, tagged))
            prev = spk
            pos += len(ln) + 1
        return out

    def line_at(self, offset: int) -> Line:
        lo, hi = 0, len(self.lines) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.lines[mid].start <= offset:
                lo = mid
            else:
                hi = mid - 1
        return self.lines[lo]

    @cached_property
    def sections(self) -> list[tuple[int, int, str]]:
        """Note sections as (start, end, HEADER); text before the first header is 'PREAMBLE'."""
        if self.field != "note":
            return [(0, len(self.raw), "DIALOGUE")]
        heads = []
        for ln in self.lines:
            s = self.norm[ln.start : ln.end].strip()
            if is_header(s):
                heads.append((ln.start, s.rstrip(":").strip().upper()))
        out, cur_start, cur_name = [], 0, "PREAMBLE"
        for st, name in heads:
            out.append((cur_start, st, cur_name))
            cur_start, cur_name = st, name
        out.append((cur_start, len(self.raw), cur_name))
        return out

    def section_at(self, offset: int) -> str:
        for st, en, name in self.sections:
            if st <= offset < en:
                return name
        return "PREAMBLE"


_HEADER_RE = re.compile(r"^(?:[A-Z][A-Z /&\-()']{1,60}|CC|HPI)\s*:?$")


def is_header(line: str) -> bool:
    """ALL-CAPS section header; false headers with digits/commas (`ECG, 12/26/2020.`) excluded."""
    if not line or any(ch.isdigit() for ch in line) or "," in line or line.endswith("."):
        return False
    return bool(_HEADER_RE.match(line)) and sum(c.isalpha() for c in line) >= 2


# --------------------------------------------------------------------------- number words

UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}  # fmt: skip
_NUMW = "|".join(sorted(list(UNITS) + list(TENS) + ["hundred", "thousand"], key=len, reverse=True))
# A spelled number of up to 5 words ("one hundred and five", "twenty eighteen", "fifty-nine").
NUMWORDS = rf"(?:(?:a )?(?:{_NUMW})(?:[ -](?:and )?(?:{_NUMW})){{0,4}})"
NUM = rf"(?:\d{{1,3}}|{NUMWORDS})"
_ORDW = "|".join(sorted(ORDINALS, key=len, reverse=True))
ORDINAL_WORDS = rf"(?:(?:twenty|thirty)[ -](?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)|{_ORDW})"


def words_to_int(s: str) -> int | None:
    """Value of a spelled number.

    Handles plain cardinals ("fifty nine" 59, "one hundred and two" 102), year style
    ("nineteen fifty seven" 1957, "twenty eighteen" 2018) and spoken hundreds ("one twenty" 120).
    """
    s = s.strip().lower()
    if re.fullmatch(r"\d+", s):
        return int(s)
    toks = [t for t in re.split(r"[ -]+", s) if t and t != "and"]
    if toks and toks[0] == "a":
        toks = (["one"] + toks[1:]) if len(toks) > 1 and toks[1] in ("hundred", "thousand") else toks[1:]
    if not toks:
        return None
    if "hundred" not in toks and "thousand" not in toks and len(toks) >= 2:
        for k in (1, 2):
            a, b = _small(toks[:k]), _small(toks[k:])
            if a is None or b is None:
                continue
            if 10 <= a <= 99 and 10 <= b <= 99:  # year style
                return a * 100 + b
            if 1 <= a <= 9 and 10 <= b <= 99 and k == 1:  # spoken hundreds
                return a * 100 + b
    total, cur = 0, 0
    for t in toks:
        if t in UNITS:
            cur += UNITS[t]
        elif t in TENS:
            cur += TENS[t]
        elif t == "hundred":
            cur = max(cur, 1) * 100
        elif t == "thousand":
            total += max(cur, 1) * 1000
            cur = 0
        else:
            return None
    return total + cur


def _small(toks: list[str]) -> int | None:
    """0-99 from up to two words ("eighteen", "fifty seven", "oh five" not supported)."""
    if not toks or len(toks) > 2:
        return None
    if len(toks) == 1:
        return UNITS.get(toks[0], TENS.get(toks[0]))
    if toks[0] in TENS and toks[1] in UNITS and UNITS[toks[1]] < 10:
        return TENS[toks[0]] + UNITS[toks[1]]
    return None


def ordinal_to_int(s: str) -> int | None:
    s = s.strip().lower()
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", s)
    if m:
        return int(m.group(1))
    parts = re.split(r"[ -]", s)
    if len(parts) == 2 and parts[0] in TENS and parts[1] in ORDINALS:
        return TENS[parts[0]] + ORDINALS[parts[1]]
    return ORDINALS.get(s)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*(?:-[A-Za-z]+)*")


def iter_words(text: str):
    """Yield (start, end, word) for alphabetic tokens (apostrophes and inner hyphens kept)."""
    for m in WORD_RE.finditer(text):
        yield m.start(), m.end(), m.group(0)


def is_sentence_start(text: str, offset: int) -> bool:
    """True if `offset` begins a sentence/line/bullet (capitalisation there is uninformative)."""
    j = offset - 1
    while j >= 0 and text[j] in " \t\"'(":
        j -= 1
    return j < 0 or text[j] in ".!?:\n•-*;" or text[max(0, j - 1) : j + 1] in ("1.", "2.", "3.")
