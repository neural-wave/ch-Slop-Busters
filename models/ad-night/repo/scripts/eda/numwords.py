"""Spelled-out number normalisation for `aci` dialogues (analysis helper).

`spans(text)` yields (start, end, value_str, kind) for maximal runs of number
words such as `fifty nine`, `one twenty over fifty`, `seven . four`,
`twenty eighteen`, `two thousand eighteen`, `eighty one`.  Digits are returned
too so callers can treat both uniformly.
"""
import re

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}
SCALES = {"hundred": 100, "thousand": 1000}
WORD = set(UNITS) | set(TENS) | set(SCALES) | {"and", "point"}
NUM_RE = re.compile(r"\b(?:(?:" + "|".join(UNITS | TENS | SCALES) + r")(?:[ -](?:" + "|".join(UNITS | TENS | SCALES) + r"|and|point))*(?: \. (?:" + "|".join(UNITS | TENS) + r")(?:[ -](?:" + "|".join(UNITS) + r"))?)?|\d+(?:[.,]\d+)*)\b")


def _value(tokens):
    """Standard English number-word evaluation; returns (int, kind)."""
    total, cur = 0, 0
    kind = "cardinal"
    prev_tens = None
    for t in tokens:
        if t in ("and",):
            continue
        if t in UNITS:
            v = UNITS[t]
            if prev_tens is not None and v >= 10:          # "twenty eighteen" -> 2018 (year style) / "one twenty" handled below
                cur = cur * 100 + v; kind = "year-ish"
            else:
                cur += v
            prev_tens = None
        elif t in TENS:
            if cur and cur < 10 and prev_tens is None and total == 0:   # "one twenty" -> 120
                cur = cur * 100 + TENS[t]; kind = "spoken-hundreds"
            elif 10 <= cur < 100 and prev_tens is None and total == 0:  # "nineteen fifty (seven)" -> 1957
                cur = cur * 100 + TENS[t]; kind = "year-ish"
            elif prev_tens is not None:                                  # "twenty twenty" -> 2020
                cur = cur * 100 + TENS[t]; kind = "year-ish"
            else:
                cur += TENS[t]
            prev_tens = TENS[t]
        elif t in SCALES:
            cur = max(cur, 1) * SCALES[t]
            if SCALES[t] == 1000:
                total += cur; cur = 0
            prev_tens = None
    return total + cur, kind


def spans(text):
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group(0)
        if s[0].isdigit():
            out.append((m.start(), m.end(), s, "digits"))
            continue
        if " . " in s or " point " in s:
            a, b = re.split(r" \. | point ", s, maxsplit=1)
            va, _ = _value(re.split(r"[ -]", a)); vb, _ = _value(re.split(r"[ -]", b))
            out.append((m.start(), m.end(), f"{va}.{vb}", "spoken-decimal"))
            continue
        v, kind = _value(re.split(r"[ -]", s))
        out.append((m.start(), m.end(), str(v), kind))
    return out


if __name__ == "__main__":
    for t in ["fifty nine year old", "one twenty over fifty", "seven . four", "twenty eighteen", "two thousand eighteen",
              "eighty one milligrams", "ninety nine percent", "five hundred milligrams", "twelve . five milligrams", "a one c of eight",
              "thirty five percent", "two or three times a week", "six hundred", "nineteen fifty seven"]:
        print(t, "->", spans(t))
