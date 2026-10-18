"""Training augmentations (TRAINING_PLAN.md §5): ICD-safe PII crossover and style transforms, labels kept in sync.

augment(record, gold, rng, pools) returns a new (record, gold, meta). Every transform also rewrites the silver
labels, so the reward and the SFT target always describe the augmented text.

  PII crossover (always)   names: consistent per-token map, sex-consistent, skipping common words and medical
                           terms; ages: same generalize.age_band, within +-4 years; dates: one per-encounter shift of
                           +-1-3 years plus up to 7 days; locations, organizations, contacts, IDs: swapped within
                           their category; occupations and OTHER_QUASI: swapped only if they are a short noun
                           phrase without an evidence link, and their reason becomes a neutral one.
                           Surrogates come from other encounters' silver labels (50%) or fake lists (50%).
                           Medical concepts, ambiguous terms and evidence spans are never touched: a PII occurrence
                           that overlaps one stays as it is.
  true-casing (0.5)        dialogue: sentence starts, "I", names and places capitalized
  de-spaced punct. (0.5)   dialogue: " , " -> ", ", " . " -> ". " (also ? ! ; :)
  speaker tags (0.5)       [doctor] -> Doctor: / Dr. <surrogate>: / D: / Clinician: / PHYSICIAN - / no tags
  field layout (0.3)       note first, dialogue only, or note only
  sub-window (0.3)         a window of >= 512 tokens of one field, with a carry-over list of identifiers from the
                           text before it (50% of the entries dropped)

Eval data is never augmented. The invariance tests are in rl/tests/test_augment.py.
"""
from __future__ import annotations

import datetime as dt
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import generalize as G  # noqa: E402
from common import norm  # noqa: E402

from render import find_all  # noqa: E402
from spec import NAME_TYPES  # noqa: E402

FIELDS = ("dialogue", "note")
TITLES = {"dr", "mr", "mrs", "ms", "miss", "doctor", "nurse", "md", "pa", "np", "jr", "sr", "dr.", "mr.", "mrs.", "ms."}
FEMALE_TITLES, MALE_TITLES = {"mrs", "ms", "miss"}, {"mr"}

FAKE_FIRST = {
    "female": ["abigail", "beatrice", "camila", "daphne", "elena", "fiona", "gloria", "helena", "irene", "jasmine",
               "katrina", "leona", "miriam", "nadia", "odette", "paula", "quinn", "rebecca", "sabrina", "tessa",
               "ursula", "vera", "wanda", "yolanda", "zoe", "alma", "bianca", "cora", "delia", "esther"],
    "male": ["adrian", "bernard", "cedric", "desmond", "elliot", "frederick", "gilbert", "hector", "ivan", "jerome",
             "kenneth", "leon", "malcolm", "nolan", "oscar", "preston", "quentin", "rodney", "stanley", "theodore",
             "ulrich", "vernon", "warren", "xavier", "yusuf", "zachary", "alvin", "boris", "clyde", "dominic"]}
FAKE_LAST = ["abernethy", "bellamy", "carrington", "dunleavy", "eastwood", "fairbanks", "galloway", "hollister",
             "iverson", "jablonski", "kensington", "lockhart", "mcallister", "northcott", "oyelaran", "pemberton",
             "quintero", "radcliffe", "stanhope", "thornbury", "underhill", "valdivia", "whitmore", "yarborough",
             "zielinski", "achebe", "brightwater", "castellanos", "drummond", "ellsworth"]
FAKE = {
    "LOCATION": ["sacramento", "albuquerque", "knoxville", "lake placid", "cape cod", "the ozarks", "vermont",
                 "oregon", "tucson", "savannah", "green bay", "yellowstone", "big sur", "chattanooga", "fort collins"],
    "ORGANIZATION": ["walgreens", "riverbend medical group", "costco", "the county library", "brightline logistics",
                     "st. anne's school", "harbor freight", "maple grove clinic", "united parcel", "the fire department",
                     "kroger", "northgate high school", "summit health", "the post office", "pinecrest academy"],
    "OCCUPATION": ["electrician", "accountant", "bus driver", "pharmacist", "teacher", "plumber", "barista",
                   "software engineer", "librarian", "mail carrier", "chef", "paralegal", "truck driver", "dentist",
                   "carpenter"],
    # noun phrases only: they replace a noun phrase in place ("peanut butter cups" -> "antique clocks")
    "OTHER_QUASI": ["antique clocks", "vintage motorcycles", "bonsai trees", "ham radio", "salsa dancing",
                    "model trains", "rare orchids", "competitive chess", "stained glass", "beekeeping"],
}
# OTHER_QUASI and OCCUPATION strings are swapped only when they look like a short noun phrase: a clause such as
# "i'm currently studying psychology" or "rides his bike 30 miles" cannot be replaced without breaking the sentence
QUASI_MAX_WORDS = 4
_CLAUSE_WORDS = {"i", "i'm", "im", "my", "me", "he", "she", "his", "her", "him", "we", "our", "they", "their", "you",
                 "your", "is", "was", "are", "were", "am", "be", "been", "has", "have", "had", "do", "does", "did",
                 "at", "in", "on", "for", "to", "from", "with", "about", "since"}
NEUTRAL_REASON = {"OTHER_QUASI": "rare personal detail", "OCCUPATION": "person's job"}


def noun_phrase(t: str) -> bool:
    """A short noun phrase without numbers or ages ("peanut butter cups", not "a mile long walks", "three-year-old")."""
    words = re.findall(r"[a-z0-9']+", t.lower())
    if not 0 < len(words) <= QUASI_MAX_WORDS or any(w in _CLAUSE_WORDS or w.isdigit() for w in words):
        return False
    return not re.search(G.NUM + r"|\b(?:year|month|week|old|mile|miles)\b", t, re.IGNORECASE)
# names that are ordinary words or medical terms: never mapped, never used as surrogates
COMMON = {"may", "will", "bill", "grace", "hope", "joy", "faith", "rose", "april", "june", "august", "mark", "frank",
          "art", "bob", "sue", "pat", "sunny", "summer", "dawn", "max", "jack", "chase", "hunter", "lane", "wood",
          "brown", "white", "black", "green", "king", "young", "long", "rich", "bell", "head", "hand", "foot",
          "walker", "baker", "cook", "price", "love", "may's", "tylenol", "dax", "dragon", "doctor", "patient"}

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
          "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
        9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
        16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth", 30: "thirtieth"}


def num_words(n: int, sep: str = " ") -> str:
    if n < 20:
        return _UNITS[n]
    t, u = divmod(n, 10)
    return _TENS[t] + (sep + _UNITS[u] if u else "")


def ord_words(n: int) -> str:
    if n in _ORD:
        return _ORD[n]
    t, u = divmod(n, 10)
    return f"{_TENS[t]} {_ORD[u]}"


def ord_suffix(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def spoken_year(y: int, style: str) -> str:
    if style == "thousand" or 2000 <= y <= 2009:
        return "two thousand" + (f" {num_words(y - 2000)}" if y > 2000 else "")
    hi, lo = divmod(y, 100)
    return f"{num_words(hi)} {('oh ' + _UNITS[lo]) if lo < 10 else num_words(lo)}"


def match_case(src: str, out: str) -> str:
    if src.isupper() and len(src) > 1:
        return out.upper()
    if src[:1].isupper():
        return " ".join(w[:1].upper() + w[1:] for w in out.split(" ")) if all(
            w[:1].isupper() for w in src.split() if w[:1].isalpha()) else out[:1].upper() + out[1:]
    return out


def patient_sex(record: dict) -> str | None:
    t = " " + norm((record.get("note") or "") + " " + (record.get("dialogue") or "")) + " "
    f = sum(t.count(w) for w in (" she ", " her ", " hers ", " female ", " woman ", " mrs ", " ms "))
    m = sum(t.count(w) for w in (" he ", " his ", " him ", " male ", " man ", " mr "))
    return None if f == m else "female" if f > m else "male"


# ---------------------------------------------------------------- surrogate pools

def build_pools(records: dict, golds: dict) -> dict:
    """Surrogates from the silver labels of all train encounters: first names by sex, surnames, and the strings of
    each swappable category."""
    pools = {"first_female": set(), "first_male": set(), "last": set(), **{c: set() for c in FAKE}}
    for eid, g in golds.items():
        rec = records.get(eid)
        if rec is None:
            continue
        sex = patient_sex(rec)
        for e in g.get("pii", []):
            t, cat = e["text"].lower(), e["category"]
            if cat in NAME_TYPES:
                toks = [w for w in re.findall(r"[a-z][a-z'-]+", t) if w.rstrip(".") not in TITLES]
                if cat == "PATIENT_NAME" and sex and len(toks) >= 2:
                    pools[f"first_{sex}"].add(toks[0])
                if len(toks) >= 2:
                    pools["last"].add(toks[-1])
            elif cat in ("LOCATION", "ORGANIZATION") and len(t.split()) <= 4:
                pools[cat].add(t)
            elif cat in ("OCCUPATION", "OTHER_QUASI") and noun_phrase(t):
                pools[cat].add(t)
    return {k: sorted(w for w in v if w not in COMMON) for k, v in pools.items()}


# ---------------------------------------------------------------- crossover

class Crossover:
    """Atom map of one augmented encounter: lowercase source string -> surrogate."""

    def __init__(self, record: dict, gold: dict, rng: random.Random, pools: dict):
        self.rec, self.gold, self.rng, self.pools = record, gold, rng, pools
        self.sex = patient_sex(record)
        self.years = rng.choice([-3, -2, -1, 1, 2, 3])
        self.days = rng.randint(-7, 7)
        all_text = norm(" ".join(record.get(f) or "" for f in FIELDS))
        self.taken = set(all_text.split())
        self.medical_words = {w for c in gold.get("medical_concepts", []) for w in norm(c["text"]).split()}
        self.medical_words |= {w for t in gold.get("ambiguous_terms", []) for w in norm(t["text"]).split()}
        self.atoms: dict[str, str] = {}
        self.protected = self._protected()
        self.evidence_ivs = self._evidence_ivs()
        self._build()

    # occurrences of medical text that crossover must not touch
    def _protected(self) -> dict:
        ivs = defaultdict(list)
        for key in ("medical_concepts", "ambiguous_terms"):
            for c in self.gold.get(key, []):
                if key == "ambiguous_terms" and c.get("is_pii"):
                    continue
                if c.get("field") in FIELDS:
                    ivs[c["field"]] += find_all(c["text"], self.rec[c["field"]])
        for ev in self.gold.get("evidence", []) or []:
            for s in ev.get("spans", []):
                if s.get("field") in FIELDS:
                    ivs[s["field"]] += find_all(s["text"], self.rec[s["field"]])
        return ivs

    def _evidence_ivs(self) -> dict:
        ivs = defaultdict(list)
        for c in self.gold.get("medical_concepts", []):
            if c.get("icd_relevant") and c.get("field") in FIELDS:
                ivs[c["field"]] += find_all(c["text"], self.rec[c["field"]])
        for ev in self.gold.get("evidence", []) or []:
            for s in ev.get("spans", []):
                if s.get("field") in FIELDS:
                    ivs[s["field"]] += find_all(s["text"], self.rec[s["field"]])
        return ivs

    def _fresh(self, cands: list[str]) -> str | None:
        cands = [c for c in cands if c not in self.taken and c not in COMMON
                 and not set(norm(c).split()) & self.medical_words]
        if not cands:
            return None
        c = self.rng.choice(cands)
        self.taken |= set(norm(c).split())
        return c

    def _pick(self, pool_key: str, fake: list[str], like: str | None = None) -> str | None:
        src = self.pools.get(pool_key) or []
        if like is not None:   # same leading sound, so "an electrician" -> "an accountant", never "an teacher"
            vowel = like[:1].lower() in "aeiou"
            src = [c for c in src if (c[:1] in "aeiou") == vowel and c != like.lower()]
            fake = [c for c in fake if (c[:1] in "aeiou") == vowel]
        first = src if src and self.rng.random() < 0.5 else fake
        return self._fresh(first) or self._fresh(fake if first is src else src)

    def _build(self):
        names = [e for e in self.gold.get("pii", []) if e["category"] in NAME_TYPES]
        role = {}   # token -> "first" | "last"
        sex_of = {}
        for e in names:
            words = re.findall(r"[A-Za-z][A-Za-z'-]+\.?", e["text"])
            toks = [w.lower().rstrip(".") for w in words]
            title = next((t for t in toks if t in TITLES), None)
            core = [t for t in toks if t not in TITLES]
            for k, t in enumerate(core):
                if len(t) < 3 or t in COMMON or t in self.medical_words:
                    continue
                if title in ("dr", "mr", "mrs", "ms", "miss", "doctor", "nurse") or (len(core) >= 2 and k == len(core) - 1):
                    role.setdefault(t, "last")
                elif len(core) >= 2 and k == 0:
                    role[t] = "first"
                else:
                    role.setdefault(t, "first")
                if e["category"] == "PATIENT_NAME" and self.sex:
                    sex_of.setdefault(t, self.sex)
                elif title in FEMALE_TITLES:
                    sex_of.setdefault(t, "female")
                elif title in MALE_TITLES:
                    sex_of.setdefault(t, "male")
        for t, r in role.items():
            if r == "last":
                s = self._pick("last", FAKE_LAST)
            else:
                sex = sex_of.get(t) or self.rng.choice(["female", "male"])
                s = self._pick(f"first_{sex}", FAKE_FIRST[sex])
            if s:
                self.atoms[t] = s
        for e in self.gold.get("pii", []):
            t, cat = e["text"], e["category"]
            key = t.lower()
            if key in self.atoms or cat in NAME_TYPES:
                continue
            new = None
            if cat == "AGE":
                new = self.shift_age(t)
            elif cat == "DATE":
                new = self.shift_date(t)
            elif cat in ("CONTACT", "ID"):
                new = self.scramble(t)
            elif cat in ("LOCATION", "ORGANIZATION"):
                new = self._pick(cat, FAKE[cat])
            elif cat in ("OCCUPATION", "OTHER_QUASI") and noun_phrase(t) and not self._linked(e):
                new = self._pick(cat, FAKE[cat], like=t)
            if new and new.lower() != key:
                self.atoms[key] = new
        # a string that contains another PII string is not swapped whole: its parts are mapped one by one, so the
        # inner string keeps its mention count ("9-year-old daughter" -> "7-year-old daughter")
        texts = {e["text"].lower() for e in self.gold.get("pii", [])} | set(role)
        for key in [k for k in self.atoms if k not in role]:
            if any(t != key and find_all(t, key) for t in texts):
                del self.atoms[key]

    def _linked(self, e: dict) -> bool:
        f = e["field"]
        if f not in FIELDS:
            return True
        return any(s < pe and ps < en for s, en in find_all(e["text"], self.rec[f]) for ps, pe in self.evidence_ivs[f])

    # ---- ages
    def shift_age(self, text: str) -> str | None:
        hits = G._age_hits(text)
        m = None
        for kind, pat in G.AGE_PATTERNS:
            m = pat.search(text)
            if m:
                break
        if not hits or m is None or not m.group("n1"):
            mm = re.search(G.NUM, text, re.IGNORECASE)
            if not mm or not re.fullmatch(r"\s*(?:about|around|almost|nearly)?\s*" + G.NUM + r"(?:\s+now)?\s*", text,
                                          re.IGNORECASE):
                return None
            span, val = (mm.start(), mm.end()), G.parse_num(mm.group(0))
            unit_months = 12.0
        else:
            span, val = m.span("n1"), G.parse_num(m.group("n1"))
            unit = (m.groupdict().get("unit") or "year").lower().rstrip("s")
            unit_months = G.MONTHS_PER.get(unit, 12.0)
        if val is None or val <= 0:
            return None
        band = G.age_band(val * unit_months)
        opts = [v for v in range(max(1, val - 4), val + 5) if v != val and G.age_band(v * unit_months) == band]
        if not opts:
            return None
        v = self.rng.choice(opts)
        src = text[span[0]:span[1]]
        rep = str(v) if src.strip().isdigit() else num_words(v, "-" if "-" in src else " ")
        return text[:span[0]] + match_case(src, rep) + text[span[1]:]

    # ---- dates
    _NUMDATE = re.compile(r"(?<![\w/.])(\d{1,2})([/-])(\d{1,2})\2(\d{4}|\d{2})(?![\w/])")
    _MONYEAR = re.compile(r"(?<![\w/.])(\d{1,2})/(\d{4})(?![\w/])")
    _MONTH = re.compile(rf"\b({'|'.join(G.MONTHS)})\b(?:([ \t]+(?:the[ \t]+)?)({G._DAY_ORD}|\d{{1,2}})\b)?", re.IGNORECASE)
    _YEAR = re.compile(rf"(?<![\d.,/$])\b({G._YEAR4})\b(?![\d%])|\b({G._SPOKEN_YEAR})", re.IGNORECASE)

    def _shift(self, y: int, m: int, d: int) -> dt.date:
        try:
            base = dt.date(y + self.years, m, min(d, 28) if m == 2 else d)
        except ValueError:
            base = dt.date(y + self.years, m, 28)
        return base + dt.timedelta(days=self.days)

    def shift_date(self, text: str) -> str | None:
        out, changed = text, False

        def numdate(mm):
            nonlocal changed
            mo, d, ys = int(mm.group(1)), int(mm.group(3)), mm.group(4)
            if not (1 <= mo <= 12 and 1 <= d <= 31):
                return mm.group(0)
            y = int(ys) if len(ys) == 4 else (2000 + int(ys) if int(ys) <= 30 else 1900 + int(ys))
            n = self._shift(y, mo, d)
            changed = True
            w = len(mm.group(1))
            ny = str(n.year) if len(ys) == 4 else f"{n.year % 100:02d}"
            return f"{n.month:0{w}d}{mm.group(2)}{n.day:0{len(mm.group(3))}d}{mm.group(2)}{ny}"

        out = self._NUMDATE.sub(numdate, out)

        def monyear(mm):
            nonlocal changed
            changed = True
            return f"{mm.group(1)}/{int(mm.group(2)) + self.years}"

        out = self._MONYEAR.sub(monyear, out)

        def month(mm):
            nonlocal changed
            mo = G.MONTHS.index(mm.group(1).lower()) + 1
            if not mm.group(3):
                return mm.group(0)
            ds = mm.group(3)
            d = int(re.match(r"\d+", ds).group()) if ds[:1].isdigit() else _ord_value(ds)
            if not d or not 1 <= d <= 31:
                return mm.group(0)
            n = self._shift(2001, mo, d)
            changed = True
            day = (ord_suffix(n.day) if re.fullmatch(r"\d+(?:st|nd|rd|th)", ds, re.IGNORECASE) else
                   str(n.day) if ds.isdigit() else ord_words(n.day))
            return match_case(mm.group(1), G.MONTHS[n.month - 1]) + mm.group(2) + day

        out = self._MONTH.sub(month, out)

        def year(mm):
            nonlocal changed
            changed = True
            if mm.group(1):
                return str(int(mm.group(1)) + self.years)
            y = G.spoken_year(mm.group(2))
            if y is None:
                return mm.group(0)
            style = "thousand" if mm.group(2).lower().startswith("two thousand") else "pair"
            return match_case(mm.group(2), spoken_year(y + self.years, style))

        # years not already shifted as part of a numeric date
        if not self._NUMDATE.search(text) and not self._MONYEAR.search(text):
            out = self._YEAR.sub(year, out)
        return out if changed and out != text else None

    # ---- contacts and IDs
    def scramble(self, text: str) -> str:
        return "".join(str(self.rng.randint(0, 9)) if c.isdigit() else
                       (chr(self.rng.randint(97, 122)) if c.islower() else chr(self.rng.randint(65, 90)))
                       if c.isalpha() and len(text) > 4 and any(ch.isdigit() for ch in text) else c for c in text)

    # ---- application
    def pattern(self):
        if not self.atoms:
            return None
        keys = sorted(self.atoms, key=len, reverse=True)
        return re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE)

    def map_string(self, s: str) -> str:
        pat = self.pattern()
        return pat.sub(lambda m: match_case(m.group(1), self.atoms[m.group(1).lower()]), s) if pat else s

    def apply(self) -> tuple[dict, dict]:
        pat = self.pattern()
        new_rec = dict(self.rec)
        for f in FIELDS:
            text = self.rec.get(f) or ""
            ivs = []
            for e in self.gold.get("pii", []):
                if e["field"] != f:
                    continue
                for iv in find_all(e["text"], text):
                    if not any(iv[0] < pe and ps < iv[1] for ps, pe in self.protected[f]):
                        ivs.append(iv)
            merged = []
            for s, e in sorted(ivs):
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            if pat is None or not merged:
                continue
            pieces, last = [], 0
            for s, e in merged:
                pieces += [text[last:s], self.map_string(text[s:e])]
                last = e
            pieces.append(text[last:])
            new_rec[f] = "".join(pieces)
        new_gold = dict(self.gold)
        seen, pii = set(), []
        for e in self.gold.get("pii", []):
            t = self.map_string(e["text"])
            if (e["field"], t.lower()) in seen:
                continue
            seen.add((e["field"], t.lower()))
            row = {**e, "text": t, "orig_text": e["text"]}
            if t.lower() != e["text"].lower() and e["category"] in NEUTRAL_REASON:
                row["comment"] = NEUTRAL_REASON[e["category"]]   # the annotator's comment described the old string
            pii.append(row)
        new_gold["pii"] = pii
        return new_rec, new_gold


def _ord_value(s: str) -> int | None:
    s = s.lower().replace("-", " ").strip()
    for n in range(1, 32):
        if ord_words(n) == s:
            return n
    return None


# ---------------------------------------------------------------- style transforms (dialogue)

def true_case(text: str, names: list[str]) -> str:
    def line(l):
        m = re.match(r"(\[[^\]]+\]\s*)?(.*)", l)
        tag, body = m.group(1) or "", m.group(2)
        body = re.sub(r"\bi\b", "I", body)
        body = re.sub(r"(^|[.?!][ \t]+)([a-z])", lambda mm: mm.group(1) + mm.group(2).upper(), body)
        return tag + body
    out = "\n".join(line(l) for l in text.split("\n"))
    for n in sorted(set(names), key=len, reverse=True):
        out = re.sub(r"(?<!\w)" + re.escape(n) + r"(?!\w)",
                     lambda m: " ".join(w[:1].upper() + w[1:] for w in m.group(0).split(" ")), out,
                     flags=re.IGNORECASE)
    return out


def despace(text: str) -> str:
    return re.sub(r" ([,.?!;:])(?= |\n|$)", r"\1", text)


SPEAKER_STYLES = ["Doctor:", "Dr.", "D:", "Clinician:", "PHYSICIAN -", ""]
PATIENT_STYLE = {"Doctor:": "Patient:", "Dr.": "Patient:", "D:": "P:", "Clinician:": "Patient:",
                 "PHYSICIAN -": "PATIENT -", "": ""}
GUEST_STYLE = {"Doctor:": "Guest:", "Dr.": "Family:", "D:": "G:", "Clinician:": "Guest:", "PHYSICIAN -": "FAMILY -",
               "": ""}


def restyle_tags(text: str, style: str, doctor_name: str | None) -> str:
    doc = f"Dr. {doctor_name.title()}:" if style == "Dr." and doctor_name else style if style != "Dr." else "Doctor:"
    rep = {"doctor": doc, "patient": PATIENT_STYLE[style], "patient_guest": GUEST_STYLE[style]}

    def sub(m):
        new = rep.get(m.group(1).strip().lower())
        if new is None:
            return m.group(0)
        return (new + " ") if new else ""
    return re.sub(r"^\[([a-z_ ]+)\]\s?", sub, text, flags=re.MULTILINE)


# ---------------------------------------------------------------- the augmentation

def _map_dialogue_gold(gold: dict, fn) -> dict:
    out = dict(gold)
    for key in ("pii", "medical_concepts", "ambiguous_terms"):
        out[key] = [{**e, "text": fn(e["text"])} if e.get("field") == "dialogue" else e for e in gold.get(key, [])]
    out["evidence"] = [{**ev, "spans": [{**s, "text": fn(s["text"])} if s.get("field") == "dialogue" else s
                                        for s in ev.get("spans", [])]} for ev in gold.get("evidence", []) or []]
    return out


def augment(record: dict, gold: dict, rng: random.Random, pools: dict, *, p_case=0.5, p_punct=0.5, p_tags=0.5,
            p_layout=0.3, p_window=0.3, crossover=True, window_tokens: int = 512,
            window_max_tokens: int | None = None) -> tuple[dict, dict, dict]:
    meta = {"sections": ["dialogue", "note"], "known": [], "ops": []}
    rec, g = dict(record), dict(gold)
    if crossover:
        cx = Crossover(rec, g, rng, pools)
        rec, g = cx.apply()
        meta["ops"].append("crossover")
    if rng.random() < p_case:
        names = [e["text"] for e in g["pii"] if e["field"] == "dialogue" and e["category"] in NAME_TYPES | {"LOCATION"}]
        rec["dialogue"] = true_case(rec["dialogue"], names)
        meta["ops"].append("truecase")
    if rng.random() < p_punct:
        rec["dialogue"] = despace(rec["dialogue"])
        g = _map_dialogue_gold(g, despace)
        meta["ops"].append("despace")
    if rng.random() < p_tags and re.search(r"^\[(?:doctor|patient)\]", rec["dialogue"], re.MULTILINE):
        style = rng.choice(SPEAKER_STYLES)
        doctors = [e["text"] for e in g["pii"] if e["category"] == "CLINICIAN_NAME"]
        last = None
        if style == "Dr.":
            toks = [w for d in doctors for w in re.findall(r"[a-z][a-z'-]+", d.lower()) if w not in TITLES]
            last = toks[-1] if toks else rng.choice(FAKE_LAST)
        rec["dialogue"] = restyle_tags(rec["dialogue"], style, last)
        if last:
            tag = f"Dr. {last.title()}"
            if not any(e["field"] == "dialogue" and e["text"].lower() == tag.lower() for e in g["pii"]):
                g["pii"] = g["pii"] + [{"text": tag, "field": "dialogue", "category": "CLINICIAN_NAME", "tier": "direct",
                                        "comment": "speaker tag"}]
        meta["ops"].append(f"tags:{style or 'none'}")
    if rng.random() < p_layout:
        meta["sections"] = rng.choice([["note", "dialogue"], ["dialogue"], ["note"]])
        meta["ops"].append("layout:" + "+".join(meta["sections"]))
    if rng.random() < p_window:
        rec, g, meta = window(rec, g, meta, rng, window_tokens, window_max_tokens)
    for f in FIELDS:
        if f not in meta["sections"]:
            rec[f] = ""
    g = restrict_gold(rec, g)
    return rec, g, meta


def window(rec: dict, g: dict, meta: dict, rng: random.Random, min_tokens: int,
           max_tokens: int | None = None) -> tuple[dict, dict, dict]:
    """A random line window of one field (min_tokens..max_tokens, ~4 characters per token) and its carry-over list."""
    f = rng.choice([s for s in meta["sections"] if (rec.get(s) or "").strip()] or ["dialogue"])
    lines = (rec.get(f) or "").split("\n")
    total = len(rec.get(f) or "")
    if total <= 4 * min_tokens or len(lines) < 4:
        return rec, g, meta
    size = rng.randint(4 * min_tokens, min(total, 4 * max_tokens) if max_tokens else total)
    starts = [0]
    for l in lines[:-1]:
        starts.append(starts[-1] + len(l) + 1)
    cands = [i for i, s in enumerate(starts) if s + size <= total] or [0]
    i0 = rng.choice(cands)
    j = i0
    while j < len(lines) and starts[j] - starts[i0] < size:
        j += 1
    before = "\n".join(lines[:i0])
    win = "\n".join(lines[i0:j])
    known = []
    for e in g["pii"]:
        if e["field"] == f and find_all(e["text"], before) and rng.random() >= 0.5:
            if not any(k["text"].lower() == e["text"].lower() for k in known):
                known.append({"text": e["text"], "type": e["category"]})
    rec = {**rec, f: win}
    meta = {**meta, "sections": [f], "known": known[:64], "ops": meta["ops"] + [f"window:{f}"]}
    return rec, g, meta


def restrict_gold(rec: dict, g: dict) -> dict:
    """Drop labels whose text no longer occurs in their field (dropped field, window)."""
    def keep(e):
        return e.get("field") in FIELDS and bool(find_all(e["text"], rec.get(e["field"]) or ""))
    out = {**g}
    for key in ("pii", "medical_concepts", "ambiguous_terms"):
        out[key] = [e for e in g.get(key, []) if keep(e)]
    out["evidence"] = [{**ev, "spans": [s for s in ev.get("spans", []) if keep(s)]} for ev in g.get("evidence", []) or []]
    return out
