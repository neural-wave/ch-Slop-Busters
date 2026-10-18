"""Corpus drift of every suite from the ACI train corpus: lexical, stylistic and structural distance of its text.

Usage: python3 -I eval drift [--suites NAME ...]

The reference is the 140 ACI-Bench train encounters (data/converted/aci_bench, labels `meta.split == "train"`), the
corpus every team developed on; a suite is the dialogue and note of its encounters.jsonl. The metrics follow
Cornelius & Rinaldi 2026 (definitions pp. 9-10): JSD of unigram and bigram distributions, vocabulary Jaccard, and
the absolute difference |suite - reference| of 17 per-corpus statistics (entropy, MTLD, Yule's K, Distinct-1/2, 4-gram
self-repetition, sentence length, token-class ratios, Flesch-Kincaid, line breaks, key-value, bullet, all-caps and
special-character ratios). The model-based metrics of the paper (GPT-2 perplexity, and the embedding metrics MMD,
Frechet, embedding-mean distance, cluster JSD) need a neural model and are not computed. docs/CORPUS-DRIFT.md defines
every metric.

Writes data/eval/drift.json (deterministic):

    reference        {"name": "aci_train", "records", "tokens", "values": the reference's own statistics,
                      "code_sha1": sha1 of this file}
    suites           {<suite>: {"records", "input_sha1" (of its encounters.jsonl), "tokens", "types", "jsd_unigram",
                      "jsd_bigram", "vocab_jaccard", "<stat>_diff" (|suite - reference|) and "<stat>" (the suite's own
                      value) per statistic}}
    complexity_rank  {<suite>: 1 = closest to ACI train by jsd_unigram (ties by name)}

drift.json always covers every suite in data/eval/suites/. `--suites` recomputes the named suites (default: all); the
others keep their previous entry while the reference, this code and their encounters.jsonl (input_sha1) are unchanged,
and are recomputed otherwise. JSD and entropy are in bits (JSD in [0, 1]).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path
from statistics import fmean, pstdev

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import CONVERTED, FIELDS, OUT, SUITES_DIR, WORD, load_jsonl, sha1_file, write_json

DRIFT_FILE = OUT / "drift.json"
CODE_SHA1 = sha1_file(Path(__file__))  # a kept entry must come from the same metric code
REFERENCE = "aci_train"
EPS = 1e-9  # JSD pseudo-count added to every n-gram of the union vocabulary
MTLD_TTR = 0.72  # McCarthy & Jarvis 2010
DIGITS = 6

STATS = ("entropy", "mtld", "yule_k", "distinct_1", "distinct_2", "self_repetition_4", "sentence_length_mean", "sentence_length_std",
         "punct_ratio", "digit_ratio", "caps_ratio", "flesch_kincaid", "line_breaks_per_100", "key_value_ratio", "bullet_ratio",
         "allcaps_ratio", "special_char_ratio")  # fmt: skip

LINE_BREAK = re.compile(r"\r\n|\r|\n|\\r\\n|\\n")  # a literal "\n" escape left in a converted text (pmc_patients) is a line break too
BOUNDARY = re.compile(r"[.!?]+[\"'’”)\]]*\s+")
ABBREVIATIONS = frozenset("dr mr mrs ms mx prof sr jr st vs approx e.g i.e cf al fig".split())
ORDINAL = re.compile(r"\(?\d{1,3}[.)]|\(?[a-z]\)", re.IGNORECASE)
BULLETS, GLUED_BULLETS = frozenset("-*+–—•·▪‣◦○●■□►▸▹➢➤✓✔"), "•▪‣◦○●■□►▸▹➢➤✓✔"
KEY = r"[^\W\d_][\w'’()/&.-]*"
KEY_VALUE = re.compile(rf"\W{{0,3}}{KEY}(?:[ \t]+{KEY}){{0,3}}[ \t]*[:=]")  # `Name: John`, `A=B`: a key of 1-4 words, then : or =
SPECIAL = "@#$%^&*_~|\\<>{}=+`"
PUNCT = "".join(c for c in map(chr, range(0x10000)) if unicodedata.category(c).startswith("P"))  # Unicode P* (re has no \p{P})
# one match per whitespace token that contains such a character
PUNCT_TOKEN, SPECIAL_TOKEN = (re.compile(rf"(?<!\S)[^\s{re.escape(cs)}]*[{re.escape(cs)}]") for cs in (PUNCT, SPECIAL))
DIGIT_TOKEN = re.compile(r"(?<!\S)[^\s\d]*\d")
FIRST_ALNUM = re.compile(r"(?<!\S)[^\w\s]*([^\W_])")  # the first letter or digit of each whitespace token
VOWELS = re.compile(r"[aeiouyàáâãäåæèéêëìíîïòóôõöøœùúûüýÿ]+")


def sentences(line: str) -> list[str]:
    """Sentences of one line: split after . ! ? and whitespace, except after an abbreviation (`Dr.`), an initial
    (`J.`) or a leading ordinal (`1. Hypertension`)."""
    out, start = [], 0
    for m in BOUNDARY.finditer(line):
        head = line[start : m.start()].split()
        last = head[-1].lower() if head else ""
        abbreviation = last in ABBREVIATIONS or (len(last) == 1 and last.isalpha())
        if line[m.start()] == "." and (abbreviation or (len(head) == 1 and last.isdigit() and len(last) <= 3)):
            continue
        out.append(line[start : m.end()].strip())
        start = m.end()
    out.append(line[start:].strip())
    return [s for s in out if s]


@cache
def syllables(word: str) -> int:
    """Syllable heuristic: vowel groups (y included), minus a silent final e: one after a consonant, not in consonant +
    `le`, in a word with another vowel group (`make` 1, but `table` 2, `agree` 2, `the` 1); at least 1 (a number 1)."""
    w = word.lower()
    n = len(VOWELS.findall(w))
    if n > 1 and w.endswith("e") and not VOWELS.match(w[-2]) and not (w.endswith("le") and len(w) > 2 and not VOWELS.match(w[-3])):
        n -= 1
    return max(1, n)


def flesch_kincaid(n_words: int, n_sentences: int, n_syllables: int) -> float | None:
    """Flesch-Kincaid grade level: 0.39 words/sentence + 11.8 syllables/word - 15.59."""
    return 0.39 * n_words / n_sentences + 11.8 * n_syllables / n_words - 15.59 if n_words and n_sentences else None


def mtld(tokens: Sequence[str], threshold: float = MTLD_TTR) -> float | None:
    """MTLD (McCarthy & Jarvis 2010): mean of the forward and backward pass. A pass counts factors, stretches whose
    type-token ratio falls to `threshold`, plus a partial factor (1 - TTR) / (1 - threshold) for the rest, and gives
    tokens / factors. A text whose TTR never falls is one factor (MTLD = its length)."""
    if not tokens:
        return None

    def one_pass(seq: Iterable[str]) -> float:
        factors, types, n, ttr = 0.0, set(), 0, 1.0
        for t in seq:
            types.add(t)
            n += 1
            ttr = len(types) / n
            if ttr <= threshold:
                factors, types, n, ttr = factors + 1, set(), 0, 1.0
        factors += (1 - ttr) / (1 - threshold)
        return len(tokens) / factors if factors else float(len(tokens))

    return (one_pass(tokens) + one_pass(reversed(tokens))) / 2


def yule_k(counts: Counter) -> float | None:
    """Yule's K = 10^4 (sum of squared type frequencies - N) / N^2; 0 when every token is distinct."""
    n = sum(counts.values())
    return 1e4 * (sum(c * c for c in counts.values()) - n) / (n * n) if n else None


def entropy(counts: Counter) -> float | None:
    """Shannon entropy of the token distribution, in bits."""
    n = sum(counts.values())
    return -math.fsum(c / n * math.log2(c / n) for c in counts.values()) if n else None


def distinct(counts: Counter) -> float | None:
    """Distinct-n: unique n-grams / all n-grams."""
    n = sum(counts.values())
    return len(counts) / n if n else None


def self_repetition(counts: Counter) -> float | None:
    """Share of n-gram occurrences whose n-gram occurs more than once in the corpus."""
    n = sum(counts.values())
    return sum(c for c in counts.values() if c > 1) / n if n else None


def jsd(p: Counter, q: Counter, eps: float = EPS) -> float | None:
    """Jensen-Shannon divergence in bits of two count distributions, eps-smoothed over the union vocabulary: 0 for
    identical distributions, 1 (up to the smoothing) for disjoint ones."""
    if not p or not q:
        return None
    vocab = p.keys() | q.keys()
    np_, nq = sum(p.values()) + eps * len(vocab), sum(q.values()) + eps * len(vocab)
    terms = []
    for w in vocab:
        a, b = (p.get(w, 0) + eps) / np_, (q.get(w, 0) + eps) / nq
        m = (a + b) / 2
        terms += [a * math.log2(a / m), b * math.log2(b / m)]
    return max(0.0, math.fsum(terms) / 2)  # fsum: exact, so the result does not depend on the set's iteration order


def jaccard(a: Iterable, b: Iterable) -> float | None:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def profile(texts: Iterable[str]) -> dict:
    """n-gram counts and per-corpus statistics of a list of texts. Word tokens are case-folded (as common.words) and
    n-grams never cross texts; the token-class ratios use whitespace tokens, the structural ones sentences."""
    uni, bi, four = Counter(), Counter(), Counter()
    stream: list[str] = []
    sent_len: list[int] = []
    n = Counter()
    for text in texts:
        toks: list[str] = []
        text, breaks = LINE_BREAK.subn("\n", text)
        surface = text.split()
        n["breaks"] += breaks
        n["surface"] += len(surface)
        n["punct"] += len(PUNCT_TOKEN.findall(text))
        n["digit"] += len(DIGIT_TOKEN.findall(text))
        n["special"] += len(SPECIAL_TOKEN.findall(text))
        n["caps"] += sum(c.isupper() for c in FIRST_ALNUM.findall(text))
        n["allcaps"] += sum(1 for t in surface if t.isupper() and sum(map(str.isalpha, t)) >= 2)
        for line in text.split("\n"):
            for s in sentences(line):
                w = [x.lower() for x in WORD.findall(s)]
                if not w:
                    continue
                first = s.split()[0]
                sent_len.append(len(w))
                toks += w
                n["bullet"] += first in BULLETS or first[0] in GLUED_BULLETS or bool(ORDINAL.fullmatch(first))
                n["key_value"] += bool(KEY_VALUE.match(s))
        uni.update(toks)
        bi.update(zip(toks, toks[1:]))
        four.update(zip(toks, toks[1:], toks[2:], toks[3:]))
        stream += toks
    values = {
        "entropy": entropy(uni),
        "mtld": mtld(stream),
        "yule_k": yule_k(uni),
        "distinct_1": distinct(uni),
        "distinct_2": distinct(bi),
        "self_repetition_4": self_repetition(four),
        "sentence_length_mean": fmean(sent_len) if sent_len else None,
        "sentence_length_std": pstdev(sent_len) if sent_len else None,
        "punct_ratio": _ratio(n["punct"], n["surface"]),
        "digit_ratio": _ratio(n["digit"], n["surface"]),
        "caps_ratio": _ratio(n["caps"], n["surface"]),
        "flesch_kincaid": flesch_kincaid(len(stream), len(sent_len), sum(map(syllables, stream))),
        "line_breaks_per_100": _ratio(100 * n["breaks"], n["surface"]),
        "key_value_ratio": _ratio(n["key_value"], len(sent_len)),
        "bullet_ratio": _ratio(n["bullet"], len(sent_len)),
        "allcaps_ratio": _ratio(n["allcaps"], n["surface"]),
        "special_char_ratio": _ratio(n["special"], n["surface"]),
    }
    return {"tokens": len(stream), "types": len(uni), "unigrams": uni, "bigrams": bi, "values": values}


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, DIGITS) + 0.0  # + 0.0: no -0.0


def compare(ref: dict, cur: dict) -> dict:
    """Drift metrics of the profile `cur` from the reference profile `ref`."""
    out = {"tokens": cur["tokens"], "types": cur["types"], "jsd_unigram": jsd(ref["unigrams"], cur["unigrams"]),
           "jsd_bigram": jsd(ref["bigrams"], cur["bigrams"]), "vocab_jaccard": jaccard(ref["unigrams"], cur["unigrams"])}  # fmt: skip
    for k in STATS:
        a, b = ref["values"][k], cur["values"][k]
        out[f"{k}_diff"] = abs(b - a) if a is not None and b is not None else None
    out.update(cur["values"])
    return {k: v if isinstance(v, int) else _round(v) for k, v in out.items()}


def reference_texts() -> tuple[int, list[str]]:
    """(records, texts) of the ACI train split: dialogue and note of each encounter."""
    src = CONVERTED / "aci_bench"
    encs, labs = load_jsonl(src / "encounters.jsonl"), load_jsonl(src / "labels.jsonl")
    rows = [e for e, lab in zip(encs, labs, strict=True) if lab.get("meta", {}).get("split") == "train"]
    return len(rows), [e.get(f) or "" for e in rows for f in FIELDS]


def suite_texts(name: str) -> tuple[int, list[str]]:
    rows = load_jsonl(SUITES_DIR / name / "encounters.jsonl")
    return len(rows), [r.get(f) or "" for r in rows for f in FIELDS]


def present() -> list[str]:
    return sorted(p.parent.name for p in SUITES_DIR.glob("*/encounters.jsonl"))


def rank(suites: dict[str, dict]) -> dict[str, int]:
    """1 = smallest jsd_unigram (closest to the reference); ties and missing values by name, missing last."""
    order = sorted(suites, key=lambda s: (suites[s].get("jsd_unigram") is None, suites[s].get("jsd_unigram") or 0.0, s))
    return {s: i for i, s in enumerate(order, 1)}


def drift(selected: list[str]) -> dict:
    """The drift.json document: the selected suites recomputed, every other present suite kept from the previous file
    when the reference, the code and the suite's encounters.jsonl are the same, else recomputed."""
    n_ref, texts = reference_texts()
    ref = profile(texts)
    reference = {"name": REFERENCE, "records": n_ref, "tokens": ref["tokens"], "values": {k: _round(v) for k, v in ref["values"].items()},
                 "code_sha1": CODE_SHA1}  # fmt: skip
    try:
        old = json.loads(DRIFT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    kept = old.get("suites", {}) if isinstance(old, dict) and old.get("reference") == reference else {}
    suites = {}
    for s in present():
        digest = sha1_file(SUITES_DIR / s / "encounters.jsonl")
        if s in selected or (kept.get(s) or {}).get("input_sha1") != digest:
            n, texts = suite_texts(s)
            suites[s] = {"records": n, "input_sha1": digest, **compare(ref, profile(texts))}
        else:
            suites[s] = kept[s]
    ranks = rank(suites)
    return {"reference": reference, "suites": {s: suites[s] for s in sorted(suites, key=ranks.get)}, "complexity_rank": ranks}


# the printed columns (d_ = |suite - reference|)
COLUMNS = {"jsd_unigram": "jsd_uni", "jsd_bigram": "jsd_bi", "vocab_jaccard": "jaccard", "entropy_diff": "d_entropy", "mtld_diff": "d_mtld",
           "yule_k_diff": "d_yule_k", "flesch_kincaid_diff": "d_fk_grade", "sentence_length_mean_diff": "d_sent_len", "line_breaks_per_100_diff": "d_breaks",
           "key_value_ratio_diff": "d_key_val"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval drift", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--suites", "--only", dest="suites", nargs="+", metavar="SUITE", help="suites to recompute (default: every suite in data/eval/suites/)")
    a = p.parse_args(argv)
    built = present()
    for s in a.suites or []:
        if s not in built:
            print(f"warning: drift: suite {s} is not built (no data/eval/suites/{s}/encounters.jsonl), skipped", file=sys.stderr)
    if not built:
        print("drift: no suite in data/eval/suites/ (run `python3 -I eval suites` first)", file=sys.stderr)
        return 1
    try:
        doc = drift(a.suites or built)
    except FileNotFoundError as e:
        print(f"drift: reference corpus missing: {e.filename} (scripts/convert/convert_all.py)", file=sys.stderr)
        return 1
    write_json(DRIFT_FILE, doc)
    ref = doc["reference"]
    print(f"reference {ref['name']}: {ref['records']} records, {ref['tokens']} tokens -> {DRIFT_FILE}")
    print(f"{'suite':18s} {'rank':>4s} {'records':>7s} {'tokens':>8s} " + " ".join(f"{c:>10s}" for c in COLUMNS.values()) + "   (d_ = |suite - reference|)")
    for s, m in doc["suites"].items():
        cells = " ".join(f"{'-':>10s}" if m.get(c) is None else f"{m[c]:10.4f}" for c in COLUMNS)
        print(f"{s:18s} {doc['complexity_rank'][s]:4d} {m['records']:7d} {m['tokens']:8d} {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
