"""Per-decision reward for the de-identification policy (TRAINING_PLAN.md §4).

Every decision of the output gets its own reward, placed on the token that closes its JSON item; PPO's token-level
critic does the credit assignment. All quantities compare the rendered encounter (rl/render.py) with silver labels,
using the evaluator's matching (common.norm, common.count_mentions, evaluate.name_tokens):

  P  share of weighted PII mentions removed (direct w=2, quasi w=1; name tokens counted separately, w=2)
  U  share of weighted medical-concept mentions kept (icd_relevant w=2, else 1)

  entities[i] applied    w_P*dP + w_U*dU - lam_fp*(#removed words not covered by silver PII)
                         - 0.5 per gold code whose last surviving evidence it removes, + 0.02 if the type matches
  entities[i] rejected   -0.1 (span limits, missing ctx, not found)      duplicates: 0
  keep[j]                +0.02 for a silver non-PII ambiguous term or a correctly vetoed regex span;
                         -0.1 if it keeps silver PII
  "]" of entities        -w_miss*(1 - P)
  dx[k] (free mode)      content-word F1 of desc vs the best unmatched gold description + code credit (exact 1.0,
                         3-char 0.5, chapter 0.1), times 1/sqrt(train frequency of that code); -0.2 if the code is
                         not in ICD-10-CM; -0.3 per false positive. A prefilled dx is prompt, so it scores 0.
  evidence[k]            +0.05 per span verbatim in the text, matching a silver concept, surviving rendering
  invalid JSON           -1 on the last token, nothing applied

dP and dU are marginal: decisions are applied one by one in output order on top of the regex pre-pass, so the entity
rewards sum to S(final) - S(regex only) whatever the order (up to overlapping spans).
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
from common import DATA_DIR, count_mentions, norm, read_jsonl  # noqa: E402
from evaluate import NAME_CATEGORIES, name_tokens  # noqa: E402

import render as R  # noqa: E402
from spec import TIER, is_valid_code, norm_code, norm_type  # noqa: E402

FIELDS = ("dialogue", "note")


@dataclass
class RewardConfig:
    w_P: float = 1.0
    w_U: float = 1.0
    w_miss: float = 1.0
    lam_fp: float = 0.03
    kill_evidence: float = 0.5
    type_bonus: float = 0.02
    rejected: float = -0.1
    keep_ok: float = 0.02
    keep_pii: float = -0.1
    evidence_ok: float = 0.05
    dx_invalid_code: float = -0.2
    dx_false_positive: float = -0.3
    invalid_json: float = -1.0


# ---------------------------------------------------------------- silver labels of one record

@dataclass
class Silver:
    pii_rows: list            # (field, text, weight)
    token_rows: list          # (field, token, weight)
    concept_rows: list        # (field, text, weight)
    pii_ivs: dict             # field -> [(s, e)] occurrences of silver PII in the original
    concept_ivs: dict         # field -> [(s, e)] occurrences of silver concepts (protected without ctx)
    pii_type: dict            # norm(text) -> category
    nonpii_amb: set           # norm(text) of non-PII ambiguous terms
    concept_norms: set
    evidence: dict            # code -> [(field, text)]
    gold_codes: list          # [(normalized code, description)]
    orig_counts: list | None = None   # mentions of every row in the original (filled by scores)


def prepare(record: dict, gold: dict) -> Silver:
    orig_n = {f: norm(record.get(f) or "") for f in FIELDS}
    pii_rows, token_rows, seen_tok = [], [], set()
    pii_ivs, pii_type = {f: [] for f in FIELDS}, {}
    for e in gold.get("pii", []):
        f, t = e["field"], e["text"]
        if f not in FIELDS or not count_mentions(t, orig_n[f]):
            continue
        pii_rows.append((f, t, 2.0 if TIER.get(e["category"]) == "direct" else 1.0))
        pii_ivs[f] += R.find_all(t, record[f])
        pii_type.setdefault(norm(t), e["category"])
        if e["category"] in NAME_CATEGORIES:
            for tok in name_tokens(t):
                if (f, tok) not in seen_tok and count_mentions(tok, orig_n[f]):
                    seen_tok.add((f, tok))
                    token_rows.append((f, tok, 2.0))
    concept_rows, concept_ivs, concept_norms = [], {f: [] for f in FIELDS}, set()
    for c in gold.get("medical_concepts", []):
        f, t = c["field"], c["text"]
        if f not in FIELDS or not count_mentions(t, orig_n[f]):
            continue
        concept_rows.append((f, t, 2.0 if c.get("icd_relevant") else 1.0))
        concept_ivs[f] += R.find_all(t, record[f])
        concept_norms.add(norm(t))
    nonpii_amb = {norm(t["text"]) for t in gold.get("ambiguous_terms", []) if not t.get("is_pii")}
    for t in gold.get("ambiguous_terms", []):
        if not t.get("is_pii") and t.get("field") in FIELDS:
            concept_ivs[t["field"]] += R.find_all(t["text"], record[t["field"]])
    evidence = {norm_code(ev["code"]): [(s["field"], s["text"]) for s in ev.get("spans", [])
                                        if s.get("field") in FIELDS]
                for ev in gold.get("evidence", []) or []}
    gold_codes = [(norm_code(c["code"]), c.get("description", "")) for c in record.get("icd10_codes") or []]
    return Silver(pii_rows, token_rows, concept_rows, pii_ivs, concept_ivs, pii_type, nonpii_amb, concept_norms,
                  evidence, gold_codes)


# ---------------------------------------------------------------- scoring a rendered text

@lru_cache(maxsize=65536)
def mention_re(needle: str):
    """Compiled common.count_mentions pattern of a string."""
    n = norm(needle)
    return re.compile(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])") if n else None


def _count(needle: str, hay_norm: str) -> int:
    pat = mention_re(needle)
    return len(pat.findall(hay_norm)) if pat else 0


def scores(sv: Silver, orig: dict, texts: dict) -> tuple[float, float, dict]:
    """(P, U, normalized texts) of a rendering. P = 1 without PII, U = 1 without concepts."""
    an = {f: norm(texts.get(f) or "") for f in FIELDS}
    if sv.orig_counts is None:
        on = {f: norm(orig.get(f) or "") for f in FIELDS}
        sv.orig_counts = [_count(t, on[f]) for f, t, _ in sv.pii_rows + sv.token_rows + sv.concept_rows]
    oc = iter(sv.orig_counts)
    tot = rem = 0.0
    for f, t, w in sv.pii_rows + sv.token_rows:
        o = next(oc)
        tot += w * o
        rem += w * (o - min(_count(t, an[f]), o))
    ctot = kept = 0.0
    for f, t, w in sv.concept_rows:
        o = next(oc)
        ctot += w * o
        kept += w * min(_count(t, an[f]), o)
    return (rem / tot if tot else 1.0), (kept / ctot if ctot else 1.0), an


def alive_codes(sv: Silver, an: dict) -> set:
    """Gold codes with evidence of which at least one span survives."""
    return {c for c, spans in sv.evidence.items() if spans and any(_count(t, an[f]) for f, t in spans)}


def uncovered_words(record: dict, f: str, ivs: list, pii_ivs: list) -> int:
    n = 0
    for s, e in ivs:
        for m in re.finditer(r"\w+", record[f][s:e]):
            ws, we = s + m.start(), s + m.end()
            if not any(ws < pe and ps < we for ps, pe in pii_ivs):
                n += 1
    return n


# ---------------------------------------------------------------- parsing with positions

def parse_with_positions(text: str):
    """Parse the output object and record where each list item ends.

    Returns (obj, ends) where ends maps (key, i) -> end offset of item i of list `key`, and (key, "]") -> end of the
    list; or (None, {}) if `text` is not exactly one JSON object (surrounding whitespace allowed).
    """
    dec = json.JSONDecoder()
    ends, obj = {}, {}
    i, n = 0, len(text)

    def ws(i):
        while i < n and text[i] in " \t\r\n":
            i += 1
        return i

    try:
        i = ws(i)
        if text[i] != "{":
            return None, {}
        i = ws(i + 1)
        if text[i] == "}":
            return ({}, {}) if ws(i + 1) == n else (None, {})
        while True:
            key, i = dec.raw_decode(text, i)
            i = ws(i)
            if text[i] != ":":
                return None, {}
            i = ws(i + 1)
            if text[i] == "[":
                vals, i = [], ws(i + 1)
                if text[i] == "]":
                    i += 1
                else:
                    while True:
                        v, i = dec.raw_decode(text, i)
                        vals.append(v)
                        ends[(key, len(vals) - 1)] = i
                        i = ws(i)
                        if text[i] == ",":
                            i = ws(i + 1)
                            continue
                        if text[i] != "]":
                            return None, {}
                        i += 1
                        break
                ends[(key, "]")] = i
                obj[key] = vals
            else:
                obj[key], i = dec.raw_decode(text, i)
            i = ws(i)
            if text[i] == ",":
                i = ws(i + 1)
                continue
            if text[i] == "}":
                return (obj, ends) if ws(i + 1) == n else (None, {})
            return None, {}
    except (json.JSONDecodeError, IndexError, TypeError):
        return None, {}


# ---------------------------------------------------------------- dx scoring

def content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", str(s).lower()) if len(w) >= 3 and w not in R.ICD_STOPWORDS}


def word_f1(a: str, b: str) -> float:
    x, y = content_words(a), content_words(b)
    tp = len(x & y)
    return 2 * tp / (len(x) + len(y)) if tp else 0.0


def code_credit(pred: str, gold: str) -> float:
    p, g = norm_code(pred), norm_code(gold)
    if not p:
        return 0.0
    if p == g:
        return 1.0
    if p[:3] == g[:3]:
        return 0.5
    return 0.1 if p[:1] == g[:1] else 0.0


@lru_cache(maxsize=1)
def train_code_freq() -> Counter:
    path = DATA_DIR / "patient_doctor_encounters.jsonl"
    return Counter(norm_code(c["code"]) for r in read_jsonl(path) for c in r["icd10_codes"]) if path.exists() \
        else Counter()


def score_dx(dx: list, gold_codes: list, cfg: RewardConfig) -> list[float]:
    freq, used, out = train_code_freq(), set(), []
    for d in dx:
        if not isinstance(d, dict):
            out.append(cfg.dx_false_positive)
            continue
        best, bj = 0.0, None
        for j, (gc, gdesc) in enumerate(gold_codes):
            if j in used:
                continue
            s = word_f1(d.get("desc", ""), gdesc) + code_credit(d.get("code", ""), gc)
            if s > best:
                best, bj = s, j
        r = 0.0
        if bj is None or best < 0.3:
            r += cfg.dx_false_positive
        else:
            used.add(bj)
            r += best / math.sqrt(max(1, freq.get(gold_codes[bj][0], 0)))
        if d.get("code") and not is_valid_code(d["code"]):
            r += cfg.dx_invalid_code
        out.append(r)
    return out


# ---------------------------------------------------------------- the reward

@dataclass
class RewardResult:
    valid: bool
    items: list = field(default_factory=list)    # (end offset in prefix+response text, reward, label)
    parts: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)    # rendered dialogue/note

    @property
    def total(self) -> float:
        return sum(r for _, r, _ in self.items)


def compute(record: dict, sv: Silver, response: str, prefix: str = "", cfg: RewardConfig | None = None,
            protect_terms: frozenset | None = None) -> RewardResult:
    """Reward items of one rollout. `prefix` is the prefilled assistant text (part of the prompt, never rewarded)."""
    cfg = cfg or RewardConfig()
    full = prefix + response
    end = len(full)
    obj, ends = parse_with_positions(full)
    if obj is None or not isinstance(obj, dict):
        return RewardResult(False, [(end, cfg.invalid_json, "invalid_json")], {"valid": 0.0, "return": cfg.invalid_json})
    if protect_terms is None:   # as at inference: diagnosis words of the given codes, else of the model's own dx
        descs = [c.get("description", "") for c in record.get("icd10_codes") or []] if prefix else \
            [d.get("desc", "") for d in obj.get("dx", []) if isinstance(d, dict)]
        protect_terms = R.terms_of(descs)
    p = R.plan(record, obj, protect_terms=protect_terms, protected=sv.concept_ivs)
    items, parts = [], Counter()
    plen = len(prefix)

    def put(pos, r, label):
        items.append((pos if pos is not None else end, r, label))
        parts[label] += r

    # baseline: the regex pre-pass minus the vetoed spans
    active = {R.decision_id(d) for d in p.decisions if d.kind == "regex" and d.ok}
    texts, _ = R.apply(p, active)
    P0, U0, an = scores(sv, record, texts)
    Pc, Uc, alive = P0, U0, alive_codes(sv, an)
    ents = [d for d in p.decisions if d.kind == "entity"]
    for d in ents:
        pos = ends.get(("entities", d.index))
        if d.status == "duplicate":
            put(pos, 0.0, "duplicate")
            continue
        if not d.ok:
            put(pos, cfg.rejected, "rejected")
            parts["n_rejected"] += 1
            continue
        active.add(R.decision_id(d))
        texts, used = R.apply(p, active)
        Pn, Un, an = scores(sv, record, texts)
        alive_n = alive_codes(sv, an)
        mine = used.get(R.decision_id(d), {})
        fp = sum(uncovered_words(record, f, [(s, e) for s, e, _ in v], sv.pii_ivs[f]) for f, v in mine.items())
        killed = len(alive - alive_n)
        put(pos, cfg.w_P * (Pn - Pc), "dP")
        put(pos, cfg.w_U * (Un - Uc), "dU")
        if fp:
            put(pos, -cfg.lam_fp * fp, "fp")
        if killed:
            put(pos, -cfg.kill_evidence * killed, "kill_evidence")
        if cfg.type_bonus and sv.pii_type.get(norm(d.text)) == d.type:
            put(pos, cfg.type_bonus, "type")
        Pc, Uc, alive = Pn, Un, alive_n
    put(ends.get(("entities", "]")), -cfg.w_miss * (1 - Pc), "miss")

    # keep: vetoes of regex spans and retained ambiguous terms
    regex = [d for d in p.decisions if d.kind == "regex"]
    for d in p.decisions:
        if d.kind != "keep" or not d.ok:
            continue
        k = norm(d.text)
        vetoed = [r for r in regex if r.status == "vetoed" and (norm(r.text) == k or f" {norm(r.text)} " in f" {k} ")]
        if k in sv.pii_type or any(norm(r.text) in sv.pii_type or _is_silver_pii(record, sv, r.text) for r in vetoed):
            r_ = cfg.keep_pii
        elif k in sv.nonpii_amb or vetoed:
            r_ = cfg.keep_ok
        else:
            r_ = 0.0
        put(ends.get(("keep", d.index)), r_, "keep")

    # dx: only what the model wrote (a prefilled dx ends inside the prefix)
    dx = obj.get("dx") if isinstance(obj.get("dx"), list) else []
    if dx and ends.get(("dx", "]"), 0) > plen:
        for k, r_ in enumerate(score_dx(dx, sv.gold_codes, cfg)):
            if ends.get(("dx", k), 0) > plen:
                put(ends.get(("dx", k)), r_, "dx")

    # evidence spans that are real, silver concepts and survive rendering
    for k, ev in enumerate(obj.get("evidence") if isinstance(obj.get("evidence"), list) else []):
        if not isinstance(ev, dict) or ends.get(("evidence", k), 0) <= plen:
            continue
        good = 0
        for s in ev.get("spans", []) if isinstance(ev.get("spans"), list) else []:
            s = str(s).strip()
            ns = norm(s)
            if not s or not any(R.find_all(s, record.get(f) or "") for f in FIELDS):
                continue
            if not any(ns == c or f" {ns} " in f" {c} " or f" {c} " in f" {ns} " for c in sv.concept_norms):
                continue
            if any(count_mentions(s, an[f]) for f in FIELDS):
                good += 1
        if good:
            put(ends.get(("evidence", k)), cfg.evidence_ok * good, "evidence")

    n_ent = len(ents)
    parts.update({"valid": 1.0, "P": Pc, "U": Uc, "P0": P0, "U0": U0, "n_entities": n_ent,
                  "n_applied": sum(d.ok for d in ents)})
    parts["return"] = sum(r for _, r, _ in items)
    return RewardResult(True, items, dict(parts), texts)


def _is_silver_pii(record: dict, sv: Silver, text: str) -> bool:
    return any(any(s < pe and ps < e for ps, pe in sv.pii_ivs[f]) for f in FIELDS
               for s, e in R.find_all(text, record.get(f) or ""))


# ---------------------------------------------------------------- tokens

@lru_cache(maxsize=4)
def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_char_ends(tokenizer, ids: list[int]) -> list[int]:
    """End character offset in the decoded text of each token (byte-level BPE; special tokens add nothing)."""
    bd = _byte_decoder()
    special = set(tokenizer.all_special_ids)
    pieces = []
    for tok, i in zip(tokenizer.convert_ids_to_tokens(ids), ids):
        if i in special or tok is None:
            pieces.append(b"")
        else:
            pieces.append(bytes(bd[c] for c in tok if c in bd))
    out, buf = [], b""
    for b in pieces:
        buf += b
        out.append(len(buf.decode("utf-8", errors="ignore")))
    return out


def to_token_rewards(result: RewardResult, char_ends: list[int], prefix_len: int) -> list[float]:
    """Dense per-token reward vector of the response: each item lands on the first token reaching its offset."""
    import bisect
    out = [0.0] * len(char_ends)
    if not out:
        return out
    for pos, r, _ in result.items:
        c = pos - prefix_len
        j = bisect.bisect_left(char_ends, c) if c > 0 else 0
        out[min(j, len(out) - 1)] += r
    return out
