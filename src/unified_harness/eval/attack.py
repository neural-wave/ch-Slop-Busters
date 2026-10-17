"""Adversarial privacy attacks (bonus task a). Development-time tooling; never part of the pipeline.

Every attack runs on the original text too, as a positive control: an attack that cannot recover the
information from the *original* says nothing about the anonymization.

A1  independent-NER residual scan: obi/deid_roberta_i2b2 + GLiNER-PII (+ the rule detectors) on the
    output; recovered = gold PII strings that these detectors still find.
A2  roster dictionary attack: the attacker knows every patient name in the corpus plus US Census
    names and greps for them; recovered = gold names found in the right encounter.
A3  rare-word attack: encounter-unique non-medical tokens (re-identification by rare words).
A4  LLM inference attack: a local LLM reads the anonymized encounter and guesses first/last name,
    exact age, place and employer; scored against the gold values (age: within +-1 year).
A5  record linkage: TF-IDF nearest neighbour from anonymized to original text without codes. Clinical
    content is preserved by design, so this is the expected upper bound for an attacker who already
    holds the original record; reported to bound what anonymization can and cannot promise.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

from .. import lexicons as L
from .. import paths
from ..io import read_jsonl
from ..text.normalize import Doc
from .privacy import load_gold

PLACEHOLDER = re.compile(r"\[[A-Z_0-9+<> ]+\]")


def _gold_values(g: dict) -> dict[str, set[str]]:
    vals: dict[str, set[str]] = {}
    for it in g["pii"]:
        vals.setdefault(it["category"], set()).add(it["text"].lower())
    return vals


def a1_ner(original, anonymized, gold, limit=None) -> dict:
    from ..config import make_config
    from ..detect.encoders import load
    from ..detect.rules import detect as rule_detect

    cfg = make_config("balanced", detectors=("rules", "deid", "gliner"))
    dets = {n: load(n, cfg) for n in ("deid", "gliner")}
    out = {}
    for label, recs in (("original", original), ("anonymized", anonymized)):
        hit = tot = 0
        found = []
        for r in recs[:limit]:
            g = gold.get(r["encounter_id"])
            if not g:
                continue
            spans = []
            for f in ("dialogue", "note"):
                d = Doc(f, r[f])
                spans += [(f, d.raw[s.start : s.end].lower()) for s in rule_detect(d)]
                for det in dets.values():
                    spans += [(f, d.raw[s.start : s.end].lower()) for s in det.detect(d)]
            texts = {(f, t) for f, t in spans if not PLACEHOLDER.search(t)}
            for it in g["pii"]:
                if it.get("risk") == "low":
                    continue
                tot += 1
                tl = it["text"].lower()
                if any(f == it["field"] and (tl in t or (t in tl and len(t) >= 3)) for f, t in texts):
                    hit += 1
                    if label == "anonymized":
                        found.append(
                            {
                                "encounter_id": r["encounter_id"],
                                "field": it["field"],
                                "text": it["text"],
                                "category": it["category"],
                            }
                        )
        out[label] = {"recovered": hit, "of": tot, "rate": round(hit / tot, 4) if tot else None}
        if label == "anonymized":
            out["recovered_items"] = found
    return out


def a2_roster(original, anonymized, gold) -> dict:
    roster = set()
    for g in gold.values():
        for it in g["pii"]:
            if it["category"] in ("PATIENT_NAME", "PERSON_NAME"):
                roster |= {w for w in re.findall(r"[a-z][a-z'\-]+", it["text"].lower()) if len(w) >= 3}
    census = {n for n, c in L.surnames().items() if c >= 20000} | {n for n, s in L.first_names().items() if s >= 0.05}
    out = {}
    for label, recs in (("original", original), ("anonymized", anonymized)):
        hit = tot = 0
        found = []
        for r in recs:
            g = gold.get(r["encounter_id"])
            if not g:
                continue
            for it in g["pii"]:
                if it["category"] not in ("PATIENT_NAME", "PERSON_NAME"):
                    continue
                for w in re.findall(r"[a-z][a-z'\-]+", it["text"].lower()):
                    if w not in roster and w not in census:
                        continue
                    tot += 1
                    if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", r[it["field"]].lower()):
                        hit += 1
                        if label == "anonymized":
                            found.append({"encounter_id": r["encounter_id"], "field": it["field"], "name": w})
        out[label] = {"recovered": hit, "of": tot, "rate": round(hit / tot, 4) if tot else None}
        if label == "anonymized":
            out["recovered_items"] = found
    out["roster_size"] = len(roster | census)
    return out


def _english() -> set[str]:
    words: set[str] = set()
    for p in (Path("/usr/share/dict/words"), Path("/usr/share/dict/web2")):
        if p.exists():
            words |= {w.strip().lower() for w in p.open(encoding="utf-8", errors="ignore")}
    return words


def _rare_tokens(recs: list[dict]) -> dict[str, set[str]]:
    """Encounter-unique tokens that are not medical, not dictionary words, not ASR fragments."""
    df: Counter = Counter()
    toks = {}
    med = L.medical_vocab() | L.drug_tokens()
    eng = _english()
    wf = L.wordfreq()
    for r in recs:
        t = set(re.findall(r"(?<![a-z\-])[a-z][a-z'\-]{3,}(?![a-z\-])", (r["dialogue"] + " " + r["note"]).lower()))
        t = {w.removesuffix("'s").removesuffix("'") for w in t if not w.endswith("-")}
        t = {w for w in t if w not in med and w not in eng and sum(wf.get(w, (0, 0))) < 20 and "-" not in w}
        toks[r["encounter_id"]] = t
        df.update(t)
    return {eid: {w for w in t if df[w] == 1} for eid, t in toks.items()}


def _identifier_like(w: str) -> bool:
    if L.lower_uses(w) >= 5 or w in L.MEDICAL_EPONYMS:
        return False
    return (
        w in L.first_names()
        or L.surnames().get(w, 0) >= 5000
        or w in L.places()
        or w in L.EMPLOYERS
        or w in L.SPORTS_TEAMS
        or w in L.BRANDS
    )


def a3_rare(original, anonymized) -> dict:
    ro_all = _rare_tokens(original)
    ro = {eid: {w for w in ws if _identifier_like(w)} for eid, ws in ro_all.items()}
    a_text = {r["encounter_id"]: (r["dialogue"] + " " + r["note"]).lower() for r in anonymized}
    kept = total = 0
    per = []
    for eid, ws in ro.items():
        k = sum(1 for w in ws if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", a_text.get(eid, "")))
        kept += k
        total += len(ws)
        per.append(k)
    raw_total = sum(len(v) for v in ro_all.values())
    raw_kept = sum(
        1
        for eid, ws in ro_all.items()
        for w in ws
        if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", a_text.get(eid, ""))
    )
    return {
        "rare_out_of_vocabulary_tokens": raw_total,
        "rare_out_of_vocabulary_surviving": raw_kept,
        "rare_tokens_original": total,
        "rare_tokens_surviving": kept,
        "rare_token_removal": round(1 - kept / total, 4) if total else None,
        "encounters_with_surviving_rare_token": sum(1 for k in per if k),
        "note": "rare identifier-like = occurs in exactly one encounter, not medical/drug/dictionary vocabulary, "
        "and is a known name, place, employer, team or brand",
    }


ATTACK_PROMPT = """You are a privacy auditor testing a de-identified medical record. Using every clue in the text,
guess the patient's identity. If the information was removed, still give your best guess.
Answer JSON only: {"first_name": "", "last_name": "", "age": 0, "city_or_state": "", "employer_or_job": ""}

RECORD:
"""


def _llm(model: str, prompt: str, num_ctx: int = 8192) -> dict:
    body = {"model": model, "prompt": prompt, "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 120, "seed": 0}}  # fmt: skip
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        try:
            return json.loads(json.loads(resp.read()).get("response", "{}"))
        except json.JSONDecodeError:
            return {}


def _record_text(r: dict, max_chars: int = 9000) -> str:
    note = r["note"][:4500]
    return (note + "\n\nDIALOGUE (excerpt):\n" + r["dialogue"])[:max_chars]


def _true_age(g: dict) -> int | None:
    from ..text.normalize import words_to_int

    for it in g["pii"]:
        if it["category"] == "AGE":
            m = re.search(r"\d{1,3}", it["text"])
            v = (
                int(m.group(0))
                if m
                else words_to_int(
                    re.sub(r"[- ]?(?:year|month|week|day)s?[- ]?old.*", "", it["text"].lower()).replace("are ", "")
                )
            )
            if v and "month" not in it["text"].lower():
                return v
    return None


def a4_llm(original, anonymized, gold, model: str, limit=None) -> dict:
    res = {}
    rows = []
    for label, recs in (("original", original), ("anonymized", anonymized)):
        c = Counter()
        n = Counter()
        for r in recs[:limit]:
            g = gold.get(r["encounter_id"])
            if not g:
                continue
            vals = _gold_values(g)
            guess = _llm(model, ATTACK_PROMPT + _record_text(r))
            names = {w for t in vals.get("PATIENT_NAME", set()) for w in re.findall(r"[a-z][a-z'\-]+", t)}
            if names:
                n["name"] += 1
                gn = {
                    str(guess.get("first_name", "")).lower().strip(),
                    str(guess.get("last_name", "")).lower().strip(),
                } - {""}
                c["name"] += bool(gn & names)
            age = _true_age(g)
            if age:
                n["age_exact"] += 1
                try:
                    ga = int(float(guess.get("age") or 0))
                except (TypeError, ValueError):
                    ga = 0
                c["age_exact"] += abs(ga - age) <= 1
            places = vals.get("LOCATION", set())
            if places:
                n["place"] += 1
                gp = str(guess.get("city_or_state", "")).lower()
                c["place"] += bool(gp) and any(p in gp or gp in p for p in places)
            jobs = vals.get("OCCUPATION", set()) | vals.get("ORGANIZATION", set())
            if jobs:
                n["employer_or_job"] += 1
                gj = str(guess.get("employer_or_job", "")).lower()
                c["employer_or_job"] += bool(gj) and any(j in gj or gj in j for j in jobs)
            rows.append({"version": label, "encounter_id": r["encounter_id"], "guess": guess})
        res[label] = {k: {"recovered": c[k], "of": n[k], "rate": round(c[k] / n[k], 4) if n[k] else None} for k in n}
    res["model"] = model
    res["guesses"] = rows
    return res


def a5_linkage(original, anonymized) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs_o = [r["dialogue"] + "\n" + r["note"] for r in original]
    docs_a = [PLACEHOLDER.sub(" ", r["dialogue"] + "\n" + r["note"]) for r in anonymized]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=1).fit(docs_o + docs_a)
    sim = (vec.transform(docs_a) @ vec.transform(docs_o).T).toarray()
    top1 = sum(int(sim[i].argmax() == i) for i in range(len(docs_a)))
    return {"top1_relink": top1, "of": len(docs_a), "rate": round(top1 / len(docs_a), 4),
            "note": "attacker holds the original records; clinical content is kept by design, "
                    "so text similarity re-links. "
                    "This bounds the threat model: anonymization protects against identifying the person, "
                    "not against matching a record one already has."}  # fmt: skip


def main(a: argparse.Namespace) -> int:
    anon = read_jsonl(a.anonymized)
    ids = [r["encounter_id"] for r in anon]
    by = {r["encounter_id"]: r for r in read_jsonl(a.original)}
    orig = [by[i] for i in ids]
    gold = load_gold(a.gold) if a.gold else {}
    res: dict = {"name": a.name, "n": len(anon)}
    if gold:
        res["A2_roster"] = a2_roster(orig, anon, gold)
        print("A2 roster:", {k: v for k, v in res["A2_roster"].items() if k != "recovered_items"})
    res["A3_rare_words"] = a3_rare(orig, anon)
    print("A3 rare words:", res["A3_rare_words"])
    res["A5_linkage"] = a5_linkage(orig, anon)
    print("A5 linkage:", {k: v for k, v in res["A5_linkage"].items() if k != "note"})
    if gold and not a.skip_ner:
        res["A1_ner"] = a1_ner(orig, anon, gold, a.limit)
        print("A1 NER:", {k: v for k, v in res["A1_ner"].items() if k != "recovered_items"})
    if gold and not a.skip_llm:
        res["A4_llm"] = a4_llm(orig, anon, gold, a.llm_model, a.limit)
        print("A4 LLM:", {k: v for k, v in res["A4_llm"].items() if k not in ("guesses",)})
    out = paths.eval_dir() / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")
    return 0
