"""Audit the merged ACI-eval gold and its three source golds: data/eval/gold/aci_eval/audit.md and audit.json.

Usage: python3 -I eval/gold/audit_gold.py
       python3 -I eval gold --audit

Needs no model; reads merged.jsonl (run `python3 -I eval gold` first) and the source golds. It joins the checks of
ketai-eval/gold/audit_gold.py and vm-backup/neuralwave/evaluation/check_annotations.py:

    verbatim         per source gold, strings with no case-insensitive whole-word occurrence in their field (and
                     those that are not even a substring)
    contradictions   merged PII that is also a keep item or a medical concept, PII made only of words of the
                     encounter's own ICD-10 descriptions (ages, dates, IDs and contacts excepted), and keep items inside
                     a longer PII item (removing the PII item removes them too)
    possibly missed  regex candidates that no union item and no keep item covers: ages, dates, title + name,
                     e-mails, phone numbers, digit runs of 6+
    single votes     union items only one gold lists, and whether they still count at the consensus level (direct)
    votes            vote histogram per category
    diff             items added, removed or changed since the previous merged.jsonl (merged.prev.jsonl)

Exit code 1 when merged.jsonl is missing. audit.md quotes gold strings: keep it under
data/eval/.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.common import FIELDS, count, load_jsonl, pattern, write_json
from eval.gold import merge_aci_gold as G
from eval.gold import taxonomy as T

_UNITS = "one|two|three|four|five|six|seven|eight|nine"
_TEENS = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
NUM = rf"(?:\d{{1,3}}|(?:{_TENS})(?:[ -](?:{_UNITS}))?|{_TEENS}|{_UNITS})"
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
NOT_UNIT = r"(?![ \t]*(?:iu|mg|milligrams?|mcg|micrograms?|units?|ml|cc|calories|steps|feet|miles|dollars|percent|%))"
CANDIDATES = {
    "AGE": re.compile(rf"\b{NUM}[ -](?:years?|yrs?|months?|weeks?)[ -]old\b|\b(?:age|aged)[ \t]+(?:of[ \t]+)?{NUM}\b"
                      rf"|\b\d{{1,3}}[ \t]*(?:y/o|y\.o\.|yo)\b", re.IGNORECASE),
    "DATE": re.compile(rf"(?<![\w/.])\d{{1,2}}[/-]\d{{1,2}}[/-](?:\d{{4}}|\d{{2}})(?![\w/])"
                       rf"|\b(?:{MONTHS})[ \t]+(?:\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?[ \t]+\d{{4}}\b)?|\d{{4}}\b)"
                       rf"|\b\d{{1,2}}(?:st|nd|rd|th)?[ \t]+of[ \t]+(?:{MONTHS})\b|(?<![\w/.,$])(?:19|20)\d{{2}}(?![\w/%]|[.,]\d){NOT_UNIT}"
                       rf"|\b(?:twenty[ -](?:{_TEENS}|twenty(?:[ -](?:{_UNITS}))?)"
                       rf"|two thousand(?:[ -]and)?[ -](?:{_TEENS}|{_UNITS}|twenty(?:[ -](?:{_UNITS}))?)"
                       rf"|nineteen[ -](?:{_TENS})(?:[ -](?:{_UNITS}))?)\b", re.IGNORECASE),
    "TITLE+NAME": re.compile(r"\b(?:dr|mr|mrs|ms)\.?[ \t]+([a-z][a-z'-]{2,})", re.IGNORECASE),
    "CONTACT": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b|(?<!\d)(?:\(?\d{3}\)?[ .-]?\d{3}[ .-]\d{4})(?!\d)"),
    "ID": re.compile(r"(?<![\w./])[a-z]*\d{6,}(?![\d./])", re.IGNORECASE),
}  # fmt: skip
NOT_NAMES = {"and", "the", "you", "was", "has", "had", "her", "his", "who", "will", "said", "says", "here", "about", "with", "would", "also",
             "again", "okay", "office", "visit", "arthrogram"}  # fmt: skip
ICD_STOPWORDS = {"and", "or", "of", "the", "with", "without", "due", "to", "in", "on", "for", "by", "not", "other", "unspecified", "specified",
                 "type", "site", "part", "encounter", "initial", "subsequent", "sequela", "elsewhere", "classified", "person", "history"}  # fmt: skip
NOT_ICD_CHECKED = {"AGE", "DATE", "DOB", "YEAR", "ID", "CONTACT"}


def norm(text: str) -> str:
    """Case and spacing-insensitive form (ketai's evaluator): hyphens, underscores and slashes as spaces, punctuation apart."""
    t = re.sub(r"[-_/]", " ", text.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", re.sub(r"([.,;:!?()\[\]\"])", r" \1 ", t)).strip()


def candidates(text: str) -> list[tuple[str, str, str]]:
    """(kind, matched string, string a gold item must cover) of the regex PII candidates in a field."""
    out = []
    for kind, rx in CANDIDATES.items():
        for m in rx.finditer(text):
            core = m.group(1) if kind == "TITLE+NAME" else m.group(0)
            if kind != "TITLE+NAME" or core.lower() not in NOT_NAMES:
                out.append((kind, m.group(0), core))
    return out


def covered(cand: str, gold: list[str]) -> bool:
    c = f" {norm(cand)} "
    return any(c in f" {g} " or f" {g} " in c for g in gold)


def _words(text: str) -> list[str]:
    return [re.sub(r"'s?$", "", w) for w in re.findall(r"[a-z][a-z']*", text.lower().replace("’", "'"))]


def icd_terms(rec: dict) -> frozenset[str]:
    """Content words of the encounter's own ICD-10 descriptions."""
    return frozenset(w for c in rec.get("icd10_codes") or [] for w in _words(str(c.get("description", ""))) if len(w) >= 3 and w not in ICD_STOPWORDS)


def is_icd_term(text: str, terms: frozenset[str]) -> bool:
    ws = [w for w in _words(text) if w not in ICD_STOPWORDS]
    return bool(ws) and all(w in terms or w in ("left", "right") for w in ws)


def verbatim(records: dict[str, dict], golds: dict[str, dict[str, dict]]) -> tuple[dict, list[list]]:
    """Per source gold: strings with no whole-word occurrence in their field (check_annotations.py, stricter)."""
    summary, rows = {}, []
    for src, by_eid in golds.items():
        n = bad = notsub = 0
        for eid, g in sorted(by_eid.items()):
            for kind in ("pii", "keep", "concepts", "not_pii"):
                for it in g.get(kind, []):
                    n += 1
                    text = records[eid].get(it["field"]) if it["field"] in FIELDS else None
                    if text is None or not count(text, it["text"]):
                        sub = text is not None and it["text"].lower() in text.lower()
                        bad, notsub = bad + 1, notsub + (not sub)
                        rows.append([src, eid, kind, it["field"], f"`{it['text']}`", "not a whole word" if sub else "not a substring"])
        summary[src] = {"items": n, "not_whole_word": bad, "not_substring": notsub}
    return summary, rows


def contradictions(r: dict, rec: dict) -> list[list]:
    """PII that is also keep or a concept, PII made of the encounter's ICD-10 description words, keep inside PII."""
    keep, concepts, terms, out = {G.key(k) for k in r["keep"]}, {G.key(c) for c in r["concepts"]}, icd_terms(rec), []
    for it in r["pii"]:
        why = "PII and keep" if G.key(it) in keep else "PII and concept" if G.key(it) in concepts else None
        if why is None and it["category"] not in NOT_ICD_CHECKED and is_icd_term(it["text"], terms):
            why = "PII made of ICD-10 description words"
        if why:
            out.append([r["encounter_id"], why, it["field"], f"`{it['text']}`", it["category"], it["votes"]])
    for k in r["keep"]:
        inside = [it for it in r["pii"] if it["field"] == k["field"] and len(it["text"]) > len(k["text"]) and pattern(k["text"]).search(it["text"])]
        if inside:
            out.append([r["encounter_id"], "keep item inside a PII item", k["field"], f"`{k['text']}` in " + "; ".join(f"`{it['text']}`" for it in inside),
                        k["category"], "/".join(str(it["votes"]) for it in inside)])  # fmt: skip
    return out


def possibly_missed(r: dict, rec: dict) -> list[list]:
    out = []
    for f in FIELDS:
        gold, seen = [norm(it["text"]) for it in r["pii"] + r["keep"] if it["field"] == f], set()
        for kind, cand, core in candidates(rec[f]):
            if (kind, cand.lower()) not in seen and not covered(core, gold):
                seen.add((kind, cand.lower()))
                out.append([r["encounter_id"], kind, f, f"`{cand}`"])
    return out


def diff(prev: list[dict], rows: list[dict]) -> tuple[dict, list[list]]:
    """PII items added, removed or changed (category, votes, direct) since the previous merged.jsonl."""
    old = {(r["encounter_id"], *G.key(it)): it for r in prev for it in r["pii"]}
    new = {(r["encounter_id"], *G.key(it)): it for r in rows for it in r["pii"]}
    state = lambda it: f"{it['category']} {it['votes']}v{' direct' if it['direct'] else ''}" if it else "–"  # noqa: E731
    out = []
    for k in sorted(old.keys() | new.keys()):
        a, b = old.get(k), new.get(k)
        if a is None or b is None or any(a.get(x) != b.get(x) for x in ("category", "votes", "direct")):
            out.append([k[0], k[1], f"`{(b or a)['text']}`", "added" if a is None else "removed" if b is None else "changed", state(a), state(b)])
    total = lambda rs, k: sum(len(r[k]) for r in rs)  # noqa: E731
    return {"pii": dict(Counter(x[3] for x in out)), **{k: [total(prev, k), total(rows, k)] for k in ("keep", "concepts")}}, out


def audit(records: dict[str, dict], golds: dict[str, dict[str, dict]], rows: list[dict], prev: list[dict] | None) -> dict:
    """Every check: {"summary": counts, "tables": {name: rows}}."""
    verb, verb_rows = verbatim(records, golds)
    contra = [x for r in rows for x in contradictions(r, records[r["encounter_id"]])]
    missed = [x for r in rows for x in possibly_missed(r, records[r["encounter_id"]])]
    pii = [(r["encounter_id"], it) for r in rows for it in r["pii"]]
    single = [(eid, it) for eid, it in pii if it["votes"] == 1]
    by_cat: dict[str, Counter] = {}
    for _, it in pii:
        by_cat.setdefault(it["category"], Counter())[it["votes"]] += 1
    changes, diff_rows = diff(prev, rows) if prev is not None else (None, [])
    summary = {
        "encounters": len(rows),
        "verbatim": verb,
        "contradictions": dict(sorted(Counter(x[1] for x in contra).items())),
        "possibly_missed": dict(sorted(Counter(x[1] for x in missed).items())),
        "single_votes": {"items": len(single), "in_consensus": sum(T.in_level(it, "consensus") for _, it in single),
                         "by_source": dict(sorted(Counter(it["sources"][0] for _, it in single).items()))},
        "votes_by_category": {c: {str(n): v[n] for n in sorted(v)} | {"direct": T.is_direct(c)} for c, v in sorted(by_cat.items())},
        "diff": changes,
    }  # fmt: skip
    singles = [[eid, it["field"], f"`{it['text']}`", it["category"], it["sources"][0], "yes" if T.in_level(it, "consensus") else "no"] for eid, it in single]
    tables = {"verbatim": verb_rows, "contradictions": contra, "possibly_missed": missed, "single_votes": singles, "diff": diff_rows}
    return {"summary": summary, "tables": tables}


HEADERS = {
    "verbatim": ["source", "encounter", "list", "field", "text", "problem"],
    "contradictions": ["encounter", "finding", "field", "text", "category", "votes"],
    "possibly_missed": ["encounter", "kind", "field", "candidate"],
    "single_votes": ["encounter", "field", "text", "category", "source", "in consensus"],
    "diff": ["encounter", "field", "text", "change", "before", "after"],
}


def markdown(res: dict) -> str:
    s, t = res["summary"], res["tables"]
    md = ["# Audit of the merged ACI-eval gold", "",
          "Written by `python3 -I eval gold --audit` (eval/gold/audit_gold.py). It quotes gold strings: keep it under data/eval/.", "", "## Summary", "",
          "- verbatim: " + "; ".join(f"{src} {v['not_whole_word']} of {v['items']} strings not found as whole words "
                                     f"({v['not_substring']} not even as a substring)" for src, v in s["verbatim"].items()),
          f"- contradictions: {s['contradictions'] or 'none'}",
          f"- possibly missed (regex candidates that no union or keep item covers): {s['possibly_missed'] or 'none'}",
          f"- single-vote items: {s['single_votes']['items']} ({s['single_votes']['in_consensus']} of them direct, so in consensus), "
          f"by source {s['single_votes']['by_source']}",
          f"- diff vs merged.prev.jsonl: {s['diff'] if s['diff'] is not None else 'no previous merged file'}", "", "## Votes per category", ""]  # fmt: skip
    votes = [[c, "yes" if v["direct"] else "no", v.get("1", 0), v.get("2", 0), v.get("3", 0)] for c, v in s["votes_by_category"].items()]
    md += G.md_table(["category", "direct", "1 vote", "2 votes", "3 votes"], votes) + [""]
    for name, header in HEADERS.items():
        md += [f"## {name.replace('_', ' ').capitalize()} ({len(t[name])})", ""] + (G.md_table(header, t[name]) if t[name] else ["none"]) + [""]
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval/gold/audit_gold.py", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    out = G.OUT_DIR
    if not (out / "merged.jsonl").exists():
        print(f"error: {out / 'merged.jsonl'} is missing: run `python3 -I eval gold` first", file=sys.stderr)
        return 1
    records, golds, _ = G.load_golds(G.HARNESS, G.VM_DIR, G.KETAI)
    prev = out / "merged.prev.jsonl"
    res = audit(records, golds, load_jsonl(out / "merged.jsonl"), load_jsonl(prev) if prev.exists() else None)
    write_json(out / "audit.json", res)
    (out / "audit.md").write_text(markdown(res), encoding="utf-8")
    s = res["summary"]
    print(f"audit {G.SUITE}: verbatim failures {sum(v['not_whole_word'] for v in s['verbatim'].values())}, contradictions "
          f"{sum(s['contradictions'].values())}, possibly missed {sum(s['possibly_missed'].values())}, single votes {s['single_votes']['items']}")
    print(f"  {out / 'audit.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
