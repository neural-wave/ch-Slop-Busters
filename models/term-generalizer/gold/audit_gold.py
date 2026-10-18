"""Cross-check the reference annotation (gold/eval_gold.jsonl) against data/ before trusting it.

The annotation was produced by LLM agents, not by the organizers, so the reference privacy and
concept metrics of evaluate.py are only as good as it is. This script needs no model and reports:

  coverage        encounters of data/eval_patient_doctor_encounters.jsonl without annotation, and spans
                  that are not verbatim in their field
  contradictions  strings that are both PII and a medical concept or a non-PII ambiguous term, and PII
                  made only of words of the encounter's own ICD-10 descriptions (the task says ICD
                  information must be preserved)
  possibly missed deterministic PII candidates in data/ that no gold PII string covers: regex ages and
                  dates (generalize.find_spans), "dr./mr./mrs./ms. <name>", emails, phone numbers and
                  long digit runs (record numbers)
  judgment calls  the categories that depend most on the annotator (OTHER_QUASI, OCCUPATION), listed
                  with the annotator's comment for manual review

  python3 ../term-generalizer/gold/audit_gold.py            (--verbose lists the judgment calls)
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generalize  # noqa: E402
from anonymize import icd_terms, is_icd_term  # noqa: E402
from common import EVAL_DATA, GOLD_EVAL, norm, read_jsonl  # noqa: E402

TITLE_NAME = re.compile(r"\b(?:dr|mr|mrs|ms)\.?[ \t]+([a-z][a-z'-]{2,})", re.IGNORECASE)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE = re.compile(r"(?<!\d)(?:\(?\d{3}\)?[ .-]?\d{3}[ .-]\d{4})(?!\d)")
DIGITS = re.compile(r"(?<![\w./])[a-z]*\d{6,}(?![\d./])", re.IGNORECASE)  # "123546", "mrn472348"
NOT_NAMES = {"and", "the", "you", "was", "has", "had", "her", "his", "who", "will", "said", "says", "here",
             "about", "with", "would", "also", "again", "okay", "office", "visit", "arthrogram"}
JUDGMENT = ("OTHER_QUASI", "OCCUPATION")


def candidates(text: str) -> list[tuple[str, str, str]]:
    """(kind, matched string, string that gold must cover) PII candidates found by deterministic patterns."""
    out = [(s["type"], s["text"], s["text"]) for s in generalize.find_spans(text)]
    out += [("TITLE+NAME", m.group(0), m.group(1)) for m in TITLE_NAME.finditer(text)
            if m.group(1).lower() not in NOT_NAMES]
    out += [("CONTACT", m.group(0), m.group(0)) for p in (EMAIL, PHONE) for m in p.finditer(text)]
    out += [("ID", m.group(0), m.group(0)) for m in DIGITS.finditer(text)]
    return out


def covered(cand: str, gold_norm: list[str]) -> bool:
    c = f" {norm(cand)} "
    return any(c in f" {g} " or f" {g} " in c for g in gold_norm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(EVAL_DATA))
    ap.add_argument("--gold", default=str(GOLD_EVAL))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data = {r["encounter_id"]: r for r in read_jsonl(args.data)}
    gold = {r["encounter_id"]: r for r in read_jsonl(args.gold)}
    issues = Counter()
    print(f"# Audit of {args.gold} against {args.data}\n")

    missing = sorted(set(data) - set(gold))
    extra = sorted(set(gold) - set(data))
    print(f"- encounters: {len(data)} in data, {len(gold)} annotated; missing {missing or 'none'};"
          f" not in data {extra or 'none'}")

    lines = []
    for eid in sorted(set(data) & set(gold)):
        rec, g = data[eid], gold[eid]
        terms = icd_terms(rec)
        med = {(c["field"], c["text"].lower()) for c in g.get("medical_concepts", [])}
        not_pii = {(t["field"], t["text"].lower()) for t in g.get("ambiguous_terms", []) if not t.get("is_pii")}
        for key in ("pii", "medical_concepts", "ambiguous_terms"):
            for e in g.get(key, []):
                if e["text"] not in rec.get(e["field"], ""):
                    issues["not_verbatim"] += 1
                    lines.append(f"| {eid} | not verbatim ({key}) | {e['field']} | `{e['text']}` |")
        for e in g.get("pii", []):
            k = (e["field"], e["text"].lower())
            if k in med or k in not_pii:
                issues["pii_also_medical"] += 1
                lines.append(f"| {eid} | PII and medical/non-PII term | {e['field']} | `{e['text']}` |")
            elif e["category"] not in ("AGE", "DATE", "ID", "CONTACT") and is_icd_term(e["text"], terms):
                issues["pii_is_icd_term"] += 1
                lines.append(f"| {eid} | PII made of ICD-10 description words | {e['field']} | `{e['text']}` |")
        for f in ("dialogue", "note"):
            gold_norm = [norm(e["text"]) for e in g.get("pii", []) if e["field"] == f]
            seen = set()
            for kind, cand, core in candidates(rec.get(f) or ""):
                if (kind, cand.lower()) in seen or covered(core, gold_norm):
                    continue
                seen.add((kind, cand.lower()))
                issues[f"possibly_missed_{kind}"] += 1
                lines.append(f"| {eid} | possibly missed {kind} | {f} | `{cand}` |")

    print(f"- findings: {dict(issues) or 'none'}\n")
    if lines:
        print("| encounter | finding | field | text |\n|---|---|---|---|")
        print("\n".join(lines))
        print()

    calls = [(eid, e) for eid, g in sorted(gold.items()) for e in g.get("pii", []) if e["category"] in JUDGMENT]
    n_pii = sum(len(g.get("pii", [])) for g in gold.values())
    print(f"- judgment calls: {len(calls)} of {n_pii} PII strings are {'/'.join(JUDGMENT)}"
          f" (hobbies, family facts, work context); their removal rate is the least reliable number"
          f" in the report{'' if args.verbose else ' (--verbose lists them)'}")
    if args.verbose:
        print("\n| encounter | category | field | text | annotator comment |\n|---|---|---|---|---|")
        for eid, e in calls:
            print(f"| {eid} | {e['category']} | {e['field']} | `{e['text']}` | {e.get('comment', '')} |")


if __name__ == "__main__":
    main()
