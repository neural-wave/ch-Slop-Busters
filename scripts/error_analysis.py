"""Error analysis of one run against silver labels: why each labeled identifier survived.

    uv run python scripts/error_analysis.py outputs/runs/<run>/<date> --split tune [--split-file data/...]

For every labeled PII occurrence still present in the output, the cause is read from the decisions
sidecar (overlapping decision in the original text):

    no detector        no decision overlaps the occurrence
    kept: <reason>     a detector found it and the policy or the medical shield kept it
    edited elsewhere   an edit overlaps the occurrence but the labeled string still survives (partial edit)

Patterns are printed as abstract shapes (digits as 9, month and weekday names as MONTH / WEEKDAY, other
words as W, a few time words kept), never as the original string. Keep-trap losses and edits outside the
labels are broken down by the detector that made the edit. Output is Markdown on stdout.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from unified_harness import paths
from unified_harness.eval.privacy import load_gold, pattern
from unified_harness.io import read_jsonl

MONTHS = set(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
WEEKDAYS = set("monday tuesday wednesday thursday friday saturday sunday".split())
TIME_WORDS = set(
    "spring summer fall autumn winter last next this ago of the in on year years old month months week weeks "
    "day days age aged early late mid".split()
)
NUM_WORDS = set(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred first second third".split()
)


def shape(text: str) -> str:
    out = []
    for tok in re.findall(r"\d+|[A-Za-z']+|[^\sA-Za-z\d]", text):
        low = tok.lower()
        if tok.isdigit():
            out.append("9" * min(len(tok), 4))
        elif low in MONTHS:
            out.append("MONTH")
        elif low in WEEKDAYS:
            out.append("WEEKDAY")
        elif low in NUM_WORDS:
            out.append("NUM")
        elif low in TIME_WORDS:
            out.append(low)
        elif tok.isalpha() or "'" in tok:
            out.append("W")
        else:
            out.append(tok)
    s = " ".join(out)
    return re.sub(r"(W )+W", "W+", s)


def reason_class(r: str | None) -> str:
    r = r or "?"
    return r.split(" (")[0] if r.startswith("shield:") else r


def overlaps(decs: list[dict], s: int, e: int) -> list[dict]:
    return [d for d in decs if d["start"] < e and s < d["end"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    anon_file = next(p for p in a.run_dir.glob("*.jsonl") if not p.name.endswith(".decisions.jsonl"))
    anon = {r["encounter_id"]: r for r in read_jsonl(anon_file)}
    orig = [r for r in read_jsonl(paths.split_file(a.split)) if r["encounter_id"] in anon]
    gold = load_gold(paths.gold_dir(a.split))
    decs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in read_jsonl(next(a.run_dir.glob("*.decisions.jsonl"))):
        decs[(d["encounter_id"], d["field"])].append(d)

    causes: dict[str, Counter] = defaultdict(Counter)
    shapes: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    trap_loss: dict[str, Counter] = defaultdict(Counter)
    for rec in orig:
        eid, g = rec["encounter_id"], gold.get(rec["encounter_id"])
        if not g:
            continue
        for it in g["pii"]:
            f, rx = it["field"], pattern(it["text"])
            occ = list(rx.finditer(rec[f]))
            if not occ:
                continue
            left = min(len(rx.findall(anon[eid][f])), len(occ))
            cat = f"{it['category']} ({it.get('risk', 'high')})"
            totals[cat] += len(occ)
            if not left:
                continue
            shapes[cat][shape(it["text"])] += left
            found = []
            for m in occ:
                ds = overlaps(decs[(eid, f)], m.start(), m.end())
                edits = [d for d in ds if d["action"] != "KEEP"]
                if edits:
                    continue
                keeps = [d for d in ds if d["action"] == "KEEP"]
                found.append("kept: " + reason_class(keeps[0]["policy_reason"]) if keeps else "no detector")
            found = (found or ["edited elsewhere"] * left)[:left]
            found += ["edited elsewhere"] * (left - len(found))
            causes[cat].update(found)
        for it in g.get("keep", []):
            f, rx = it["field"], pattern(it["text"])
            occ = list(rx.finditer(rec[f]))
            if len(rx.findall(anon[eid][f])) >= len(occ):
                continue
            lost = len(occ) - len(rx.findall(anon[eid][f]))
            for m in occ:
                edits = [d for d in overlaps(decs[(eid, f)], m.start(), m.end()) if d["action"] != "KEEP"]
                if edits and lost:
                    d, lost = edits[0], lost - 1
                    trap_loss[it["category"]][f"{d['source']} -> {d['category']}"] += 1
            if lost:
                trap_loss[it["category"]]["no overlapping edit (text changed by an edit next to it)"] += lost

    print(f"## {a.run_dir.parent.name} on {a.split}\n")
    print("### Surviving labeled PII occurrences by category and cause\n")
    print("| category (risk) | occurrences | left | causes | top shapes |")
    print("|---|---:|---:|---|---|")
    for cat in sorted(causes, key=lambda c: -sum(causes[c].values())):
        c = ", ".join(f"{k} {v}" for k, v in causes[cat].most_common())
        s = ", ".join(f"`{k}` {v}" for k, v in shapes[cat].most_common(a.top))
        print(f"| {cat} | {totals[cat]} | {sum(causes[cat].values())} | {c} | {s} |")
    print("\n### Keep-trap occurrences removed, by trap category and the edit that removed them\n")
    print("| trap category | removed by (detector -> category) |")
    print("|---|---|")
    for cat, cnt in sorted(trap_loss.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"| {cat} | " + ", ".join(f"{k} {v}" for k, v in cnt.most_common(a.top)) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
