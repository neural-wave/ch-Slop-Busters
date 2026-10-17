"""Build the merged silver labels for every evaluation split (labels/merged/<split>/).

    uv run python scripts/labels/merge.py                 # merge, write conflicts for adjudication
    uv run python scripts/labels/merge.py --stats         # also print label statistics as Markdown

Splits (configs/splits.json):
  tune        all 140 train encounters: upstream dev labels (40) + M2 labels (100), upstream format
  validation  the 67 eval encounters: upstream and Ketai labels merged; conflicts adjudicated
  test        the 24-document held-out set (upstream R4 labels), scored once at the end

Conflicts (a string that one source calls PII and the other calls clinical) are written to
labels/merged/conflicts.json with a short context window. Decisions go to labels/merged/adjudication.json
as [{"field", "text", "decision": "pii" | "keep" | "neither", "category", "why"}]; rerun this script to
apply them. Everything under labels/ is git-ignored.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from unified_harness import paths
from unified_harness.eval.labels import (
    apply_resolutions,
    from_upstream,
    load_ketai,
    load_upstream,
    merge_encounter,
    occurrences,
    write_merged,
)
from unified_harness.io import read_jsonl

CONTEXT = 90


def _context(record: dict, field: str, text: str) -> str:
    occ = occurrences(text, record[field])
    if not occ:
        return ""
    s, e = occ[0]
    return record[field][max(0, s - CONTEXT) : e + CONTEXT].replace("\n", " ")


def load_decisions(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    return {(d["encounter_id"], d["field"], d["text"].lower()): d for d in json.loads(path.read_text(encoding="utf-8"))}


def stats(name: str, merged: list[dict]) -> str:
    pii = [p for m in merged for p in m["pii"]]
    keep = [k for m in merged for k in m["keep"]]
    by_source = Counter("+".join(sorted(p["sources"])) for p in pii)
    by_cat = Counter(p["category"] for p in pii)
    by_risk = Counter(p["risk"] for p in pii)
    lines = [
        f"### {name}: {len(merged)} encounters, {len(pii)} PII strings, {len(keep)} keep traps",
        "",
        "| | count |",
        "|---|---:|",
        *[f"| source {k} | {v} |" for k, v in sorted(by_source.items())],
        *[f"| risk {k} | {v} |" for k, v in sorted(by_risk.items())],
        *[f"| {k} | {v} |" for k, v in by_cat.most_common()],
        f"| unresolved conflicts | {sum(len(m['conflicts']) for m in merged)} |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    labels, out = paths.labels_dir(), paths.labels_dir() / "merged"
    report = []

    # tune: upstream-format labels only (no Ketai labels exist for train)
    train = {r["encounter_id"]: r for r in read_jsonl(paths.split_file("tune"))}
    upstream_dev, new_train = load_upstream(labels / "upstream" / "dev"), load_upstream(labels / "upstream" / "train")
    tune = [
        from_upstream(upstream_dev[i], "upstream") if i in upstream_dev else from_upstream(new_train[i], "m2-agents")
        for i in paths.split_ids("tune")
    ]
    missing = [i for i in paths.split_ids("tune") if i not in upstream_dev and i not in new_train]
    if missing or set(paths.split_ids("tune")) - set(train):
        raise SystemExit(f"tune labels missing for {missing}")
    write_merged(out / "tune", tune)
    report.append(stats("tune", tune))

    # validation: merge both teams' eval labels, then apply adjudications
    evals = read_jsonl(paths.split_file("validation"))
    upstream_eval, ketai = load_upstream(labels / "upstream" / "eval"), load_ketai(labels / "ketai" / "eval_gold.jsonl")
    decisions = load_decisions(out / "adjudication.json")
    validation, conflicts = [], []
    for rec in evals:
        eid = rec["encounter_id"]
        m = merge_encounter(rec, upstream_eval.get(eid), ketai.get(eid))
        mine = {(f, t): d for (e, f, t), d in decisions.items() if e == eid}
        m = apply_resolutions(m, mine)
        for c in m["conflicts"]:
            conflicts.append({"encounter_id": eid, **c, "context": _context(rec, c["field"], c["text"])})
        validation.append(m)
    write_merged(out / "validation", validation)
    (out / "conflicts.json").write_text(json.dumps(conflicts, indent=1, ensure_ascii=False), encoding="utf-8")
    report.append(stats("validation", validation))

    # test: the untouched held-out set, as labeled upstream
    holdout = load_upstream(labels / "upstream" / "robust_holdout")
    test = [from_upstream(v, "upstream") for _, v in sorted(holdout.items())]
    write_merged(out / "test", test)
    report.append(stats("test", test))

    print(f"tune {len(tune)}, validation {len(validation)}, test {len(test)} encounters -> {out}")
    print(f"unresolved conflicts: {len(conflicts)} -> {out / 'conflicts.json'}")
    if args.stats:
        print("\n" + "\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
