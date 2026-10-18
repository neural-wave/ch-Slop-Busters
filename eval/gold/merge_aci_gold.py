"""Merge the three ACI-eval golds (harness, VM, ketai) into one gold: data/eval/gold/aci_eval/merged.jsonl.

Usage: python3 -I eval gold [--audit]
       python3 -I eval/gold/merge_aci_gold.py [--audit]

Reads (never writes) the harness gold data/converted/aci_bench/labels.jsonl (eval split), the VM teacher gold
vm-backup/neuralwave/annotations/eval/<id>.json and ketai's gold ketai-eval/gold/eval_gold.jsonl. A missing VM or
ketai gold is left out with a warning; without data/converted/aci_bench nothing is written.

Writes data/eval/gold/aci_eval/:

    merged.jsonl       one row per encounter: `pii` (merged items with votes), `pii_by_source` (each gold's own list,
                       as the suites used to score it), `keep`, `concepts`
    merged_stats.json  counts per source, exact overlap, items per level, nesting, conflicts
    merged_review.md   every decision worth a look: category and PII/keep conflicts, nesting, dropped concepts
    merged.prev.jsonl  the previous merged.jsonl, kept when the content changed (audit_gold.py diffs against it)

Rules (eval/docs/GOLD.md): labels map to the common categories (taxonomy.py) and items match per (encounter, field,
lowercased text). Nesting: the longest span is canonical; a shorter item that is a whole-word substring of it is
listed in `contains` and gives its golds a nested vote; it stays an item of its own only when it also occurs outside
every container (`standalone_occurrences`), with the containers' votes. `votes` counts the golds behind an item. A
gold that lists a string as not PII (keep) never votes for it through nesting or inheritance. Keep = harness keep +
ketai's non-PII ambiguous terms; a string in both stays PII when two golds list it (or a part of it) themselves,
else keep wins. Concepts = VM + ketai, minus those equal to a direct PII item. Building is deterministic: the same
inputs give byte-identical files, so `eval all` runs it every time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import suites as S
from eval.common import CATEGORY_RISK, CONTAINER, CONVERTED, FIELDS, count, distinct_items, load_jsonl, occurrences, pattern, sha1_file, sha1_text, write_json
from eval.gold import taxonomy as T

HARNESS = CONVERTED / "aci_bench"
VM_DIR = S.VM_ANNOTATIONS
KETAI = CONTAINER / "ketai-eval" / "gold" / "eval_gold.jsonl"
SUITE = "aci_eval"
OUT_DIR = S.GOLD_DIR / SUITE

Key = tuple[str, str]  # (field, lowercased text)


def key(it: dict) -> Key:
    return it["field"], it["text"].lower()


def _order(k: Key) -> tuple:
    return FIELDS.index(k[0]) if k[0] in FIELDS else len(FIELDS), -len(k[1]), k[1]


def _rel(p: Path) -> str:
    """A path relative to the container folder (stable across checkouts), else absolute."""
    p = Path(p).resolve()
    return str(p.relative_to(CONTAINER)) if p.is_relative_to(CONTAINER) else str(p)


def _ordered(sources) -> list[str]:
    return [s for s in T.SOURCES if s in sources]


# Sources ---------------------------------------------------------------------------------------------------------


def _concepts(items: list[dict]) -> list[dict]:
    return [{"field": c["field"], "text": c["text"], "type": c["type"], "icd_relevant": bool(c.get("icd_relevant"))} for c in items]


def load_golds(harness: Path, vm_dir: Path, ketai: Path) -> tuple[dict[str, dict], dict[str, dict[str, dict]], dict[str, dict]]:
    """(records, {source: {encounter_id: raw gold}}, inputs) of the eval split. Raw gold: `pii` items with the source's
    `label`, the harness `keep`, `concepts`, and ketai's `not_pii` ambiguous terms."""
    encs, labs = load_jsonl(harness / "encounters.jsonl"), load_jsonl(harness / "labels.jsonl")
    rows = [(e, lab) for e, lab in zip(encs, labs, strict=True) if lab["meta"].get("split") == "eval"]
    records = {e["encounter_id"]: e for e, _ in rows}
    golds: dict[str, dict[str, dict]] = {"harness": {e["encounter_id"]: {"pii": lab["pii"], "keep": lab["keep"]} for e, lab in rows}}
    inputs = {"harness": {"path": _rel(harness / "labels.jsonl"), "sha1": sha1_file(harness / "labels.jsonl")}}
    files = sorted(p for p in (vm_dir / f"{i}.json" for i in records) if p.exists())
    if files:
        anns = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}
        golds["vm"] = {eid: {"pii": [{"field": it["field"], "text": it["text"], "label": it["type"]} for it in a.get("phi", [])],
                             "concepts": _concepts(a.get("medical_concepts", []))} for eid, a in anns.items()}  # fmt: skip
        inputs["vm"] = {"path": _rel(vm_dir), "files": len(files), "sha1": sha1_text(*(p.name + "\0" + p.read_text(encoding="utf-8") for p in files))}
    else:
        print(f"warning: VM gold {vm_dir} not found, merging without it", file=sys.stderr)
    if ketai.exists():
        golds["ketai"] = {
            g["encounter_id"]: {"pii": [{"field": it["field"], "text": it["text"], "label": it["category"]} for it in g.get("pii", [])],
                                "concepts": _concepts(g.get("medical_concepts", [])),
                                "not_pii": [{"field": t["field"], "text": t["text"], "why": t.get("comment", "")}
                                            for t in g.get("ambiguous_terms", []) if not t.get("is_pii")]}
            for g in load_jsonl(ketai) if g["encounter_id"] in records
        }  # fmt: skip
        inputs["ketai"] = {"path": _rel(ketai), "sha1": sha1_file(ketai)}
    else:
        print(f"warning: ketai gold {ketai} not found, merging without it", file=sys.stderr)
    return records, golds, inputs


def source_pii(source: str, raw: list[dict], rec: dict, harness_dob: set[Key] | frozenset = frozenset()) -> list[dict]:
    """One gold's own PII list, as the suites scored it: usable, distinct per (field, lowercased text), categories
    harmonized. The harness list is exactly the old aci_eval `pii` (its categories and risks kept)."""
    if source == "harness":
        return distinct_items([p for p in raw if S._usable(p["text"])], "category", "risk", "label")
    items = []
    for it in raw:
        if it["field"] in FIELDS and S._usable(it["text"]):
            cat = T.category(source, it["label"], it["text"], rec[it["field"]], key(it) in harness_dob)
            items.append({"field": it["field"], "text": it["text"], "category": cat, "risk": CATEGORY_RISK[cat], "label": it["label"]})
    return distinct_items(items, "category", "risk", "label")


# Merging ---------------------------------------------------------------------------------------------------------


def nesting(rec: dict, by_source: dict[str, list[dict]], drop: set[Key] | frozenset = frozenset(),
            rejected: dict[Key, set[str]] | None = None) -> tuple[dict[Key, dict[str, dict]], dict[Key, list[Key]], dict[Key, set[str]]]:  # fmt: skip
    """(exact, within, support) of one encounter: each key's items per gold, the longer keys it is a whole-word
    substring of, and the golds that list it or a key inside it (minus those that `rejected` it). Items with no
    whole-word occurrence in their field and the keys in `drop` are left out."""
    rejected = rejected or {}
    exact: dict[Key, dict[str, dict]] = {}
    for src in T.SOURCES:
        for it in by_source.get(src, []):
            if key(it) not in drop and it["field"] in FIELDS and count(rec[it["field"]], it["text"]):
                exact.setdefault(key(it), {}).setdefault(src, it)
    within = {k: [a for a in exact if a[0] == k[0] and len(a[1]) > len(k[1]) and pattern(k[1]).search(a[1])] for k in exact}
    support = {k: set(v) for k, v in exact.items()}
    for k, outer in within.items():
        for a in outer:
            support[a] |= set(exact[k]) - rejected.get(a, set())
    return exact, within, support


def merge_pii(rec: dict, by_source: dict[str, list[dict]], drop: set[Key] | frozenset = frozenset(),
              rejected: dict[Key, set[str]] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """(merged items, absorbed nested items, category conflicts) of one encounter."""
    rejected = rejected or {}
    exact, within, support = nesting(rec, by_source, drop, rejected)
    surface = {k: v[next(s for s in T.PRIORITY if s in v)]["text"] for k, v in exact.items()}

    def standalone(k: Key) -> int:
        spans = [o for a in within[k] for o in occurrences(rec[k[0]], a[1])]
        return sum(1 for s, e in occurrences(rec[k[0]], k[1]) if not any(cs <= s and e <= ce for cs, ce in spans))

    items, absorbed, conflicts = [], [], []
    for k in sorted(exact, key=_order):
        n, outer = standalone(k), sorted(within[k], key=_order)
        if outer and not n:
            absorbed.append({"field": k[0], "text": surface[k], "sources": _ordered(exact[k]), "within": [surface[a] for a in outer]})
            continue
        srcs = set(exact[k]) | (support[k].union(*(support[a] for a in outer)) - rejected.get(k, set()))
        cats = {s: it["category"] for s, it in exact[k].items()}
        cat = T.majority(cats)
        if len(set(cats.values())) > 1:
            conflicts.append({"field": k[0], "text": surface[k], "votes": {s: cats[s] for s in _ordered(cats)}, "category": cat})
        item = {
            "field": k[0], "text": surface[k], "category": cat, "risk": CATEGORY_RISK[cat], "direct": T.is_direct(cat),
            "sources": _ordered(srcs), "votes": len(srcs), "nested_votes": len(srcs - set(exact[k])),
            "source_labels": {s: exact[k][s]["label"] for s in _ordered(exact[k])},
            "label": exact[k][next(s for s in T.PRIORITY if cats.get(s) == cat)]["label"],
        }  # fmt: skip
        inner = sorted((b for b in exact if k in within[b]), key=_order)
        if inner:
            item["contains"] = [surface[b] for b in inner]
        if outer:
            item["within"] = [surface[a] for a in outer]
        item["standalone_occurrences"] = n
        items.append(item)
    return items, absorbed, conflicts


def merge_concepts(rec: dict, by_source: dict[str, list[dict]]) -> list[dict]:
    """VM and ketai concepts, one per (field, lowercased text): type by vote, `icd_relevant` when any gold says so."""
    exact: dict[Key, dict[str, dict]] = {}
    for src in T.SOURCES:
        for c in by_source.get(src, []):
            if c["field"] in FIELDS and S._usable(c["text"]) and count(rec[c["field"]], c["text"]):
                exact.setdefault(key(c), {}).setdefault(src, c)
    out = []
    for k in sorted(exact, key=_order):
        v = exact[k]
        rel = [s for s in _ordered(v) if v[s].get("icd_relevant")]
        out.append({"field": k[0], "text": v[next(s for s in T.PRIORITY if s in v)]["text"],
                    "type": T.majority({s: T.concept_type(c["type"]) for s, c in v.items()}),
                    "icd_relevant": bool(rel), "icd_relevant_votes": len(rel), "sources": _ordered(v), "votes": len(v)})  # fmt: skip
    return out


def merge_keep(rec: dict, harness_keep: list[dict], ketai_terms: list[dict], concepts: list[dict]) -> list[dict]:
    """Harness keep items (as the suites used them) + ketai's non-PII ambiguous terms: EPONYM when the term is (inside)
    a concept, else NOT_A_NAME."""
    out = {key(it): {**it, "sources": ["harness"]} for it in distinct_items([k for k in harness_keep if S._usable(k["text"])], "category", "why")}
    for t in ketai_terms:
        if t["field"] not in FIELDS or not S._usable(t["text"]) or not count(rec[t["field"]], t["text"]):
            continue
        if key(t) in out:
            out[key(t)]["sources"] = _ordered({*out[key(t)]["sources"], "ketai"})
            continue
        eponym = any(c["field"] == t["field"] and pattern(t["text"]).search(c["text"]) for c in concepts)
        out[key(t)] = {"field": t["field"], "text": t["text"], "category": "EPONYM" if eponym else "NOT_A_NAME", "why": t["why"], "sources": ["ketai"]}
    return [out[k] for k in sorted(out, key=_order)]


def merge_encounter(rec: dict, raw: dict[str, dict]) -> tuple[dict, dict]:
    """(merged row, log) of one encounter. `raw`: {source: raw gold of this encounter}. A PII item that is also a keep
    item stays PII when at least two golds list it or a part of it themselves (inherited votes do not count)."""
    hdob = {key(p) for p in raw.get("harness", {}).get("pii", []) if p["category"] == "DOB"}
    lists = {s: source_pii(s, raw[s]["pii"], rec, hdob) for s in T.SOURCES if s in raw}
    concepts_all = merge_concepts(rec, {s: raw[s].get("concepts", []) for s in raw})
    keep = merge_keep(rec, raw.get("harness", {}).get("keep", []), raw.get("ketai", {}).get("not_pii", []), concepts_all)
    rejected: dict[Key, set[str]] = {}
    for k in keep:
        rejected.setdefault(key(k), set()).update(k["sources"])
    if "ketai" in raw:
        for it in (it for lst in lists.values() for it in lst if it["text"].lower() in T.KETAI_NOT_PII):
            rejected.setdefault(key(it), set()).add("ketai")
    items, absorbed, conflicts = merge_pii(rec, lists, rejected=rejected)
    own = nesting(rec, lists, rejected=rejected)[2]
    keep_keys = {key(k) for k in keep}
    clash = [it for it in items if key(it) in keep_keys]
    pii_wins = {key(it) for it in clash if len(own[key(it)]) >= 2}
    if len(pii_wins) < len(clash):
        items, absorbed, conflicts = merge_pii(rec, lists, {key(it) for it in clash} - pii_wins, rejected)
    keep = [k for k in keep if key(k) not in pii_wins]
    direct = {key(it) for it in items if it["direct"]}
    eid = rec["encounter_id"]
    log = {
        "absorbed": [{"encounter_id": eid, **a} for a in absorbed],
        "kept_nested": [{"encounter_id": eid, **{k: it[k] for k in ("field", "text", "category", "votes", "sources", "source_labels", "within")},
                         "standalone_occurrences": it["standalone_occurrences"]} for it in items if "within" in it],
        "conflicts": [{"encounter_id": eid, **c} for c in conflicts],
        "pii_keep": [{"encounter_id": eid, "field": it["field"], "text": it["text"], "votes": len(own[key(it)]), "sources": _ordered(own[key(it)]),
                      "winner": "pii" if key(it) in pii_wins else "keep"} for it in clash],
        "dropped_concepts": [{"encounter_id": eid, **{k: c[k] for k in ("field", "text", "type", "sources")}} for c in concepts_all if key(c) in direct],
    }  # fmt: skip
    row = {"encounter_id": eid, "pii": items, "pii_by_source": lists, "keep": keep, "concepts": [c for c in concepts_all if key(c) not in direct]}
    return row, log


def overlap(lists: dict[str, dict[str, list[dict]]]) -> dict:
    """Exact agreement of the golds' own lists, per (encounter, field, lowercased text), before nesting."""
    sets = {s: {(eid, *key(it)) for eid, items in by_eid.items() for it in items} for s, by_eid in lists.items()}
    votes = Counter(k for s in sets.values() for k in s)
    out = {"union": len(votes), "at_least_two": sum(v >= 2 for v in votes.values()), "all_three": sum(v >= 3 for v in votes.values()) if len(sets) == 3 else None}
    if {"harness", "ketai"} <= set(sets):
        h, kt = sets["harness"], sets["ketai"]
        inside = {k for k in h - kt if any(o[:2] == k[:2] and len(o[2]) > len(k[2]) and pattern(k[2]).search(o[2]) for o in kt)}
        only = h - set().union(*(v for s, v in sets.items() if s != "harness"))
        out |= {"harness_not_in_ketai": len(h - kt), "harness_not_in_ketai_inside_a_ketai_item": len(inside),
                "harness_only": len(only), "harness_only_inside_a_ketai_item": len(only & inside)}  # fmt: skip
    return out


def merge(records: dict[str, dict], golds: dict[str, dict[str, dict]]) -> tuple[list[dict], dict, dict]:
    """(rows, stats, log) of every encounter in `records`, in encounter_id order."""
    rows, log = [], {}
    for eid in sorted(records):
        row, lg = merge_encounter(records[eid], {s: g[eid] for s, g in golds.items() if eid in g})
        rows.append(row)
        for k, v in lg.items():
            log.setdefault(k, []).extend(v)
    pii = [(r["encounter_id"], it) for r in rows for it in r["pii"]]
    occ = {lvl: sum(count(records[eid][it["field"]], it["text"]) for eid, it in pii if T.in_level(it, lvl)) for lvl in T.LEVELS}
    by_cat: dict[str, Counter] = {}
    for _, it in pii:
        by_cat.setdefault(it["category"], Counter())[str(it["votes"])] += 1
    concepts, keep = [c for r in rows for c in r["concepts"]], [k for r in rows for k in r["keep"]]
    lists = {s: {r["encounter_id"]: r["pii_by_source"][s] for r in rows if s in r["pii_by_source"]} for s in T.SOURCES if s in golds}
    size = {"harness": ("keep_items", lambda g: len(distinct_items([k for k in g["keep"] if S._usable(k["text"])], "category"))),
            "vm": ("concepts", lambda g: len(g["concepts"])), "ketai": ("concepts", lambda g: len(g["concepts"]))}  # fmt: skip
    sources = {s: {"encounters": len(lists[s]), "pii_items": sum(map(len, lists[s].values())), size[s][0]: sum(map(size[s][1], golds[s].values()))} for s in lists}
    if "ketai" in sources:
        sources["ketai"]["not_pii_terms"] = sum(len(g["not_pii"]) for g in golds["ketai"].values())
    stats = {
        "suite": SUITE,
        "encounters": len(rows),
        "sources": sources,
        "exact_overlap": overlap(lists),
        "pii": {
            "items": len(pii),
            "levels": {lvl: sum(T.in_level(it, lvl) for _, it in pii) for lvl in T.LEVELS},
            "occurrences": occ,
            "votes": dict(sorted(Counter(str(it["votes"]) for _, it in pii).items())),
            "with_nested_votes": sum(it["nested_votes"] > 0 for _, it in pii),
            "containers": sum("contains" in it for _, it in pii),
            "absorbed": len(log["absorbed"]),
            "kept_nested": len(log["kept_nested"]),
            "category_conflicts": len(log["conflicts"]),
            "by_category": {c: {"items": sum(v.values()), "votes": dict(sorted(v.items())), "direct": T.is_direct(c)} for c, v in sorted(by_cat.items())},
        },
        "keep": {"items": len(keep), "by_source": dict(sorted(Counter("+".join(k["sources"]) for k in keep).items())),
                 "pii_wins": sum(x["winner"] == "pii" for x in log["pii_keep"]), "keep_wins": sum(x["winner"] == "keep" for x in log["pii_keep"])},
        "concepts": {"items": len(concepts), "by_source": dict(sorted(Counter("+".join(c["sources"]) for c in concepts).items())),
                     "icd_relevant": sum(c["icd_relevant"] for c in concepts), "dropped_direct_pii": len(log["dropped_concepts"]),
                     "by_type": dict(sorted(Counter(c["type"] for c in concepts).items()))},
    }  # fmt: skip
    return rows, stats, log


# Output ------------------------------------------------------------------------------------------------------------


def md_table(header: list[str], rows: list[list]) -> list[str]:
    cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows]


def review(stats: dict, log: dict) -> str:
    st, ov, q = stats["pii"], stats["exact_overlap"], lambda t: f"`{t}`"
    md = ["# Merged ACI-eval gold: review", "",
          "Written by `python3 -I eval gold` (eval/gold/merge_aci_gold.py). It quotes gold strings: keep it under data/eval/.", "",
          f"- encounters: {stats['encounters']}; " + ", ".join(f"{s} {v['pii_items']} PII items" for s, v in stats["sources"].items()),
          f"- exact overlap before nesting: all three {ov['all_three']}, at least two {ov['at_least_two']}, union {ov['union']}; "
          f"harness items not in ketai {ov.get('harness_not_in_ketai')}, {ov.get('harness_not_in_ketai_inside_a_ketai_item')} of them inside a ketai item",
          f"- merged: {st['items']} items (union), {st['levels']['consensus']} consensus, {st['levels']['direct']} direct; votes {st['votes']}; "
          f"{st['absorbed']} nested items absorbed, {st['kept_nested']} kept for their standalone occurrences",
          f"- keep: {stats['keep']['items']} items ({stats['keep']['pii_wins']} PII/keep conflicts won by PII, {stats['keep']['keep_wins']} by keep); "
          f"concepts: {stats['concepts']['items']} ({stats['concepts']['dropped_direct_pii']} dropped as direct PII)", ""]  # fmt: skip
    sections = [
        ("Category conflicts (majority, tie: harness > ketai > VM)", ["encounter", "field", "text", "votes", "category"],
         [[c["encounter_id"], c["field"], q(c["text"]), ", ".join(f"{s}: {v}" for s, v in c["votes"].items()), c["category"]] for c in log["conflicts"]]),
        ("PII that is also a keep item (PII with two own votes, else keep)", ["encounter", "field", "text", "own votes", "sources", "winner"],
         [[x["encounter_id"], x["field"], q(x["text"]), x["votes"], "+".join(x["sources"]), x["winner"]] for x in log["pii_keep"]]),
        ("Nested items kept for their standalone occurrences (votes inherited from the containers)",
         ["encounter", "field", "text", "category", "votes", "own labels", "within", "standalone"],
         [[x["encounter_id"], x["field"], q(x["text"]), x["category"], f"{x['votes']} ({'+'.join(x['sources'])})",
           ", ".join(f"{s}: {v}" for s, v in x["source_labels"].items()), "; ".join(map(q, x["within"])), x["standalone_occurrences"]] for x in log["kept_nested"]]),
        ("Concepts dropped because they equal a direct PII item", ["encounter", "field", "text", "type", "sources"],
         [[c["encounter_id"], c["field"], q(c["text"]), c["type"], "+".join(c["sources"])] for c in log["dropped_concepts"]]),
        ("Nested items absorbed by a longer item (no standalone occurrence)", ["encounter", "field", "text", "sources", "within"],
         [[a["encounter_id"], a["field"], q(a["text"]), "+".join(a["sources"]), "; ".join(map(q, a["within"]))] for a in log["absorbed"]]),
    ]  # fmt: skip
    for title, header, rows in sections:
        md += [f"## {title} ({len(rows)})", ""] + (md_table(header, rows) if rows else ["none"]) + [""]
    return "\n".join(md)


def write(out_dir: Path, rows: list[dict], stats: dict, log: dict) -> dict:
    """Write the files; returns the stats with `merged_sha1`. A changed merged.jsonl first moves to merged.prev.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path, text = out_dir / "merged.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        path.replace(out_dir / "merged.prev.jsonl")
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    stats = {**stats, "merged_sha1": sha1_text(text)}
    write_json(out_dir / "merged_stats.json", stats)
    (out_dir / "merged_review.md").write_text(review(stats, log), encoding="utf-8")
    return stats


def run(harness: Path | None = None, vm_dir: Path | None = None, ketai: Path | None = None, out_dir: Path | None = None) -> dict | None:
    """Merge and write; returns the stats, or None (with a warning) when the harness source is missing."""
    harness, vm_dir, ketai, out_dir = harness or HARNESS, vm_dir or VM_DIR, ketai or KETAI, out_dir or OUT_DIR
    if not all((harness / f).exists() for f in ("encounters.jsonl", "labels.jsonl")):
        print(f"warning: {harness} is missing (scripts/convert/convert_all.py), the merged gold is not built", file=sys.stderr)
        return None
    records, golds, inputs = load_golds(harness, vm_dir, ketai)
    rows, stats, log = merge(records, golds)
    return write(out_dir, rows, {**stats, "inputs": inputs}, log)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval gold", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit", action="store_true", help="also run the gold audit (eval/gold/audit_gold.py): data/eval/gold/aci_eval/audit.md")
    a = p.parse_args(argv)
    stats = run()
    if stats is None:
        return 0
    st, ov = stats["pii"], stats["exact_overlap"]
    print(f"gold {SUITE}: {stats['encounters']} encounters, golds {'+'.join(stats['sources'])}; exact overlap: all three {ov['all_three']}, "
          f">=2 {ov['at_least_two']}, union {ov['union']}")
    print(f"  merged PII {st['items']} items: consensus {st['levels']['consensus']}, direct {st['levels']['direct']}; votes {st['votes']}; "
          f"{st['absorbed']} nested absorbed; keep {stats['keep']['items']}; concepts {stats['concepts']['items']}")
    print(f"  {OUT_DIR / 'merged.jsonl'} (sha1 {stats['merged_sha1'][:12]})")
    if a.audit:
        from eval.gold import audit_gold

        return audit_gold.main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
