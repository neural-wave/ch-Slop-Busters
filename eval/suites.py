"""Build the evaluation suites: fixed, sampled subsets of `data/converted/*` that every model is run on.

Usage: python3 -I eval suites [--only NAME ...] [--vm-annotations DIR] [--no-merged | --merged] [--if-stale]

Each suite goes to `data/eval/suites/<name>/`:

    encounters.jsonl  model input: exactly the challenge schema (labels never go here)
    labels.jsonl      gold per encounter: `pii` and `keep` as distinct strings, optional `pii_by_source`, `pii_alt`
                      and `concepts`
    suite.json        definition, gold provenance, counts, the sha1 of both files (`input_sha1`: runs are cached
                      against it, `labels_sha1`), of everything the suite is built from (`source_sha1`) and of the
                      build code, this file and common.py (`code_sha1`)

`eval suites` rebuilds every selected suite; with `--if-stale` (what `eval run` and `eval all` do) only a suite whose
suite.json is missing, whose definition, source or build code changed, or whose files no longer match their sha1.
Building is deterministic, so an unchanged source gives byte-identical files and cached runs stay valid.

Gold strings are scored by counting their whole-word occurrences (common.pattern), the harness's rule, so offsets
are not needed and a model may rewrite freely. `pii_gold` tells how complete the PII list is: on `partial`
suites only leak recall is meaningful, not over-removal.

A suite with a merged gold (data/eval/gold/<name>/merged.jsonl, written by `eval gold`: today aci_eval) takes its
`pii`, `pii_by_source`, `keep` and `concepts` from it. `pii` then holds every merged item of the three ACI golds
(harness, VM, ketai) with its votes, so the scorer can report the union, consensus and direct levels, and
`pii_by_source` holds each gold's own list (eval/docs/GOLD.md). The merged file is part of `source_sha1`, so the suite
rebuilds when the gold changes. `--no-merged` falls back to the old ACI labels: `pii` is the harness gold (made by the
A+D team's annotators, imported by scripts/convert/aci_bench.py), `pii_alt` the VM teacher annotation
(models/anonymous/annotations/eval, made for the anonymous model) and `concepts` its medical concepts. suite.json
records `--no-merged` (`merged_gold: false`) and keeps it until `--merged` is given. A suite built with
`--vm-annotations DIR` likewise records DIR (`vm_annotations`) and keeps it on every rebuild until another
`--vm-annotations` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import CATEGORY_RISK, CONTAINER, REPO, CONVERTED, ENCOUNTER_KEYS, OUT, SUITES_DIR, distinct_items, load_jsonl, sha1_file, write_json, write_jsonl
from eval.gold.taxonomy import LEVELS, SOURCES, in_level

VM_ANNOTATIONS = REPO / "models" / "anonymous" / "annotations" / "eval"
GOLD_DIR = OUT / "gold"  # merged golds: <suite>/merged.jsonl (eval/gold/merge_aci_gold.py)

# name -> definition. `source`: data/converted/<source>; `where`/`where_not`: filters on labels.meta;
# `pii_gold`: allowed per-record gold levels; `limit`: deterministic sample size (seeded, original order kept).
SUITES: dict[str, dict] = {
    "aci_eval": {"source": "aci_bench", "where": {"split": "eval"}, "pii_gold": ["exhaustive"], "role": "main",
                 "about": "challenge eval split (67), merged exhaustive gold of three golds (harness, VM, ketai) + their medical concepts, gold ICD-10"},
    "aci_dev": {"source": "aci_bench", "where": {"split": "train"}, "pii_gold": ["exhaustive"], "role": "main",
                "about": "40 train encounters with harness gold (the A+D team's dev set, so A+D is tuned on it)"},
    "devp_discharge": {"source": "devp_discharge", "role": "kaggle",
                       "about": "synthetic discharge summaries, exhaustive gold, 5 gold ICD-10 codes (A+D developed on this source: their robust_discharge)"},
    "imerit_en": {"source": "imerit_scribe", "where": {"language": "en"}, "role": "kaggle",
                  "about": "English scribe transcripts + notes, partial gold (patient, physician, DOB)"},
    "imerit_multi": {"source": "imerit_scribe", "where_not": {"language": "en"}, "limit": 60, "role": "kaggle",
                     "about": "DE/FR/ES/NL scribe transcripts + notes, partial gold (robustness)"},
    "sadiqguru": {"source": "sadiqguru_conversations", "role": "kaggle",
                  "about": "4 long English dialogues, hand-labelled names, years, ages"},
    "nbme_notes": {"source": "nbme_notes", "pii_gold": ["partial"], "limit": 100, "role": "kaggle",
                   "about": "USMLE exam notes, partial gold (patient age), graded findings as must-keep gold"},
    "pmc_patients": {"source": "pmc_patients", "limit": 100, "role": "kaggle",
                     "about": "published case reports, partial gold (patient age)"},
    "persuade_pii": {"source": "persuade_pii", "limit": 100, "role": "generic",
                     "about": "student essays with inserted PII (names), exhaustive, not medical"},
    "pii_gpt": {"source": "pii_gpt", "limit": 100, "role": "generic",
                "about": "LLM essays with injected names, IDs, e-mails, phones, addresses, partial, not medical"},
    "robust_discharge": {"source": "robust_discharge", "role": "robust",
                         "about": "R1: devp discharge summaries as lowercase dictation (the layout devp_discharge lacks), injected dates, age, physician, phone; exhaustive"},
    "robust_medconv": {"source": "robust_medconv", "role": "robust",
                       "about": "R2: cased web medical Q&A as dialogue with injected patient PII, partial (the posts' own PII is unlabelled), no codes"},
    "robust_synthetic": {"source": "robust_synthetic", "role": "robust",
                         "about": "R3: 24 synthetic hospital documents (triage, letters, e-mails, handoffs, reports, phone calls), exhaustive, hand-assigned codes"},
}  # fmt: skip
SEED = 13
BUILD_CODE = (Path(__file__), Path(__file__).with_name("common.py"), Path(__file__).parent / "gold" / "taxonomy.py")
CODE_SHA1 = hashlib.sha1(b"\x00".join(p.read_bytes() for p in BUILD_CODE)).hexdigest()  # the build code

# VM annotation types -> common categories (risk then follows CATEGORY_RISK).
VM_TYPES = {
    "NAME_PATIENT": "PATIENT_NAME", "NAME_RELATIVE": "PERSON_NAME", "NAME_CLINICIAN": "PERSON_NAME",
    "NAME_OTHER": "PERSON_NAME", "AGE": "AGE", "DATE": "DATE", "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION",
    "CONTACT": "ID", "ID": "ID", "OCCUPATION": "OCCUPATION", "ACTIVITY": "OTHER_QI", "LIFE_EVENT": "OTHER_QI",
    "TEMPORAL": "OTHER_QI", "PERSONAL_DETAIL": "OTHER_QI",
}  # fmt: skip


def _matches(meta: dict, where: dict | None, where_not: dict | None) -> bool:
    if where and any(meta.get(k) != v for k, v in where.items()):
        return False
    return not (where_not and any(meta.get(k) == v for k, v in where_not.items()))


def _usable(text: str) -> bool:
    """Single characters (`M` for male in NBME) match too much to be scored as gold strings."""
    return sum(c.isalnum() for c in text) >= 2


def _vm_gold(eid: str, ann_dir: Path) -> tuple[list[dict] | None, list[dict] | None]:
    p = ann_dir / f"{eid}.json"
    if not p.exists():
        return None, None
    ann = json.loads(p.read_text(encoding="utf-8"))
    pii = []
    for it in ann.get("phi", []):
        cat = VM_TYPES.get(it["type"], "OTHER_QI")
        pii.append({"field": it["field"], "text": it["text"], "category": cat, "risk": CATEGORY_RISK[cat], "label": it["type"]})
    concepts = [{"field": c["field"], "text": c["text"], "type": c["type"], "icd_relevant": bool(c.get("icd_relevant"))} for c in ann.get("medical_concepts", [])]
    return distinct_items([p for p in pii if _usable(p["text"])], "category", "risk", "label"), distinct_items([c for c in concepts if _usable(c["text"])], "type", "icd_relevant")


SOURCE_FILES = ("encounters.jsonl", "labels.jsonl", "manifest.json")


def merged_gold(name: str, info: dict | None = None, use: bool | None = None) -> Path | None:
    """The suite's merged gold when it exists and is wanted: `use` (--merged/--no-merged), else what suite.json recorded."""
    p = GOLD_DIR / name / "merged.jsonl"
    return p if (use if use is not None else (info or {}).get("merged_gold", True)) and p.exists() else None


def source_sha1(spec: dict, vm_dir: Path, ids: Iterable[str], merged: Path | None = None) -> str | None:
    """sha1 of what a suite is built from: the converted source, for ACI the VM annotations of its records, and the
    merged gold it uses. None when the converted source is missing."""
    src = CONVERTED / spec["source"]
    files = [src / n for n in SOURCE_FILES]
    if not all(p.exists() for p in files):
        return None
    if spec["source"] == "aci_bench":
        files += sorted(p for p in {vm_dir / f"{i}.json" for i in ids} if p.exists())
    if merged:
        files.append(merged)
    h = hashlib.sha1()
    for p in files:
        h.update(p.name.encode("utf-8") + b"\x00" + p.read_bytes() + b"\x00")
    return h.hexdigest()


def build(name: str, spec: dict, vm_dir: Path, use_merged: bool = True) -> dict:
    src = CONVERTED / spec["source"]
    merged = merged_gold(name, use=use_merged)
    by_eid = {g["encounter_id"]: g for g in load_jsonl(merged)} if merged else {}
    if spec["source"] == "aci_bench" and not merged and not vm_dir.is_dir():
        print(f"warning: {name}: VM annotations {vm_dir} not found, the suite gets no pii_alt and no concepts", file=sys.stderr)
    encs, labs = load_jsonl(src / "encounters.jsonl"), load_jsonl(src / "labels.jsonl")
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    rows = [(e, lab) for e, lab in zip(encs, labs, strict=True)
            if _matches(lab["meta"], spec.get("where"), spec.get("where_not")) and lab["pii_gold"] in spec.get("pii_gold", ["exhaustive", "partial"])]  # fmt: skip
    if spec.get("limit") and len(rows) > spec["limit"]:
        keep = set(random.Random(SEED).sample(range(len(rows)), spec["limit"]))
        rows = [r for i, r in enumerate(rows) if i in keep]
    out = SUITES_DIR / name
    labels, n_alt = [], 0
    for e, lab in rows:
        gold = {
            "encounter_id": e["encounter_id"],
            "pii_gold": lab["pii_gold"],
            "icd10_source": lab["icd10_source"],
            "language": lab["meta"].get("language", "en"),
            "pii": distinct_items([p for p in lab["pii"] if _usable(p["text"])], "category", "risk", "label"),
            "keep": distinct_items([k for k in lab["keep"] if _usable(k["text"])], "category", "why"),
        }
        m = by_eid.get(e["encounter_id"])
        if m is not None:  # merged gold: every item with its votes, each gold's own list, merged keep and concepts
            gold |= {"pii": [p for p in m["pii"] if _usable(p["text"])], "keep": m["keep"], "pii_by_source": m["pii_by_source"],
                     "concepts": m["concepts"]}  # fmt: skip
        elif merged:
            print(f"warning: {name}: {e['encounter_id']} is not in {merged}, it keeps the converted gold", file=sys.stderr)
        if m is None and spec["source"] == "aci_bench":
            alt, concepts = _vm_gold(e["encounter_id"], vm_dir)
            if alt is not None:
                gold["pii_alt"], n_alt = alt, n_alt + 1
                if concepts:
                    gold["concepts"] = concepts
        labels.append(gold)
    write_jsonl(out / "encounters.jsonl", ({k: e[k] for k in ENCOUNTER_KEYS} for e, _ in rows))
    write_jsonl(out / "labels.jsonl", labels)
    info = {
        "name": name,
        **{k: v for k, v in spec.items() if k != "about"},
        "about": spec["about"],
        "seed": SEED,
        "license": manifest.get("license"),
        "language": manifest.get("language"),
        "records": len(rows),
        "pii_gold": sorted({g["pii_gold"] for g in labels}),
        "pii_items": sum(len(g["pii"]) for g in labels),
        "keep_items": sum(len(g["keep"]) for g in labels),
        "pii_alt_records": n_alt,
        "gold_sources": [s for s in SOURCES if any(s in g.get("pii_by_source", {}) for g in labels)] if by_eid else ["harness"] + ["vm"] * bool(n_alt),
        "pii_levels": {lvl: sum(in_level(it, lvl) for g in labels for it in g["pii"]) for lvl in LEVELS},
        "concept_items": sum(len(g.get("concepts", [])) for g in labels),
        "with_icd10": sum(1 for e, _ in rows if e["icd10_codes"]),
        "input_sha1": sha1_file(out / "encounters.jsonl"),
        "labels_sha1": sha1_file(out / "labels.jsonl"),
        "source_sha1": source_sha1(spec, vm_dir, (e["encounter_id"] for e, _ in rows), merged),
        "code_sha1": CODE_SHA1,
    }
    if merged:
        info["merged_sha1"] = sha1_file(merged)
    if not use_merged:
        info["merged_gold"] = False
    if spec["source"] == "aci_bench" and vm_dir.resolve() != VM_ANNOTATIONS.resolve():
        info["vm_annotations"] = str(vm_dir.resolve())
    write_json(out / "suite.json", info)
    return info


def _info(name: str) -> dict | None:
    p = SUITES_DIR / name / "suite.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _vm_dir(info: dict | None) -> Path:
    """The VM annotations the suite was built with: recorded in suite.json when not the default."""
    return Path(info["vm_annotations"]) if info and info.get("vm_annotations") else VM_ANNOTATIONS


def stale(name: str, info: dict | None, vm_dir: Path | None = None, use_merged: bool | None = None) -> str:
    """Why the suite must be rebuilt, or "" when it is up to date. `vm_dir` and `use_merged` default to what it was built with."""
    spec, out = SUITES[name], SUITES_DIR / name
    enc, lab = out / "encounters.jsonl", out / "labels.jsonl"
    if not info or not info.get("source_sha1") or info.get("seed") != SEED:
        return "no suite.json" if not info else "suite.json of an older build"
    if any(info.get(k) != spec.get(k) for k in ("source", "where", "where_not", "limit", "role", "about")):  # pii_gold: levels found
        return "definition changed"
    if info.get("code_sha1") != CODE_SHA1:
        return "build code changed"
    if not (enc.exists() and lab.exists()) or info.get("input_sha1") != sha1_file(enc) or info.get("labels_sha1") != sha1_file(lab):
        return "suite files changed"
    if use_merged is not None and use_merged != info.get("merged_gold", True):
        return "merged gold switched " + ("on" if use_merged else "off")
    ids = (r["encounter_id"] for r in load_jsonl(enc))
    if info["source_sha1"] != source_sha1(spec, vm_dir or _vm_dir(info), ids, merged_gold(name, info, use_merged)):
        return "source changed"
    return ""


def ensure(names: list[str], vm_dir: Path | None = None, force: bool = False, use_merged: bool | None = None) -> dict[str, dict]:
    """Build missing or stale suites (all with `force`); returns the suite.json of every suite that exists.

    Without its converted source a suite is kept as it is (with a warning), or left out when it was never built.
    `use_merged` None keeps what the suite was built with (the merged gold by default)."""
    out = {}
    for n in names:
        info = _info(n)
        if not all((CONVERTED / SUITES[n]["source"] / f).exists() for f in SOURCE_FILES):
            kept = "keeping the existing suite" if info else "suite skipped"
            print(f"warning: {n}: data/converted/{SUITES[n]['source']} is missing (scripts/convert/convert_all.py), {kept}", file=sys.stderr)
            if info:
                out[n] = info
            continue
        why = "" if force else stale(n, info, vm_dir, use_merged)
        if why:
            print(f"suite {n}: rebuilt ({why})")
        use = use_merged if use_merged is not None else (info or {}).get("merged_gold", True)
        out[n] = build(n, SUITES[n], vm_dir or _vm_dir(info), use) if force or why else info
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval suites", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", "--suites", dest="only", nargs="*", default=list(SUITES), choices=list(SUITES), help="suites to build (default: all)")
    p.add_argument("--vm-annotations", type=Path, help="VM teacher annotations for the ACI eval split (default: the dir the suite was built with, "
                   "else models/anonymous/annotations/eval)")  # fmt: skip
    p.add_argument("--merged", action=argparse.BooleanOptionalAction, help="labels from the merged gold data/eval/gold/<suite>/merged.jsonl when it "
                   "exists (default: what the suite was built with, else yes); --no-merged: the old harness + VM labels")  # fmt: skip
    p.add_argument("--if-stale", action="store_true", help="build only missing or stale suites (default: rebuild every selected suite)")
    a = p.parse_args(argv)
    for n, info in ensure(a.only, a.vm_annotations, force=not a.if_stale, use_merged=a.merged).items():
        print(f"{n:15s} {info['records']:4d} records  pii items {info['pii_items']:5d} ({'/'.join(info['pii_gold'])})"
              f"  keep {info['keep_items']:4d}  gold {'+'.join(info.get('gold_sources', [])):17s}  concepts {info['concept_items']:5d}  icd {info['with_icd10']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
