"""Export converted gold in the evaluation harness's text-based format: one `<encounter_id>.json` per encounter.

Usage: python3 -I scripts/convert/export_gold.py data/converted/<name> [...] [--out DIR]

Each encounter whose `pii_gold` is not `none` gets a file
    {"encounter_id": ..., "pii": [{field, text, category, risk}], "keep": [{field, text, category, why}]}
with the spans reduced to distinct strings per field (case-insensitive), longest first. The harness counts every
whole-word occurrence of each string, and aci_bench.py imports the same format with `--gold-dir`, so exporting
aci_bench and importing it again gives the same labels. The files don't say whether the gold is `partial`: on
those datasets read only leak recall, not over-removal.

Output goes to `<dataset>/gold/`, or `<DIR>/<name>/` with `--out`. Earlier `*.json` files there are replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def distinct(spans: list[dict], extra: str) -> list[dict]:
    """One item per (field, lowercased text). Longest first, so an importer adds full names before their tokens."""
    out, seen = [], set()
    for it in sorted(spans, key=lambda it: -len(it["text"])):
        key = (it["field"], it["text"].lower())
        if key not in seen:
            seen.add(key)
            out.append({"field": it["field"], "text": it["text"], "category": it["category"], extra: it[extra]})
    return out


def export(d: Path, out: Path) -> tuple[int, int]:
    """Write the gold files of dataset folder `d` into `out`; returns (written, skipped for `pii_gold: none`)."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    written = skipped = 0
    for line in (d / "labels.jsonl").open(encoding="utf-8"):
        lab = json.loads(line)
        eid = lab["encounter_id"]
        if lab["pii_gold"] == "none":
            skipped += 1
            continue
        if "/" in eid or eid.startswith("."):
            raise ValueError(f"{eid}: not usable as a file name")
        gold = {"encounter_id": eid, "pii": distinct(lab["pii"], "risk"), "keep": distinct(lab["keep"], "why")}
        (out / f"{eid}.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("datasets", type=Path, nargs="+", help="converted dataset folders, e.g. data/converted/devp_discharge")
    p.add_argument("--out", type=Path, help="output root; each dataset goes to <out>/<name>/ (default: <dataset>/gold/)")
    a = p.parse_args(argv)
    for d in a.datasets:
        out = a.out / d.name if a.out else d / "gold"
        written, skipped = export(d, out)
        print(f"{d.name}: {written} gold files, {skipped} encounters without privacy gold skipped -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
