"""Adapter for the anonymous model (vm-backup/neuralwave): run its detection layers and replace what they find.

Usage: python -I eval/adapters/anonymous_vm.py INPUT OUTPUT [--layers regex,names,deid,gliner] [--tiers 1,2] [--device auto]
                                                      [--root VM_DIR] [--words FILE]

The VM code stops at detection: its policy/ and guard/ folders are empty and the Qwen fine-tune never finished,
so this adapter adds the smallest policy that makes the detectors scorable, and nothing else:

1. Run the chosen layers unchanged: regex (dates, ages, IDs, contacts, holidays), names (structural cues,
   propagated to every occurrence in both fields), deid_roberta (i2b2) and GLiNER (zero-shot PII labels).
2. Merge overlapping spans per field; the merged span takes the type of its most confident member.
3. Replace each merged span of an enabled tier with `[TYPE]`. icd10_codes and every other key stay as they are.
4. Log one decision per merged span in `<output stem>.decisions.jsonl` (the explainability sidecar): its offsets and
   text in the original field, `remove` (or `keep` for a tier not replaced), the type, the reason (each member's
   detector reason, layer and confidence) and the layers that found it.

The names layer reads /usr/share/dict/american-english-large (Debian `wamerican-large`) to skip common words.
Without that file every capitalized-looking word is a name candidate; `--words` points to another list.
Run it with a Python that has `regex` (and torch, transformers, gliner for the ML layers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "vm-backup" / "neuralwave"  # <container>/neuralwave/eval/adapters/
FIELDS = ("dialogue", "note")


def _load_words(path: Path) -> tuple[set[str], set[str]]:
    lower, proper = set(), set()
    for w in path.open(encoding="utf-8", errors="ignore"):
        w = w.strip()
        if w and "'" not in w:
            (lower if w[0].islower() else proper).add(w.lower())
    return lower, proper


def merge(spans: list) -> list[tuple[int, int, str, int, list]]:
    """Union of overlapping spans as (start, end, type, tier, members), type of the most confident (then longest) member."""
    out: list[list] = []
    for s in sorted(spans, key=lambda s: (s.start, -s.end)):
        if out and s.start < out[-1][1]:
            cur = out[-1]
            cur[1] = max(cur[1], s.end)
            cur[2].append(s)
        else:
            out.append([s.start, s.end, [s]])
    merged = []
    for start, end, members in out:
        best = max(members, key=lambda m: (m.confidence, m.end - m.start))
        merged.append((start, end, best.type, min(m.tier for m in members), members))
    return merged


def decision(eid: str, field: str, text: str, span: tuple[int, int, str, int, list], replaced: bool) -> dict:
    """The explainability row of one merged span: why it was replaced (or kept), from its members' detector reasons."""
    start, end, typ, tier, members = span
    why = dict.fromkeys(f"{m.reason or f'{m.source} found {m.type}'} [{m.source}, conf {m.confidence:.2f}]" for m in members)
    head = f"tier {tier} {typ} -> [{typ}]" if replaced else f"tier {tier} {typ} kept (tier not replaced)"
    return {"encounter_id": eid, "field": field, "start": start, "end": end, "text": text[start:end], "action": "remove" if replaced else "keep",
            "category": typ, "reason": f"{head}: {'; '.join(why)}", "source": "+".join(sorted({m.source for m in members}))}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--layers", default="regex,names,deid,gliner")
    p.add_argument("--tiers", default="1,2", help="tiers to replace: 1 = direct identifiers, 2 = quasi-identifiers")
    p.add_argument("--device", default="auto", help="ML layers: auto (cuda if available) | cpu | cuda")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the VM project folder")
    p.add_argument("--words", type=Path, default=Path("/usr/share/dict/american-english-large"))
    a = p.parse_args(argv)
    layers = set(a.layers.split(","))
    tiers = {int(t) for t in a.tiers.split(",")}

    if a.device == "auto" and not layers & {"deid", "gliner"}:
        a.device = "cpu"
    elif a.device == "auto":
        try:
            import torch

            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            a.device = "cpu"
    os.environ["NW_DEVICE"] = a.device
    sys.path.insert(0, str(a.root))
    from detect import names, regex_layer  # VM code

    if a.words.exists():
        words = _load_words(a.words)
        names._dictionary.cache_clear()
        names._dictionary = lambda: words  # noqa: E731 - same contract as the VM's reader
    else:
        print(f"warning: {a.words} not found, the names layer runs without its common-word filter", file=sys.stderr)
    if layers & {"deid", "gliner"}:
        from detect import ml_layers

    recs = [json.loads(line) for line in a.input.open(encoding="utf-8") if line.strip()]
    n_spans = 0
    sidecar = a.output.with_name(a.output.stem + ".decisions.jsonl")
    with a.output.open("w", encoding="utf-8") as out, sidecar.open("w", encoding="utf-8") as dec:
        for i, rec in enumerate(recs):
            spans = []
            if "regex" in layers:
                spans += [s for f in FIELDS for s in regex_layer.detect(f, rec[f])]
            if "names" in layers:
                spans += names.detect(rec)[0]
            if "deid" in layers:
                spans += [s for f in FIELDS if rec[f] for s in ml_layers.deid_detect(f, rec[f])]
            if "gliner" in layers:
                spans += [s for f in FIELDS if rec[f] for s in ml_layers.gliner_detect(f, rec[f])]
            new = dict(rec)
            for f in FIELDS:
                text = rec[f]
                merged = merge([s for s in spans if s.field == f])
                for start, end, typ, tier, _ in reversed(merged):
                    if tier in tiers:
                        text = text[:start] + f"[{typ}]" + text[end:]
                        n_spans += 1
                new[f] = text
                for m in merged:
                    dec.write(json.dumps(decision(rec["encounter_id"], f, rec[f], m, m[3] in tiers), ensure_ascii=False) + "\n")
            out.write(json.dumps(new, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(recs)} records, {n_spans} spans replaced", file=sys.stderr, flush=True)
    print(f"done: {len(recs)} records, {n_spans} spans replaced, layers={sorted(layers)}, device={a.device}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
