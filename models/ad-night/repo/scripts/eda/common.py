"""Shared helpers for the EDA scripts (stdlib only).

Every script does `from common import *` and works on the two read-only
JSONL files under resources/data/.  Nothing here mutates the data.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "resources" / "data"
OUT_DIR = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": DATA_DIR / "patient_doctor_encounters.jsonl",
    "eval": DATA_DIR / "eval_patient_doctor_encounters.jsonl",
}
SPLITS = ("train", "eval")
DATASETS = ("aci", "virtassist", "virtscribe")


def load(split: str) -> list[dict]:
    """Load one split as a list of dicts, adding a `split` key."""
    recs = []
    with FILES[split].open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["split"] = split
            recs.append(r)
    return recs


def load_all() -> list[dict]:
    """Train + eval, in file order."""
    return load("train") + load("eval")


def by_split(recs: list[dict]) -> dict[str, list[dict]]:
    out = {s: [r for r in recs if r["split"] == s] for s in SPLITS}
    out["all"] = list(recs)
    return out


def by_dataset(recs: list[dict]) -> dict[str, list[dict]]:
    return {d: [r for r in recs if r["dataset"] == d] for d in DATASETS}


# ---------- dialogue helpers -------------------------------------------------

SPEAKER_RE = re.compile(r"^\[([^\]]*)\]\s?")


def dialogue_lines(dialogue: str) -> list[tuple[str, str, int]]:
    """Return [(speaker_tag_or_None, text, char_offset_of_line_start)]."""
    out, pos = [], 0
    for line in dialogue.split("\n"):
        m = SPEAKER_RE.match(line)
        if m:
            out.append((m.group(1), line[m.end():], pos))
        else:
            out.append((None, line, pos))
        pos += len(line) + 1
    return out


# ---------- note helpers ----------------------------------------------------

# A header line: whole line is upper-case letters / spaces / a few symbols,
# at least 3 letters, optional trailing colon.
HEADER_RE = re.compile(r"^(?P<h>[A-Z][A-Z0-9 /&,\-\(\)'\.]{2,})\s*:?\s*$")


def note_sections(note: str) -> list[tuple[str | None, str, int]]:
    """Split a note into [(header_or_None, body, char_offset_of_body_start)].

    Text before the first header gets header None.
    """
    sections, cur_h, cur_lines, cur_start, pos = [], None, [], 0, 0
    for line in note.split("\n"):
        m = HEADER_RE.match(line.strip())
        if m and len(re.sub(r"[^A-Z]", "", m.group("h"))) >= 3:
            if cur_h is not None or cur_lines:
                sections.append((cur_h, "\n".join(cur_lines), cur_start))
            cur_h = m.group("h").strip()
            cur_lines = []
            cur_start = pos + len(line) + 1
        else:
            cur_lines.append(line)
        pos += len(line) + 1
    sections.append((cur_h, "\n".join(cur_lines), cur_start))
    return sections


# ---------- misc -------------------------------------------------------------

def ctx(text: str, start: int, end: int, w: int = 40) -> str:
    """Return a one-line context window around [start,end) with the span marked."""
    s = max(0, start - w)
    e = min(len(text), end + w)
    return (text[s:start] + "«" + text[start:end] + "»" + text[end:e]).replace("\n", " ⏎ ")


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "0/0"


def save_json(name: str, obj) -> Path:
    p = OUT_DIR / name
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=_default)
    print(f"[saved] {p.relative_to(ROOT)}")
    return p


def _default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Counter):
        return dict(o)
    raise TypeError(type(o))


def head(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def per_dataset_table(recs: list[dict], fn, label: str = "") -> None:
    """Print `label` computed by fn(list_of_recs) for each split x dataset."""
    rows = []
    for split in ("train", "eval", "all"):
        sub = by_split(recs)[split]
        cells = [fn(sub)] + [fn(v) for v in by_dataset(sub).values()]
        rows.append((split, cells))
    print(f"{label:<10}" + "".join(f"{c:>16}" for c in ("ALL",) + DATASETS))
    for split, cells in rows:
        print(f"{split:<10}" + "".join(f"{str(c):>16}" for c in cells))


def sample(items, k: int, seed: int = 0):
    import random
    rnd = random.Random(seed)
    items = list(items)
    if len(items) <= k:
        return items
    return rnd.sample(items, k)
