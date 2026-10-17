"""Shared helpers for the dataset converters: paths, ICD-10-CM lookup, gold spans, writers.

Every converter writes `data/converted/<name>/` with three files (see README.md):

    encounters.jsonl  challenge schema, exactly: encounter_id, dataset, dialogue, note, icd10_codes
    labels.jsonl      gold for evaluation, one line per encounter in the same order (never fed to the anonymizer)
    manifest.json     dataset-level provenance and what the labels cover

The scripts only read files under `data/kaggle/` (parsed as data, never executed) and use the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
KAGGLE = DATA / "kaggle"
OUT = DATA / "converted"

csv.field_size_limit(sys.maxsize)

ENCOUNTER_KEYS = ("encounter_id", "dataset", "dialogue", "note", "icd10_codes")
FIELDS = ("dialogue", "note")
ICD_SOURCES = ("gold", "mapped", "none")
# How complete an encounter's `pii` list is. exhaustive: every private item in the text is labelled (removing anything
# else is over-removal). partial: labelled items are real PII but others exist (recall only). none: no privacy gold.
PII_GOLD = ("exhaustive", "partial", "none")

# Same categories and default risk as the privacy gold of the evaluation harness (gold_annotation_guidelines.md),
# so ACI-Bench gold and converted gold can be scored together. The original label is kept per span in `label`.
CATEGORY_RISK = {
    "PATIENT_NAME": "high",  # name of the data subject (patient, essay author)
    "PERSON_NAME": "high",  # any other person: clinician, relative, ...
    "DOB": "high",
    "ID": "high",  # MRN, NPI, insurance/account numbers, phone, e-mail, username, URL
    "DATE": "high",
    "AGE": "high",
    "YEAR": "medium",
    "LOCATION": "medium",  # address, city, named hospital/clinic
    "ORGANIZATION": "medium",  # employer, insurer, school
    "OCCUPATION": "medium",
    "OTHER_QI": "low",  # other quasi-identifier: hobby, event, family structure
}
KEEP_CATEGORIES = ("CODE_EVIDENCE", "EPONYM", "CLINICAL_NUMBER", "NOT_A_NAME", "CLINICAL_FEATURE")
# Labels of the Kaggle "PII Data Detection" family (essay datasets) -> categories above.
PII_LABELS = {"NAME_STUDENT": "PATIENT_NAME", "EMAIL": "ID", "USERNAME": "ID", "ID_NUM": "ID", "PHONE_NUM": "ID",
              "URL_PERSONAL": "ID", "STREET_ADDRESS": "LOCATION"}  # fmt: skip

_EDGE_LEFT = " \t\n\"'(“‘«„["
_EDGE_RIGHT = " \t\n\"')”’»].,;:!?"


# ---------------------------------------------------------------- ICD-10


def dotted(code: str) -> str:
    """`e119` / `E11.9` -> `E11.9`."""
    c = code.strip().upper().replace(".", "")
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


@cache
def icd10cm() -> dict[str, str]:
    """ICD-10-CM 2023 billable codes (dotted) -> official long description (what the challenge data uses)."""
    out = {}
    with (KAGGLE / "mrhell__icd10cm-codeset-2023" / "icd10cm_codes_2023.txt").open(encoding="utf-8") as fh:
        for line in fh:
            code, desc = line[:7].strip(), line[8:].strip()
            if code:
                out[dotted(code)] = desc
    return out


def icd_code(code: str, description: str | None = None) -> dict:
    """`{"code", "description"}` with the official description; raises for a code that is not in ICD-10-CM 2023."""
    c = dotted(code)
    desc = description if description is not None else icd10cm().get(c)
    if desc is None:
        raise KeyError(f"{c} is not an ICD-10-CM 2023 code")
    return {"code": c, "description": desc}


# ---------------------------------------------------------------- records


def clean(text: str | None) -> str:
    """Normalize line endings and strip the outer whitespace. Apply before computing any span."""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def encounter(eid: str, dataset: str, *, dialogue: str = "", note: str = "", codes: Iterable[dict] = ()) -> dict:
    return {"encounter_id": eid, "dataset": dataset, "dialogue": dialogue, "note": note, "icd10_codes": list(codes)}


def dialogue_text(turns: Iterable[tuple[str, str]]) -> str:
    """`[("doctor", "hi ."), ...]` -> challenge-style `[doctor] hi .` lines, one turn per line."""
    lines = []
    for speaker, text in turns:
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)


def find_all(text: str, needle: str, *, ignore_case: bool = True) -> list[tuple[int, int]]:
    """Every occurrence of `needle` in `text`, whole-word at alphanumeric edges, any whitespace between its words."""
    words = needle.split()
    if not words:
        return []
    body = r"\s+".join(map(re.escape, words))
    left = r"(?<!\w)" if needle.strip()[0].isalnum() else ""
    right = r"(?!\w)" if needle.strip()[-1].isalnum() else ""
    return [m.span() for m in re.finditer(left + body + right, text, re.IGNORECASE if ignore_case else 0)]


class Labels:
    """Gold spans of one encounter. A span overlapping one already added to the same list is rejected,
    so callers add the most specific evidence first (full names before name tokens)."""

    def __init__(self, enc: dict):
        self.enc = enc
        self.pii: list[dict] = []
        self.keep: list[dict] = []

    def _span(self, items: list[dict], field: str, start: int, end: int, trim: bool) -> tuple[int, int] | None:
        text = self.enc[field]
        if trim:  # token-level labels carry punctuation ("Popova,"); keep a bracket whose partner is inside: "(95) 94215"
            while start < end and text[start] in _EDGE_LEFT and not (text[start] == "(" and ")" in text[start:end]):
                start += 1
            while end > start and text[end - 1] in _EDGE_RIGHT and not (text[end - 1] == ")" and "(" in text[start:end]):
                end -= 1
        if not 0 <= start < end <= len(text):
            return None
        if any(it["field"] == field and start < it["end"] and it["start"] < end for it in items):
            return None
        return start, end

    def add_pii(self, field: str, start: int, end: int, category: str, label: str, *, risk: str | None = None, trim: bool = True) -> bool:
        if category not in CATEGORY_RISK:
            raise ValueError(f"unknown PII category {category!r}")
        span = self._span(self.pii, field, start, end, trim)
        if span is None:
            return False
        s, e = span
        self.pii.append({"field": field, "start": s, "end": e, "text": self.enc[field][s:e], "category": category,
                         "risk": risk or CATEGORY_RISK[category], "label": label})  # fmt: skip
        return True

    def add_keep(self, field: str, start: int, end: int, category: str, why: str, *, trim: bool = True) -> bool:
        if category not in KEEP_CATEGORIES:
            raise ValueError(f"unknown keep category {category!r}")
        span = self._span(self.keep, field, start, end, trim)
        if span is None:
            return False
        s, e = span
        self.keep.append({"field": field, "start": s, "end": e, "text": self.enc[field][s:e], "category": category, "why": why})
        return True

    def match_pii(self, needle: str, category: str, label: str, *, fields: Iterable[str] = FIELDS, ignore_case: bool = True,
                  risk: str | None = None) -> int:  # fmt: skip
        """Add every occurrence of `needle` in the given fields; returns how many were added."""
        n = 0
        for f in fields:
            for s, e in find_all(self.enc[f], needle, ignore_case=ignore_case):
                n += self.add_pii(f, s, e, category, label, risk=risk, trim=False)
        return n

    def match_keep(self, needle: str, category: str, why: str, *, fields: Iterable[str] = FIELDS) -> int:
        n = 0
        for f in fields:
            for s, e in find_all(self.enc[f], needle):
                n += self.add_keep(f, s, e, category, why, trim=False)
        return n

    def record(self, icd10_source: str, *, pii_gold: str, meta: dict | None = None) -> dict:
        """The labels.jsonl line. `pii_gold` says how complete this encounter's `pii` list is (see PII_GOLD)."""
        if icd10_source not in ICD_SOURCES:
            raise ValueError(f"icd10_source must be one of {ICD_SOURCES}")
        if pii_gold not in PII_GOLD or (pii_gold == "none" and self.pii):
            raise ValueError(f"{self.enc['encounter_id']}: pii_gold must be one of {PII_GOLD}, and not 'none' with spans")
        order = {f: i for i, f in enumerate(FIELDS)}
        return {
            "encounter_id": self.enc["encounter_id"],
            "icd10_source": icd10_source,
            "pii_gold": pii_gold,
            "pii": sorted(self.pii, key=lambda it: (order[it["field"]], it["start"])),
            "keep": sorted(self.keep, key=lambda it: (order[it["field"]], it["start"])),
            "meta": meta or {},
        }


class TextBuilder:
    """Concatenate text pieces and remember where the labelled ones landed (for rendered documents)."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.size = 0
        self.spans: list[tuple[int, int, str, str]] = []  # start, end, category, label

    def add(self, text: str, category: str | None = None, label: str | None = None) -> TextBuilder:
        if category and text:
            self.spans.append((self.size, self.size + len(text), category, label or category))
        self.parts.append(text)
        self.size += len(text)
        return self

    def text(self) -> str:
        return "".join(self.parts)


def bio_spans(tokens: list[str], labels: list[str], offsets: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """Merge BIO token labels into `(start, end, type)` character spans. An `I-` continues the span only when the
    previous token carries the same type; after `O` or another type it starts a new span (upstream label noise)."""
    spans: list[list] = []
    prev = None
    for tok, tok_label, (s, e) in zip(tokens, labels, offsets):
        if tok_label == "O":
            if tok.strip():  # whitespace tokens labelled O (line breaks inside an address) don't end a span
                prev = None
            continue
        tag, _, typ = tok_label.partition("-")
        if tag == "I" and prev == typ:
            spans[-1][1] = e
        else:
            spans.append([s, e, typ])
        prev = typ
    return [tuple(x) for x in spans]


# ---------------------------------------------------------------- CLI + output


def parser(doc: str | None, src: Path, *, limit: int = 0) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(doc or "").strip().splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=doc)
    p.add_argument("--src", type=Path, default=src, help=f"source folder or file (default: {src.relative_to(REPO)})")
    p.add_argument("--out", type=Path, default=OUT, help="output root (default: data/converted)")
    p.add_argument("--limit", type=int, default=limit, help=f"keep a seeded random sample of N records, 0 = all (default: {limit})")
    p.add_argument("--seed", type=int, default=13)
    return p


def sample(rows: list, limit: int, seed: int) -> list:
    """Seeded random subset that keeps the source order."""
    if not limit or len(rows) <= limit:
        return rows
    keep = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[i] for i in keep]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_dataset(name: str, items: Iterable[tuple[dict, dict]], manifest: dict, out: Path = OUT) -> Path:
    """Write encounters.jsonl + labels.jsonl + manifest.json under `out/name/` and print a one-line summary.
    A single-language dataset gets its manifest `language` copied into every record's `meta.language`."""
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    n = n_dialogue = n_note = 0
    icd_src: Counter = Counter()
    gold: Counter = Counter()
    cats: Counter = Counter()
    ids: set[str] = set()
    with (d / "encounters.jsonl").open("w", encoding="utf-8") as fe, (d / "labels.jsonl").open("w", encoding="utf-8") as fl:
        for enc, lab in items:
            eid = enc.get("encounter_id")
            if tuple(enc) != ENCOUNTER_KEYS:
                raise ValueError(f"{eid}: encounter keys {list(enc)} != {list(ENCOUNTER_KEYS)}")
            if lab["encounter_id"] != eid or eid in ids:
                raise ValueError(f"{eid}: duplicate id or labels out of step")
            if not (enc["dialogue"] or enc["note"]):
                raise ValueError(f"{eid}: empty encounter")
            if isinstance(manifest.get("language"), str):
                lab["meta"].setdefault("language", manifest["language"])
            ids.add(eid)
            fe.write(json.dumps(enc, ensure_ascii=False) + "\n")
            fl.write(json.dumps(lab, ensure_ascii=False) + "\n")
            n += 1
            n_dialogue += bool(enc["dialogue"])
            n_note += bool(enc["note"])
            icd_src[lab["icd10_source"]] += 1
            gold[lab["pii_gold"]] += 1
            cats.update(it["category"] for it in lab["pii"])
    summary = {
        "records": n,
        "with_dialogue": n_dialogue,
        "with_note": n_note,
        "icd10_source": dict(icd_src),
        "pii_gold_records": dict(gold),
        "pii_spans": dict(cats.most_common()),
    }
    (d / "manifest.json").write_text(json.dumps({"name": name, "stage": "convert", **manifest, **summary}, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")  # fmt: skip
    print(f"{name}: {n} encounters ({n_dialogue} dialogue, {n_note} note), ICD {dict(icd_src)}, PII gold {dict(gold)}, "
          f"{sum(cats.values())} PII spans -> {d.relative_to(REPO) if d.is_relative_to(REPO) else d}")  # fmt: skip
    return d
