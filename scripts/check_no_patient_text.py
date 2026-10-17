"""Release guard: no labeled identifier from the challenge data may appear in a tracked file.

Collects every silver-label PII string of a direct-identifier category (names, DOB, IDs, dates) of the
challenge encounters (train and eval labels in labels/; the robustness sets are synthetic and skipped) and
searches all files tracked by Git (whole word, case-insensitive). Ordinary English words
are skipped because they also occur in code and prose. Run before every commit that will be published:

    uv run python scripts/check_no_patient_text.py          # exit 1 when anything is found

Findings are printed locally only; they quote the matched string so it can be removed. Reviewed generic
words (eponyms, word-sense examples, synthetic generator names) are listed per file in
configs/release_guard_allow.txt.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from unified_harness import lexicons as L
from unified_harness import paths

CATEGORIES = {"PATIENT_NAME", "PERSON_NAME", "CLINICIAN_NAME", "OTHER_PERSON_NAME", "DOB", "ID", "DATE", "CONTACT"}
CHALLENGE_SPLITS = ("tune", "validation")
ALLOWLIST = "configs/release_guard_allow.txt"
SKIP_SUFFIXES = {".gz", ".lock", ".png", ".jpg", ".pdf"}


def label_strings() -> dict[str, str]:
    """{lower-cased string: category} over every label source in labels/."""
    found: dict[str, str] = {}
    root = paths.labels_dir()
    folders = [root / "upstream" / s for s in ("eval", "dev", "train")] + [
        root / "merged" / s for s in CHALLENGE_SPLITS
    ]
    files = [f for d in folders for f in d.glob("*.json")] + list((root / "ketai").glob("*.jsonl"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        rows = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if f.suffix == ".jsonl"
            else [json.loads(text)]
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            for item in row.get("pii", []) or []:
                s, cat = str(item.get("text", "")).strip(), str(item.get("category", "")).upper()
                if cat in CATEGORIES and len(s) >= 4 and not L.is_common_word(s.lower()):
                    found.setdefault(s.lower(), cat)
    return found


def allowlist(root: Path) -> dict[str, set[str]]:
    """{tracked file: allowed lower-cased words} from configs/release_guard_allow.txt."""
    path = root / ALLOWLIST
    allowed: dict[str, set[str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                name, _, words = line.partition(":")
                allowed.setdefault(name.strip(), set()).update(w.strip().lower() for w in words.split(","))
    return allowed


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [root / p for p in out.splitlines() if Path(p).suffix not in SKIP_SUFFIXES]


def main() -> int:
    strings = label_strings()
    if not strings:
        print("no labels found in labels/; nothing to check against", file=sys.stderr)
        return 1
    rx = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(s) for s in sorted(strings, key=len, reverse=True)) + r")(?!\w)", re.I
    )
    hits, allowed = 0, allowlist(paths.home())
    for f in tracked_files(paths.home()):
        rel = str(f.relative_to(paths.home()))
        if rel == ALLOWLIST:
            continue  # the reviewed list itself
        ok = allowed.get(rel, set())
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, line in enumerate(lines, 1):
            for m in rx.finditer(line):
                if m.group(1).lower() in ok:
                    continue
                hits += 1
                print(f"{f.relative_to(paths.home())}:{n}: {strings[m.group(1).lower()]}: {m.group(1)!r}")
    print(f"checked {len(strings)} labeled identifiers against tracked files: {hits} finding(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
