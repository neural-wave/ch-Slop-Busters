"""Project folders. Everything patient-related lives in git-ignored folders under the project home.

    data/                 challenge input files (read-only) and data/robustness/ test sets
    labels/               silver labels: labels/merged/<split>/ (used for scoring), labels/upstream/, labels/ketai/
    outputs/              runs, evaluation tables, caches
    configs/splits.json   encounter ids of the development splits (ids only, committed)

The home is the current directory unless UNIFIED_HARNESS_HOME is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DATA_FILES = {
    "eval": "eval_patient_doctor_encounters.jsonl",
    "train": "patient_doctor_encounters.jsonl",
    "dev": "patient_doctor_encounters.jsonl",  # 40 train encounters (upstream tuning set, kept for comparison)
    "tune": "patient_doctor_encounters.jsonl",  # all 140 train encounters: development and tuning
    "validation": "eval_patient_doctor_encounters.jsonl",  # the 67 eval encounters, reported as validation
    "test": "robustness/holdout.jsonl",  # untouched 24-document held-out set, scored once
}
MERGED_SPLITS = ("tune", "validation", "test")


def home() -> Path:
    return Path(os.environ.get("UNIFIED_HARNESS_HOME", ".")).resolve()


def data_dir() -> Path:
    return home() / "data"


def labels_dir() -> Path:
    return home() / "labels"


def outputs_dir() -> Path:
    return home() / "outputs"


def runs_dir() -> Path:
    return outputs_dir() / "runs"


def eval_dir() -> Path:
    return outputs_dir() / "eval"


def cache_dir() -> Path:
    return outputs_dir() / "cache"


def robustness_dir() -> Path:
    return data_dir() / "robustness"


def split_file(split: str) -> Path:
    return data_dir() / DATA_FILES[split]


def split_ids(split: str) -> list[str] | None:
    """Encounter ids of a development split from configs/splits.json (None = the whole file)."""
    path = home() / "configs" / "splits.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(split)


def gold_dir(split: str) -> Path:
    """Silver labels of a split: merged labels for tune / validation / test, upstream labels otherwise."""
    if split in MERGED_SPLITS:
        return labels_dir() / "merged" / split
    return labels_dir() / "upstream" / split
