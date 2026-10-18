"""Resumable teacher/ground-truth annotation with headless, tool-scoped Claude Code runs.

Survives SSH loss and VM reboots when run from the nw-annotate systemd service: every pass re-checks which
encounters still lack a valid annotation file and only (re)annotates those.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
from check_annotations import check, load_split  # noqa: E402

SPLITS = {
    "eval": ROOT / "data/eval_patient_doctor_encounters.jsonl",
    "train": ROOT / "data/patient_doctor_encounters.jsonl",
}
CHUNK, WORKERS, PASSES, TIMEOUT = 5, 7, 3, 3600
LOG_DIR = ROOT / "outputs/logs"
TASK_DIR = ROOT / "annotations/tasks/single"

PROMPT = """You are producing {purpose} for a research benchmark on de-identifying patient–doctor encounters
(public research dataset). Work carefully and exhaustively; quality matters more than speed.

1. Read the guidelines: {root}/annotations/GUIDELINES.md and follow them exactly. {extra}
2. Annotate each of these encounters (each task file holds one encounter: "=== FIELD: dialogue" then "=== FIELD: note"):
{tasks}
   Read each task file completely. Annotate PHI (tier 1 and tier 2) separately for the dialogue and the note: every
   name form (first name alone in the dialogue, full name in the note, surname after mr./ms./dr.), exact ages,
   absolute dates/years, places, organizations, contacts/IDs, occupations, hobbies/activities, life events,
   holidays/weekday timing, rare personal details. Do NOT annotate "dax", generic relation words, sex, or clinically
   needed relative time; disease eponyms are medical, not names.
3. Write one JSON file per encounter to {root}/annotations/{split}/<encounter_id>.json with the Write tool:
   {{"encounter_id": ..., "phi": [...], "medical_concepts": [...]}}
4. Validate with: python3 evaluation/check_annotations.py {data} {outs}
   Fix every reported problem until it reports 0 problems.
5. Reply with one line per encounter: id and PHI count."""

EXTRA = {
    "eval": "Also annotate the medical concepts with icd_relevant flags (the task file lists the ICD-10 codes). "
            "Ignore the 'Teacher labels for the training split' section.",
    "train": "Follow the final section 'Teacher labels for the training split': every PHI item needs a "
             "\"generalized\" field. Medical concepts are NOT needed: use \"medical_concepts\": [].",
}
PURPOSE = {"eval": "gold-standard evaluation annotations", "train": "teacher labels (PHI annotations)"}


def missing(split):
    records = load_split(SPLITS[split])
    out = []
    for eid in records:
        f = ROOT / f"annotations/{split}/{eid}.json"
        if not f.exists() or check(f, records):
            out.append(eid)
    return records, out


def write_task(split, rec):
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    codes = "; ".join(f"{c['code']} {c['description']}" for c in rec["icd10_codes"])
    head = f"########## ENCOUNTER {rec['encounter_id']}\n" + (f"ICD-10 codes: {codes}\n" if split == "eval" else "")
    path = TASK_DIR / f"{split}_{rec['encounter_id']}.txt"
    path.write_text(head + f"=== FIELD: dialogue\n{rec['dialogue']}\n=== FIELD: note\n{rec['note']}\n")
    return path


def run_chunk(split, records, ids):
    tasks = "\n".join(f"   - {write_task(split, records[i])}" for i in ids)
    outs = " ".join(f"annotations/{split}/{i}.json" for i in ids)
    prompt = PROMPT.format(purpose=PURPOSE[split], root=ROOT, extra=EXTRA[split], tasks=tasks, split=split,
                           data=SPLITS[split].relative_to(ROOT), outs=outs)
    cmd = ["claude", "-p", prompt, "--model", "opus", "--allowedTools", "Read", "Edit(annotations/**)",
           "Bash(python3 evaluation/check_annotations.py:*)"]
    log = LOG_DIR / f"annotate_{split}_{ids[0]}.log"
    t = time.time()
    try:
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
        log.write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
    except subprocess.TimeoutExpired:
        log.write_text("TIMEOUT")
    print(f"[{time.strftime('%H:%M:%S')}] {split} {ids} done in {time.time() - t:.0f}s", flush=True)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    splits = sys.argv[1:] or ["eval", "train"]
    for p in range(PASSES):
        jobs = []
        for split in splits:
            records, todo = missing(split)
            print(f"pass {p}: {split} missing {len(todo)}", flush=True)
            jobs += [(split, records, todo[i:i + CHUNK]) for i in range(0, len(todo), CHUNK)]
        if not jobs:
            break
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda j: run_chunk(*j), jobs))
    left = {s: missing(s)[1] for s in splits}
    print("remaining:", left, flush=True)
    sys.exit(1 if any(left.values()) else 0)


if __name__ == "__main__":
    main()
