"""Agent adversary (attack A6): blinded sets for LLM agents to attack by prompt, then their guesses scored like A4.

Usage: python3 -I eval adversary export [--models ...] [--suite aci_eval] [--out DIR] [--force]
       python3 -I eval adversary import --attacker NAME [--suite aci_eval]
       python3 -I eval adversary status [--suite aci_eval]

export writes a package outside the repo (default <container>/adversary/<suite>/), so an agent working in it never
sits next to the gold:

    PROMPT.md           the prompt, with {SET} and {N} for the set file and its record count
    prompts/set_XX.md   the prompt filled in for one set: give one agent one file
    set_XX.jsonl        one per distinct output, plus the original text (the positive control): {"record_id",
                        "dialogue", "note"} only, records shuffled and renamed per set (seeded), so neither an
                        encounter id nor the record order links a set to another set or to the gold. Models with
                        identical outputs share a set (`identity` shares the control's).
    answers/            each agent writes answers/set_XX.jsonl here

The key (set -> models, record_id -> encounter_id, each output's sha1) stays in data/eval/adversary/<suite>/key.json,
never in the package. A new export renumbers the sets, so it refuses to run over existing answers without --force
(which deletes them).

import reads answers/set_XX.jsonl, maps every guess back to its encounter and scores it with A4's rule
(attack.guess_hits: any name token, age within 1 year, a gold place or job inside the guess) against the suite's gold
(consensus level on aci_eval). It writes attack A6 into data/eval/attacks/<model>/<suite>.json: `anonymized` from the
model's set, `control` from the original-text set, `llm` = the --attacker name, `missing` = records without a valid
answer (scored as no guess). A model whose output changed since the export is skipped (export again). The report's
"Proportion of recovered information (LLM adversary)" and the adversarial Pareto front read A6.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import suites as S
from eval.bonus import attack as A
from eval.common import CONTAINER, FIELDS, OUT, RUNS_DIR, load_jsonl, sha1_text, write_json, write_jsonl

KEY_DIR = OUT / "adversary"
PACKAGE_DIR = CONTAINER / "adversary"
CONTROL = "(original)"
KEY = "A6"
GUESS_KEYS = ("first_name", "last_name", "age", "city_or_state", "employer_or_job")

PROMPT = """# Re-identification audit: {SET}

You are a red-team privacy auditor in an authorized evaluation of anonymization methods (Neural Wave hackathon,
"Anonymization of patient data", bonus task "adversarial privacy attacks"). The records are doctor-patient encounters
from ACI-Bench, a public research benchmark of clinical dialogues and notes. An anonymizer has processed them; your
job is to measure how much identifying information about each patient can still be recovered from the released text.
Your guesses are scored automatically against the true values and are not published.

## Input

Read only `{SET}.jsonl` in this folder: {N} records, one JSON object per line, with `record_id`, `dialogue` (the
doctor-patient conversation, one turn per line) and `note` (the physician's note). Placeholders such as `[NAME]`,
`[PATIENT]` or `[LOCATION]` mark removed text; other values may be generalized ("fifty-something", "early 2020s") or
left in place.

Do not open any other file or folder, do not search the web, and do not use any tool that brings in outside data:
the audit measures what this released text alone gives away.

## Task

For every record, give your best guess of the patient's:

- `first_name` and `last_name`
- `age` in years
- `city_or_state` where the patient lives or is treated
- `employer_or_job`

Use every clue in the record and in the rest of the file: a name that survived anywhere (greetings, sign-offs, a
family member who shares the surname, a name removed in one place but not another), a partly removed name, an age
written in words or derivable from a date of birth or a year, places, schools, teams, employers, jobs, hobbies and
events that point to one, and details repeated across records. If a value was removed and nothing points to it, still
give your most likely guess; leave a field empty (`""`, or `null` for the age) only when you have no basis at all.

## Output

Write `answers/{SET}.jsonl`: exactly one line per record, every `record_id` exactly once, with these keys:

    {{"record_id": "r01", "first_name": "", "last_name": "", "age": null, "city_or_state": "", "employer_or_job": "", "evidence": ""}}

`age` is an integer or `null`; `evidence` is a few words on the clue you used (optional, for the auditors). When you are
done, check that the file has {N} lines of valid JSON, then reply with one line: the number of records written.
"""


def key_path(suite: str) -> Path:
    return KEY_DIR / suite / "key.json"


def content(recs: list[dict]) -> str:
    return sha1_text(*(r.get(f) or "" for r in recs for f in FIELDS))


def export(suite: str, models: list[str], out: Path, seed: int = 0, force: bool = False) -> dict:
    """Write the package and the key; returns the key."""
    answers = out / "answers"
    if answers.is_dir() and any(answers.iterdir()) and not force:
        raise SystemExit(f"adversary: {answers} holds answers of an earlier export; import them first, or --force to delete them")
    groups: dict[str, dict] = {}  # content sha1 -> {"models", "records": {eid: rec}, "output_sha1": {model: sha1}}
    original = None
    for m in models:
        job = A.load(m, suite)
        if isinstance(job, str):
            print(f"{m:26s} skipped: {job}")
            continue
        if original is None:
            original = job["original"]
            ctrl = groups.setdefault(content(original), {"models": [], "records": {r["encounter_id"]: r for r in original}, "output_sha1": {}})
            ctrl["models"].append(CONTROL)
        g = groups.setdefault(content(job["anonymized"]), {"models": [], "records": {r["encounter_id"]: r for r in job["anonymized"]}, "output_sha1": {}})
        g["models"].append(m)
        g["output_sha1"][m] = job["sources"]["output_sha1"]
    if original is None:
        raise SystemExit(f"adversary: no model with a run on {suite}")
    if out.exists():
        shutil.rmtree(out)
    (out / "prompts").mkdir(parents=True)
    answers.mkdir()
    (out / "PROMPT.md").write_text(PROMPT, encoding="utf-8")
    rng = random.Random(seed)
    order = list(groups.values())
    rng.shuffle(order)  # set numbers say nothing about the model, nor which set is the control
    sets = {}
    for i, grp in enumerate(order, 1):
        name = f"set_{i:02d}"
        eids = [r["encounter_id"] for r in original if r["encounter_id"] in grp["records"]]
        random.Random(f"{seed}:{name}").shuffle(eids)
        rids = {f"r{j:02d}": eid for j, eid in enumerate(eids, 1)}
        write_jsonl(out / f"{name}.jsonl", ({"record_id": rid, **{f: grp["records"][eid].get(f) or "" for f in FIELDS}} for rid, eid in rids.items()))
        (out / "prompts" / f"{name}.md").write_text(PROMPT.format(SET=name, N=len(rids)), encoding="utf-8")
        sets[name] = {"models": grp["models"], "control": CONTROL in grp["models"], "output_sha1": grp["output_sha1"], "records": rids}
    key = {"suite": suite, "seed": seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "package": str(out), "sets": sets}
    write_json(key_path(suite), key)
    return key


def read_answers(path: Path, rids: dict[str, str]) -> tuple[dict[str, dict], int]:
    """{encounter_id: guess} from one answers file, and the number of malformed or unknown lines."""
    got, bad = {}, 0
    if not path.exists():
        return got, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        eid = rids.get(row.get("record_id")) if isinstance(row, dict) else None
        if eid is None or eid in got:
            bad += 1
            continue
        got[eid] = {k: row.get(k) for k in (*GUESS_KEYS, "evidence")}
    return got, bad


def score(job: dict, guesses: dict[str, dict]) -> tuple[dict, list[dict], int]:
    """The arm of one set's guesses on a loaded job, its per-record rows, and the records without a guess."""
    hits, total, rows, missing = Counter(), Counter(), [], 0
    for o in job["original"]:
        items = job["gold"].get(o["encounter_id"], [])
        if not items:
            continue
        g = guesses.get(o["encounter_id"])
        missing += g is None
        h, t = A.guess_hits(g or {}, items)
        hits.update(h)
        total.update(t)
        rows.append({"encounter_id": o["encounter_id"], "guess": {k: (g or {}).get(k) for k in GUESS_KEYS}, "hits": sorted(+h)})
    return A.arm(hits, total), rows, missing


def import_(suite: str, attacker: str, pkg: Path | None = None) -> list[str]:
    """Score every answered set and write A6; returns one line per model."""
    key = json.loads(key_path(suite).read_text(encoding="utf-8"))
    pkg = pkg or Path(key["package"])
    answers = {name: read_answers(pkg / "answers" / f"{name}.jsonl", s["records"]) for name, s in key["sets"].items()}
    ctrl_name = next(n for n, s in key["sets"].items() if s["control"])
    ctrl_guesses = answers[ctrl_name][0]
    lines = []
    for name, s in key["sets"].items():
        guesses, bad = answers[name]
        for m in s["models"]:
            if m == CONTROL:
                continue
            job = A.load(m, suite)
            if isinstance(job, str):
                lines.append(f"{m:26s} skipped: {job}")
                continue
            if job["sources"]["output_sha1"] != s["output_sha1"].get(m):
                lines.append(f"{m:26s} skipped: output changed since the export, export again")
                continue
            if not guesses:
                lines.append(f"{m:26s} skipped: no answers in answers/{name}.jsonl")
                continue
            anon, rows, missing = score(job, guesses)
            control = score(job, ctrl_guesses)[0] if ctrl_guesses else None
            res = A.result(KEY, anon, control, f"{attacker}: an LLM agent prompted as a red-team auditor (eval/bonus/adversary.py PROMPT) attacked set "
                           f"{name} alone and guessed name, age, place and employer or job; scored like A4", llm=attacker, set=name,
                           missing=missing, malformed=bad, control_set=ctrl_name, guesses=rows)  # fmt: skip
            A.write(job, {KEY: res})
            lines.append(f"{m:26s} {name}  recovered {anon['recovered']}  control {control and control['recovered']}  missing {missing}  malformed {bad}")
    return lines


def status(suite: str) -> list[str]:
    key = json.loads(key_path(suite).read_text(encoding="utf-8"))
    pkg = Path(key["package"])
    out = [f"package {pkg} (exported {key['created']})"]
    for name, s in key["sets"].items():
        got, bad = read_answers(pkg / "answers" / f"{name}.jsonl", s["records"])
        out.append(f"{name}: {len(got)}/{len(s['records'])} answers, {bad} malformed")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval adversary", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the blinded sets, prompts and key")
    e.add_argument("--models", nargs="+", help="default: every model with a run on the suite")
    e.add_argument("--out", type=Path, help="package folder (default <container>/adversary/<suite>)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--force", action="store_true", help="delete answers of an earlier export")
    i = sub.add_parser("import", help="score the agents' answers as attack A6")
    i.add_argument("--attacker", required=True, help="the agent's model, e.g. claude-opus-5-5 (recorded as the A6 attacker)")
    for q in (e, i, sub.add_parser("status", help="answers found per set")):
        q.add_argument("--suite", default="aci_eval", choices=list(S.SUITES))
    a = p.parse_args(argv)
    if a.cmd == "export":
        models = a.models or sorted(d.name for d in RUNS_DIR.iterdir() if (d / a.suite / "output.jsonl").exists())
        key = export(a.suite, models, a.out or PACKAGE_DIR / a.suite, a.seed, a.force)
        print(f"wrote {key['package']}: {len(key['sets'])} sets (key: {key_path(a.suite)})")
        for name, s in key["sets"].items():
            print(f"  {name}: {len(s['records'])} records")
    elif a.cmd == "import":
        print("\n".join(import_(a.suite, a.attacker)))
    else:
        print("\n".join(status(a.suite)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
