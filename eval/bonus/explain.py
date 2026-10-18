"""Explainability bonus: the share of the changed text that an anonymization decision with a reason explains.

Usage: python3 -I eval/bonus/explain.py [--models ...] [--suites ...]   (prints the scores of finished runs;
       `eval score` stores the same result as `explain` in every score file)

The task asks for the "percentage of anonymization decisions having an explanation". Counting the decisions a model
chooses to log rewards logging little, so the unit here is the text: every original word token that the token
alignment of `metrics.align` marks as changed must be covered by a decision (action other than `keep`) with a
non-empty reason. A model logs its decisions in an optional sidecar `runs/<model>/<suite>/output.decisions.jsonl`,
or inline as an `explanations` list in each output record (the one extra key validity allows). One row per decision:

    {"encounter_id", "field", "start", "end", "text", "action": "remove"|"generalize"|"perturb"|"keep",
     "category", "reason", "source"}

`start`/`end` index the ORIGINAL field. Rows are normalized first:
- A+D's sidecar: `action` REPLACE/GENERALIZE/KEEP (REPLACE by a non-placeholder becomes `perturb`), the reason from
  `explanation` (else `detector_reason` + `policy_reason`), no `text` (taken from the original slice).
- Ketai's generalize log: `type` for the category, `removed`/`generalized`/`retained`, no offsets.
- A row whose `text` differs from the original slice is moved to the nearest whole-word occurrence of its text; a
  row without offsets covers every occurrence of its text in its field (both fields when it names none). A row that
  cannot be placed, names an unknown record, or is not valid UTF-8 JSON counts as `unaligned`.

Result: {"explained": covered / changed tokens, None when nothing changed and 0.0 when no explanation source exists,
"changed_tokens", "covered_tokens", "decisions" (rows), "with_reason", "unaligned", "source": "sidecar"|"inline"|"none"}.
`decisions`/`with_reason` is the literal "decisions having an explanation" ratio, shown beside it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if not __package__:  # run as a file: the repo root makes the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval import metrics as M
from eval.common import FIELDS, RUNS_DIR, SUITES_DIR, load_jsonl, occurrences, ratio

SIDECAR = "output.decisions.jsonl"
ACTIONS = {
    "remove": "remove", "removed": "remove", "replace": "remove", "replaced": "remove", "redact": "remove", "redacted": "remove",
    "mask": "remove", "masked": "remove", "suppress": "remove", "delete": "remove", "deleted": "remove",
    "generalize": "generalize", "generalized": "generalize", "generalise": "generalize", "generalised": "generalize",
    "perturb": "perturb", "perturbed": "perturb", "pseudonymize": "perturb", "pseudonymized": "perturb", "surrogate": "perturb",
    "substitute": "perturb", "substituted": "perturb", "shift": "perturb", "shifted": "perturb",
    "keep": "keep", "kept": "keep", "retain": "keep", "retained": "keep", "skip": "keep", "skipped": "keep",
}  # fmt: skip
PLACEHOLDER = re.compile(r"\s*\[[^\[\]]*\]\s*")


def _text(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def normalize(row: dict) -> dict:
    """One decision row in the common schema; offsets and text are checked against the original later (`place`)."""
    act = _text(row.get("action")).lower()
    action = ACTIONS.get(act, "remove")  # a logged decision without a known action changed the text
    rep = row.get("replacement")
    if act in ("replace", "replaced") and isinstance(rep, str) and rep.strip() and not PLACEHOLDER.fullmatch(rep):
        action = "perturb"  # a surrogate value, not a placeholder
    reason = _text(row.get("reason")) or _text(row.get("explanation"))
    reason = reason or "; ".join(x for x in (_text(row.get("detector_reason")), _text(row.get("policy_reason"))) if x)
    start, end = row.get("start"), row.get("end")
    ok = all(isinstance(x, int) and not isinstance(x, bool) for x in (start, end)) and 0 <= start < end
    eid = row.get("encounter_id")
    return {"encounter_id": eid if isinstance(eid, (str, int)) else None, "field": row.get("field") if row.get("field") in FIELDS else None,
            "start": start if ok else None, "end": end if ok else None, "text": row.get("text") if isinstance(row.get("text"), str) else None,
            "action": action, "category": _text(row.get("category")) or _text(row.get("type")) or _text(row.get("label")),
            "reason": reason, "source": _text(row.get("source")) or _text(row.get("detector"))}  # fmt: skip


def place(d: dict, rec: dict) -> list[tuple[str, int, int]]:
    """The (field, start, end) spans of a normalized decision in the original record (empty: it cannot be placed)."""
    fields = [d["field"]] if d["field"] else list(FIELDS)
    if d["start"] is not None and d["field"]:
        orig = rec.get(d["field"]) or ""
        s, e = d["start"], d["end"]
        if e <= len(orig) and (d["text"] is None or orig[s:e].lower() == d["text"].lower()):
            return [(d["field"], s, e)]
        if d["text"] and (found := occurrences(orig, d["text"])):  # offsets off: the nearest occurrence of its text
            fs, fe = min(found, key=lambda o: abs(o[0] - s))
            return [(d["field"], fs, fe)]
        return []
    if not d["text"]:
        return []
    return [(f, s, e) for f in fields for s, e in occurrences(rec.get(f) or "", d["text"])]


def decisions(run_dir: Path, anonymized: list[dict]) -> tuple[list[dict], str]:
    """The raw decision rows of a run and where they came from: the sidecar, else the inline `explanations`."""
    path = Path(run_dir) / SIDECAR
    if path.is_file():
        rows = []
        with path.open("rb") as fh:  # decoded per line: a line that is not UTF-8 is one malformed row, not a crash
            for line in fh:
                if line.strip():
                    try:
                        rows.append(json.loads(line.decode("utf-8-sig")))
                    except ValueError:  # not UTF-8 or not JSON
                        rows.append(None)  # counted as unaligned
        return rows, "sidecar"
    rows = [{"encounter_id": a.get("encounter_id"), **x} if isinstance(x, dict) else None
            for a in anonymized if isinstance(a.get("explanations"), list) for x in a["explanations"]]  # fmt: skip
    return rows, "inline" if rows else "none"


def score(run_dir: Path, original: list[dict], anonymized: list[dict]) -> dict:
    rows, source = decisions(run_dir, anonymized)
    by_id = {r["encounter_id"]: r for r in original}
    covering: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    with_reason = unaligned = 0
    for row in rows:
        d = normalize(row) if isinstance(row, dict) else None
        with_reason += bool(d and d["reason"])
        spans = place(d, by_id[d["encounter_id"]]) if d and d["encounter_id"] in by_id else []
        unaligned += not spans
        if spans and d["reason"] and d["action"] != "keep":
            for f, s, e in spans:
                covering[(d["encounter_id"], f)].append((s, e))
    changed = covered = 0
    out_by_id = {a.get("encounter_id"): a for a in anonymized}
    for r in original:
        a = out_by_id.get(r["encounter_id"])
        if a is None:
            continue
        for f in FIELDS:
            spans = covering.get((r["encounter_id"], f), [])
            for _, s, e, kept in M.align(r[f], a.get(f) if isinstance(a.get(f), str) else ""):
                if not kept:
                    changed += 1
                    covered += any(s < e2 and s2 < e for s2, e2 in spans)
    return {"explained": ratio(covered, changed) if changed else None, "changed_tokens": changed, "covered_tokens": covered,
            "decisions": len(rows), "with_reason": with_reason, "unaligned": unaligned, "source": source}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -I eval/bonus/explain.py", description=__doc__.splitlines()[0], epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--models", nargs="+", help="default: every model with a run")
    p.add_argument("--suites", nargs="+", help="default: every suite with a run")
    a = p.parse_args(argv)
    for mdir in sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []:
        if a.models and mdir.name not in a.models:
            continue
        for run_dir in sorted(x for x in mdir.iterdir() if (x / "output.jsonl").exists()):
            if (a.suites and run_dir.name not in a.suites) or not (SUITES_DIR / run_dir.name / "encounters.jsonl").exists():
                continue
            try:
                anonymized = load_jsonl(run_dir / "output.jsonl")
            except ValueError:  # not UTF-8 or not JSON: `eval score` reports it
                continue
            x = score(run_dir, load_jsonl(SUITES_DIR / run_dir.name / "encounters.jsonl"), anonymized)
            print(f"{mdir.name:18s} {run_dir.name:15s} explained {x['explained']}  changed {x['changed_tokens']}  covered {x['covered_tokens']}  "
                  f"decisions {x['decisions']} (with reason {x['with_reason']}, unaligned {x['unaligned']})  {x['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
