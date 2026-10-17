"""The adapters that wrap code outside the repo: the zero-shot baseline runner and the decision-tree coder."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from eval.adapters import tree_coder, zeroshot_llm
from eval.tests.conftest import ORIGINAL

FAKE_ANONYMIZE = """
import argparse, json, pathlib
p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--out"); p.add_argument("--mode"); p.add_argument("--drop", type=int, default=0)
a = p.parse_args()
recs = [json.loads(l) for l in open(a.input)]
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
kept = recs[a.drop:]
(out / "anonymized.jsonl").write_text("".join(json.dumps(dict(r, note="[PERSON]")) + "\\n" for r in kept))
(out / "anonymize_meta.json").write_text(json.dumps({"mode": a.mode, "n_records": len(kept), "n_missing": a.drop}))
"""


def test_zeroshot_copies_a_complete_output_and_refuses_a_partial_one(tmp_path):
    root = tmp_path / "zeroshot"
    root.mkdir()
    (root / "anonymize.py").write_text(FAKE_ANONYMIZE)
    inp, out = tmp_path / "in.jsonl", tmp_path / "run" / "output.jsonl"
    out.parent.mkdir()
    inp.write_text("".join(json.dumps(r) + "\n" for r in ORIGINAL))
    assert zeroshot_llm.main([str(inp), str(out), "--root", str(root), "--mode", "extract"]) == 0
    got = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["encounter_id"] for r in got] == ["e1", "e2", "e3"] and {r["note"] for r in got} == {"[PERSON]"}
    assert (out.parent / "output.zeroshot" / "anonymized.jsonl").exists()  # kept next to the output, so a rerun resumes
    out.unlink()
    assert zeroshot_llm.main([str(inp), str(out), "--root", str(root), "--mode", "extract", "--drop", "1"]) == 1
    assert not out.exists()


@pytest.mark.skipif(not (tree_coder.DEFAULT_ROOT / "icd_rules.py").exists(), reason="decision-models not on this machine")
def test_tree_coder_writes_the_command_contract(tmp_path):
    pytest.importorskip("numpy")
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    recs = [dict(ORIGINAL[1], note="ASSESSMENT AND PLAN\n\n1. Hypertension.\n2. Type 2 diabetes.\n", dialogue=""), ORIGINAL[2]]
    inp.write_text("".join(json.dumps({**r, "icd10_codes": []}) + "\n" for r in recs))
    rc = subprocess.run([sys.executable, "-I", tree_coder.__file__, "--input", str(inp), "--output", str(out)], check=False).returncode
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rc == 0 and [r["encounter_id"] for r in rows] == ["e2", "e3"]
    assert {"I10", "E11.9"} <= set(rows[0]["codes"]) and all(isinstance(c, str) for r in rows for c in r["codes"])
