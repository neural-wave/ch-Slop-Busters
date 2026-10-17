"""External coder: any program that reads challenge-schema JSONL and writes one line of codes per encounter.

The command (an argv list, or a string split like a shell would) gets `{input}`, a JSONL of the records with
`icd10_codes: []`, and writes `{output}`: one {"encounter_id", "codes": [codes, ranked unless the registry says
ranked = false]} per line. Other placeholders: {python} {repo} {container} {here}. It runs with stdin closed and a
timeout (default 3600 s). A record missing from the output, or with a malformed line, is a failed prediction
(`missing_output`, `invalid_output`); every record fails with `timeout` or `exit_<code>` when the command times out or
fails without output. A command that cannot start is a coder failure.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from eval.coders.base import Coder, CoderError, OnResult, Prediction
from eval.common import ADAPTERS, CONTAINER, REPO, sha1_text, write_jsonl


class Command(Coder):
    def __init__(self, template: str | list[str], name: str | None = None, timeout: float = 3600, ranked: bool = True, inputs: str = "both"):
        self.template, self.timeout, self.ranked, self.inputs = template, timeout, ranked, inputs
        text = template if isinstance(template, str) else json.dumps(template)
        self.name = name or "cmd-" + sha1_text(text)[:8]
        self.config = f"cmd:{text}"  # a plain string template keeps its scripts/eval cache key

    def argv(self, inp: Path, outp: Path) -> list[str]:
        parts = shlex.split(self.template) if isinstance(self.template, str) else list(self.template)
        ctx = {"input": inp, "output": outp, "python": sys.executable, "repo": REPO, "container": CONTAINER, "here": ADAPTERS}
        return [p.format(**ctx) for p in parts]

    def predict(self, records: list[dict], on_result: OnResult = None) -> dict[str, Prediction]:
        with tempfile.TemporaryDirectory() as tmp:
            inp, outp = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            write_jsonl(inp, ({**r, "icd10_codes": []} for r in records))
            cmd = self.argv(inp, outp)
            try:
                rc = subprocess.run(cmd, check=False, timeout=self.timeout, stdin=subprocess.DEVNULL).returncode
            except subprocess.TimeoutExpired:
                return {r["encounter_id"]: Prediction([], "timeout") for r in records}
            except OSError as e:
                raise CoderError(f"cannot start {cmd[0]!r}: {type(e).__name__}") from None
            if not outp.exists():
                return {r["encounter_id"]: Prediction([], "missing_output" if rc == 0 else f"exit_{rc}") for r in records}
            got: dict[str, Prediction] = {}
            for line in outp.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line) if line.strip() else None
                except ValueError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("encounter_id"), str):
                    codes = row.get("codes")
                    got[row["encounter_id"]] = Prediction(list(codes)) if isinstance(codes, list) else Prediction([], "invalid_output")
        return {r["encounter_id"]: got.get(r["encounter_id"]) or Prediction([], "missing_output") for r in records}
