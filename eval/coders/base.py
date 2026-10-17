"""The coder protocol: what every ICD-10 coder exposes to eval/icd.py.

A coder has a `name` (its registry entry and cache folder), a `config` string (everything its predictions depend on:
it keys the cache), `ranked` (True: a ranked list that icd.score cuts at k = number of gold codes; False: a set that
is scored whole), `inputs` ("note": reads only the note; "both": dialogue and note) and
`predict(records, on_result=None) -> {encounter_id: Prediction}`. A coder may call `on_result(record, prediction)`
as each prediction finishes, so a long run is cached as it goes.

A failed prediction has `error` set and empty `codes`: it is scored as empty and never cached. A failure of the
whole coder (server unreachable, three consecutive failed predictions) raises CoderError instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class CoderError(RuntimeError):
    """The coder as a whole failed; scoring records {"error": message} for it and goes on."""


@dataclass
class Prediction:
    codes: list[str]
    error: str | None = None
    attempts: int = 1
    seconds: float = 0.0
    raw_invalid: int = 0  # codes in the raw output that are not ICD-10-CM syntax (dropped)
    usage: dict = field(default_factory=dict)
    backend: str | None = None  # where an LLM ran: gpu, partial or cpu (ollama.backend)


OnResult = Callable[[dict, Prediction], None] | None


class Coder:
    name: str = ""
    config: str = ""
    ranked: bool = True
    inputs: str = "both"
    endpoint: str | None = None  # where the text was sent, when that is not this machine

    def fields(self) -> tuple[str, ...]:
        return ("note",) if self.inputs == "note" else ("dialogue", "note")

    def predict(self, records: list[dict], on_result: OnResult = None) -> dict[str, Prediction]:
        raise NotImplementedError
