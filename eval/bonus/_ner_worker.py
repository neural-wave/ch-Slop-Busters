"""Worker of attack A1: run independent PII detectors on texts and print the spans they find.

Usage: <repo>/models/anonymous/.venv/bin/python -I -B eval/bonus/_ner_worker.py INPUT OUTPUT [--layers regex,deid,gliner]
                                                                                    [--device cpu|cuda] [--threads N] [--root VM_DIR]

INPUT is JSONL `{"key", "text"}`; OUTPUT gets one JSONL line per input line, written as it goes:
`{"key", "spans": [[start, end, text, type, layer, confidence], ...]}`. The detectors are the VM's (models/anonymous, the VM backup;
neuralwave/detect): obi/deid_roberta_i2b2, GLiNER urchade/gliner_multi_pii-v1 and the regex layer (dates, ages, IDs,
contacts), loaded once, offline. It needs torch, transformers, gliner and regex, so the harness (standard library only)
runs it with the anonymous venv's interpreter and `-B`, which keeps the VM folder free of bytecode files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "models" / "anonymous"  # <repo>/eval/bonus/


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--layers", default="regex,deid,gliner")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=4, help="CPU threads of torch (the VM code takes all cores but one)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the VM project folder")
    a = p.parse_args(argv)
    layers = set(a.layers.split(","))
    if a.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("cuda not available, using cpu", file=sys.stderr)
            a.device = "cpu"
    os.environ["NW_DEVICE"] = a.device  # read by ml_layers at import
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.root))
    from detect import regex_layer  # VM code

    if layers & {"deid", "gliner"}:
        import torch
        from detect import ml_layers

        if "deid" in layers:
            ml_layers._deid()  # loads the model and sets torch's thread count, which is capped next
        torch.set_num_threads(max(1, a.threads))
    jobs = [json.loads(line) for line in a.input.open(encoding="utf-8") if line.strip()]
    with a.output.open("w", encoding="utf-8") as out:
        for i, job in enumerate(jobs):
            text, spans = job["text"], []
            if "regex" in layers:
                spans += regex_layer.detect("x", text)
            if "deid" in layers:
                spans += ml_layers.deid_detect("x", text)
            if "gliner" in layers:
                spans += ml_layers.gliner_detect("x", text)
            rows = [[s.start, s.end, text[s.start : s.end], s.type, s.source, round(float(s.confidence), 3)] for s in spans]
            out.write(json.dumps({"key": job["key"], "spans": rows}, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(jobs)} texts", file=sys.stderr, flush=True)
    print(f"done: {len(jobs)} texts, layers={sorted(layers)}, device={a.device}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
