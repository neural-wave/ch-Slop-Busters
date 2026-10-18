"""Deployment gate (TRAINING_PLAN.md §7, gate 2): the exported policy runs within the 4 GB budget at full context.

Start the server first (a merged checkpoint, e.g. the dummy one from rl/export.py --dummy):
  VRAM_GB=3.8 MAX_MODEL_LEN=18944 bash ../term-generalizer/serve_vllm.sh ckpt/export/dummy nw-policy
then
  LLM_BACKEND=vllm python3 rl/gate_deploy.py --model nw-policy --out runs/gate_deploy

Runs the 5 longest and the 5 most PII-dense eval encounters plus one ~16K-token multi-visit record built from train
encounters, while polling nvidia-smi. Pass: every record returns (no server error, no truncated call) and the
nvidia-smi peak is <= --max-mib (3.8 GiB). The untrained or dummy model need not write valid policy JSON; that is
checked on the trained model. Writes <out>/gate.json with the runtime settings, per-record timings and the peak.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
from common import DATA_DIR, EVAL_DATA, GOLD_EVAL, VLLM_URL, read_jsonl  # noqa: E402

import infer  # noqa: E402


class GpuPeak(threading.Thread):
    def __init__(self, every: float = 0.2):
        super().__init__(daemon=True)
        self.every, self.peak, self.stop = every, 0, False

    def run(self):
        while not self.stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()
            if out:
                self.peak = max(self.peak, int(out[0]))
            time.sleep(self.every)


def long_record(target_chars: int = 46000) -> dict:
    """Multi-visit record of consecutive train encounters (~16K tokens at ~3.2 characters per token)."""
    recs = read_jsonl(DATA_DIR / "patient_doctor_encounters.jsonl")
    d, n, codes = [], [], []
    for r in recs:
        d.append(r["dialogue"])
        n.append(r["note"])
        codes += [c for c in r["icd10_codes"] if c not in codes]
        if sum(map(len, d + n)) >= target_chars:
            break
    return {"encounter_id": "LONG16K", "dataset": "multi", "dialogue": "\n".join(d), "note": "\n\n".join(n),
            "icd10_codes": codes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nw-policy")
    ap.add_argument("--out", default="runs/gate_deploy")
    ap.add_argument("--max-mib", type=int, default=int(3.8 * 1024))
    ap.add_argument("--max-prompt", type=int, default=16384)
    ap.add_argument("--max-resp", type=int, default=2560)
    args = ap.parse_args()
    recs = read_jsonl(EVAL_DATA)
    gold = {g["encounter_id"]: g for g in read_jsonl(GOLD_EVAL)}
    longest = sorted(recs, key=lambda r: -len(r["dialogue"] + r["note"]))[:5]
    dense = [r for r in sorted(recs, key=lambda r: -len(gold.get(r["encounter_id"], {}).get("pii", [])))
             if r not in longest][:5]
    todo = longest + dense + [long_record()]
    a = argparse.Namespace(model=args.model, icd=True, max_prompt=args.max_prompt, max_resp=args.max_resp,
                           window_tokens=15000, overlap=512)
    mon = GpuPeak()
    mon.start()
    rows, ok = [], True
    for r in todo:   # sequential: the peak is measured at batch size 1 as well as under the server's own batching
        try:
            _, log = infer.anonymize(r, a)
            trunc = sum(c["finish"] == "length" for c in log["calls"])
            rows.append({"encounter_id": r["encounter_id"], "prompt_tokens": log["prompt_tokens"],
                         "calls": len(log["calls"]), "windows": log.get("windows", 1), "truncated_calls": trunc,
                         "seconds": log["seconds"], "parse_ok": log.get("parse_ok", True)})
            ok &= trunc == 0
        except Exception as e:   # noqa: BLE001 - reported in the gate result
            rows.append({"encounter_id": r["encounter_id"], "error": str(e)[:500]})
            ok = False
        print(rows[-1], flush=True)
    mon.stop = True
    mon.join()
    models = json.loads(__import__("urllib.request").request.urlopen(f"{VLLM_URL}/v1/models").read())
    res = {"pass": ok and mon.peak <= args.max_mib, "nvidia_smi_peak_mib": mon.peak, "max_mib": args.max_mib,
           "server": models, "settings": {"max_prompt": args.max_prompt, "max_resp": args.max_resp,
                                          "note": "nvidia-smi peak covers every process on the GPU: run alone"},
           "records": rows}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "gate.json").write_text(json.dumps(res, indent=1))
    print(f"[gate] {'PASS' if res['pass'] else 'FAIL'}: nvidia-smi peak {mon.peak} MiB (limit {args.max_mib}), "
          f"{sum('error' in r for r in rows)} errors, {sum(r.get('truncated_calls', 0) for r in rows)} truncated calls")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
