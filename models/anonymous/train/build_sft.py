"""Build chunk-level SFT data (excerpt -> JSON PHI spans) from teacher annotations.

python train/build_sft.py   -> outputs/sft/train.jsonl, outputs/sft/eval.jsonl
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import find_all, load_jsonl  # noqa: E402
from detect.llm_prompt import messages, target  # noqa: E402
from detect.ml_layers import chunks  # noqa: E402

CHUNK_CHARS = 1200


def build(split_file, ann_dir):
    out = []
    for rec in load_jsonl(split_file):
        ann = json.load(open(f"{ann_dir}/{rec['encounter_id']}.json"))
        for field in ("dialogue", "note"):
            items = [i for i in ann["phi"] if i["field"] == field]
            for off, piece in chunks(rec[field], CHUNK_CHARS):
                seen, gold = set(), []
                for it in items:
                    key = it["text"].lower()
                    if key not in seen and find_all(piece, it["text"]):
                        seen.add(key)
                        gold.append(it)
                gold.sort(key=lambda i: piece.lower().find(i["text"].lower()))
                out.append({"encounter_id": rec["encounter_id"], "field": field, "offset": off, "text": piece,
                            "gold": gold, "messages": messages(field, piece) + [{"role": "assistant", "content": target(gold)}]})
    return out


if __name__ == "__main__":
    Path("outputs/sft").mkdir(parents=True, exist_ok=True)
    tr = build("data/patient_doctor_encounters.jsonl", "annotations/train")
    ev = build("data/eval_patient_doctor_encounters.jsonl", "annotations/eval")
    random.Random(0).shuffle(tr)
    for name, rows in (("train", tr), ("eval", ev)):
        with open(f"outputs/sft/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), "chunks,", sum(1 for r in rows if not r["gold"]), "empty,",
              sum(len(r["gold"]) for r in rows), "spans")
