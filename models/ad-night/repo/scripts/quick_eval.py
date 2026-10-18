"""Quick privacy check on the gold splits (development helper; the full harness is `neural-wave run`)."""
import json, sys
from pathlib import Path
from neural_wave.io import read_jsonl
from neural_wave.config import make_config
from neural_wave.pipeline import anonymize_records
from neural_wave.eval.privacy import score, load_gold

ROOT = Path(__file__).resolve().parents[1]
preset = sys.argv[1] if len(sys.argv) > 1 else "balanced"
splits = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dev"]
for split in splits:
    path = ROOT / ("resources/data/eval_patient_doctor_encounters.jsonl" if split == "eval" else "resources/data/patient_doctor_encounters.jsonl")
    gold = load_gold(ROOT / f"outputs/gold/{split}")
    recs = [r for r in read_jsonl(path) if r["encounter_id"] in gold]
    res = list(anonymize_records(recs, make_config(preset, explain_text=True)))
    sc = score(recs, [o for o, _, _ in res], gold, [d for _, ds, _ in res for d in ds])
    r = sc["removed"]
    print(f"{split:5s} {preset:12s} ALL={r['ALL']} high={r['risk:high']} med={r['risk:medium']} low={r['risk:low']} "
          f"keep={sc['keep_survival']['ALL']} prec={sc['edit_precision']} | names P={r.get('cat:PATIENT_NAME')} O={r.get('cat:PERSON_NAME')} "
          f"date={r.get('cat:DATE')} loc={r.get('cat:LOCATION')} org={r.get('cat:ORGANIZATION')} occ={r.get('cat:OCCUPATION')}")
    (ROOT / "outputs/cache").mkdir(parents=True, exist_ok=True)
    json.dump(sc, open(ROOT / f"outputs/cache/quick_{split}_{preset}.json", "w"), indent=1)
