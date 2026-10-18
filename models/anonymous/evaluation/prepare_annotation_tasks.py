"""Write batch task files (encounter texts) for the annotation subagents."""
import json
import sys
from pathlib import Path

split, data_file, batch_size, with_codes = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
rows = [json.loads(l) for l in open(data_file)]
out = Path("annotations/tasks")
for b in range(0, len(rows), batch_size):
    parts = []
    for r in rows[b:b + batch_size]:
        codes = "; ".join(f"{c['code']} {c['description']}" for c in r["icd10_codes"]) if with_codes else ""
        parts.append(f"########## ENCOUNTER {r['encounter_id']}\n" + (f"ICD-10 codes: {codes}\n" if codes else "")
                     + f"=== FIELD: dialogue\n{r['dialogue']}\n=== FIELD: note\n{r['note']}\n")
    (out / f"{split}_{b // batch_size:02d}.txt").write_text("\n".join(parts))
    print(out / f"{split}_{b // batch_size:02d}.txt", [r["encounter_id"] for r in rows[b:b + batch_size]])
