"""Assemble outputs/eda/DATASET_ANALYSIS.md from the per-phase section files."""
from common import *

ORDER = ["12_executive_summary.md", "00_reading_notes.md", "01_structure.md", "02_normalization.md", "03_pii_inventory.md", "04_names.md",
         "05_ambiguous.md", "06_temporal.md", "07_clinical_vs_codes.md", "08_artifacts.md", "09_typos.md", "10_reid.md", "11_shift.md",
         "13_traps_decisions_questions.md"]
parts = [(OUT_DIR / "sections" / f).read_text(encoding="utf-8").rstrip() for f in ORDER]
toc = "\n## Contents\n\n" + "\n".join(f"- {l.lstrip('# ').strip()}" for p in parts for l in p.split("\n") if l.startswith("## ")) + "\n"
text = parts[0] + "\n" + toc + "\n" + "\n\n".join(parts[1:]) + "\n"
(OUT_DIR / "DATASET_ANALYSIS.md").write_text(text, encoding="utf-8")
print(f"[saved] outputs/eda/DATASET_ANALYSIS.md ({len(text.split())} words, {len(text)} chars)")
