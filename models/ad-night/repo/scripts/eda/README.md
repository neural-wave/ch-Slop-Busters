# EDA scripts (Task 1 — dataset analysis)

All scripts are stdlib-only and read `resources/data/*.jsonl` read-only; outputs go to `outputs/eda/`.
Run any of them with `uv run python scripts/eda/<file>.py`. Order = phase order; each is independent except `piicats.py`/`namelib.py`/`common.py`/`numwords.py`, which are shared modules.

| file | one-line description |
|---|---|
| `common.py` | loaders, dialogue/section parsers, context printer, JSON saver (shared) |
| `namelib.py` | hand-written gazetteers (names, relatives, occupations, hobbies, places, brands, insurers, eponyms) and name-pattern regexes (shared) |
| `numwords.py` | spelled-number → digit spans (`fifty nine`, `one twenty over fifty`, `seven . four`) for `aci` dialogues (shared) |
| `piicats.py` | the phase-3 PII category regexes + stop-word filters, shared by phases 3, 10, 11 |
| `00_read_samples.py` | print whole encounters for reading by eye (`--pick` = the 12 read for the report) |
| `01_structure.py` | schema, lengths, speaker tags, note headers, bullets, non-ASCII inventory → `headers_by_dataset.json` |
| `02_dialogue_normalization.py` | titles, abbreviations, decimals/units, date/time formats, stutters, contractions, number words (dialogue vs note) |
| `03_pii_inventory.py` | per-category PII hits, forms, positions, recall probes → `pii_inventory.json`, `pii_samples.txt` |
| `04_names.py` | per-encounter name lexicon, cross-field consistency, positions, title usage, collisions, shared names → `name_lexicon.json`, `collisions.json` |
| `05_ambiguous.py` | numeric inventory with context classes and must-keep/privacy labels, anchored vs identifying dates, drug/brand look-alikes → `numeric_patterns.json` |
| `06_temporal.py` | temporal expression inventory, codes depending on age/sex/laterality/thresholds → `temporal_expressions.json` |
| `07_clinical_vs_codes.py` | lexical overlap between code descriptions and texts, evidence sections, sensitive-but-coded content |
| `08_artifacts.py` | boilerplate, note-from-dialogue copying, near-duplicates, templates, uniformity, vocabulary noise, train/eval overlap |
| `09_typos.py` | non-dictionary tokens, abbreviation variants, note casing errors, broken turns |
| `10_reid.py` | quasi-identifier profiles, k-anonymity tables, rarest details → `reid_profiles.json` |
| `11_shift.py` | train vs eval: mix, codes, PII density, headers, structure → `train_eval_shift.json` |
| `12_hard_cases.py` | curated hard cases validated against the data → `hard_cases.jsonl` |
| `build_report.py` | concatenates `outputs/eda/sections/*.md` into `outputs/eda/DATASET_ANALYSIS.md` |
