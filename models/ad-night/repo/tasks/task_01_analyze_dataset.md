# Task 1 — Deep dataset analysis for the anonymization pipeline

Read `CLAUDE.md` first. You are inside the `neural_wave` repo. The data is in `resources/data/` (train: 140 encounters, eval: 67). Treat both files as read-only.

## Why this task exists

Before we write a single line of the anonymizer we need to know **everything in this data that can make an anonymization model fail**: ambiguous tokens, typos, synthetic-data artifacts, formatting quirks, cross-field inconsistencies, and clinical details that look like PII (or PII that looks clinical). The output of this task decides three things:

1. Pure neural vs neuro-symbolic pipeline (rules + model).
2. Whether a deterministic pre-processing script (normalization, detokenization, truecasing) must run before any model.
3. How to fine-tune: which entity categories need a learned model at all, label schema, how to build labels from 140 encounters, and which hard cases go into the test suite.

Be an investigator, not a summarizer. Every claim must be backed by a number (count, % of encounters, per-`dataset` breakdown) and 2–3 verbatim examples with `encounter_id`. Say explicitly when a finding comes from a regex (approximate) vs from reading the text yourself (verified).

## Method

- Work phase by phase in the order below. For each phase: write a script under `scripts/eda/<nn>_<phase>.py`, run it with `uv run`, inspect the output, then write up. Scripts must be re-runnable and must not load the entire corpus into the chat — print aggregates and small samples.
- Keep train and eval statistics **separate** (also report combined). The organizers' test set is non-public with the same structure, so prefer findings about *types* of problems over memorizing specific strings.
- **Read at least 12 encounters end to end by eye** (4 per `dataset`: `aci`, `virtassist`, `virtscribe`), both `dialogue` and `note`. Regex will miss things; note what you only caught by reading.
- Do not use any cloud model. Local tools only (Python stdlib, `regex`, `pandas`, optionally spaCy if already available — do not add dependencies > 500 MB without asking).
- No fixes, no anonymization yet. Only analysis and recommendations.

## Known signals to confirm and extend (already observed in a quick probe)

- `dialogue` is fully lowercase with tokenized punctuation (`hi , andrew .`, `mr . james`, `9/29/1957`). `note` is normal cased prose with ALL-CAPS section headers.
- Speaker tags: `[doctor]`, `[patient]`, and a third speaker **`[patient_guest]`** (~130 lines). At least one transcription artifact: `[ inaudible 00:09:25 ]`.
- Non-ASCII characters present: `•` (~790), `’ “ ”`, narrow no-break space `\u202f` (~35), `\xa0`, zero-width space `\u200b`, `²`, `ß`.
- Patient names are reused across different encounters (`Michelle King` ×3, `Bruce Ward`, `Anna Diaz`, `Sophia Jackson` ×2 …). Determine whether these are the same patient (same age/history) or synthetic name reuse.
- At least one name typo across fields: `jaqueline` in dialogue vs `Jacqueline Price` in note (`D2N188`).
- Eponyms present: `lyme` (~33), `cushing` (~5), `hodgkin`, `wilson`. The word `dax` (~70) is the scribe product name, not a person.
- Ages ~1.4 per encounter, DOB in ~15% of encounters, explicit sex (`male`/`female`) in roughly half of the notes.

## Phases

### 1. Structure and format
Schema, lengths, per-`dataset` differences. Speaker-tag inventory (all variants, including malformed). Note section-header inventory per dataset: which headers exist, order, frequency, empty sections, headers with inconsistent spelling (`PHYSICAL EXAM` vs `PHYSICAL EXAMINATION`, `MEDICATIONS` vs `CURRENT MEDICATIONS`). Bullet/list formatting in notes. Full inventory of non-ASCII and control characters with where they occur and why it matters for tokenization and span offsets.

### 2. Normalization quirks of `dialogue`
Catalogue how these surface in dialogue vs note: titles (`mr .`, `dr .`, `ms .`), abbreviations (`a.m.`, `b.i.d.`, `x-ray`, `copd`), decimals and units (`8.3`, `8 . 3`, `140 / 90`, `140/90`, `mg`, `mcg`), dates and times in every format found, hyphen/stutter tokens (`w- we're`, `fifty-`), contractions, numbers spelled out (`fifty-nine`) vs digits. Conclude: is a detokenizer / truecaser needed before NER, and what would it break (offset mapping back to original text)?

### 3. PII inventory
For each category below: occurrences, encounters affected, field (dialogue/note), note section, position (first N lines?), all surface forms. Build a first-pass regex/gazetteer for each category, measure its hit rate, then hand-check a random sample of ≥20 hits and ≥20 misses to estimate precision/recall.

Categories: patient first name, patient last name, doctor/staff names, relatives and friends (`my wife sarah`), DOB, exact age, absolute dates, relative dates (`five months ago`, `last summer`, `since covid`), clock times, locations (cities, states, countries, neighborhoods, hospital/clinic names), employers / occupations / schools, hobbies, events, sports teams, religion/ethnicity/nationality, phone/email/MRN/IDs/insurance, vehicles, pets, product/brand names, any other quasi-identifier you find.

### 4. Names deep-dive
Extract every person-name candidate from both fields. Then:
- Cross-field consistency: dialogue first name vs note full name; nickname vs formal; spelling variants and typos; encounters where the name appears **only** in one field.
- Name positions: is the patient name always in the first lines? Does the doctor ever dictate a header (`today i am seeing … date of birth …`)? In what % of encounters?
- `mr . james`: is `james` a first or last name? How often is the title + first-name pattern used?
- Collisions: names that are also common words, months, medical terms, drug names, body parts, or eponyms (`bell`, `wilson`, `graves`, `may`, `june`, `hope`, `grace`, `bill`, `mark`, `ray`, `pat`, `sue`, `lyme`, `cushing`, …). For each collision found in the data, show the sentence and say how a model could tell them apart.
- Names shared across encounters: same person or reuse? Impact for a "consistent surrogate per patient" strategy.

### 5. Ambiguous and adversarial tokens
Build a table of **every token class that is PII in one context and clinical in another**:
- Eponymous diseases vs surnames.
- Numbers: DOB / phone / MRN vs BP, HbA1c, dosage, lab values, vitals, dates of procedures, counts (`2 times per week`). Produce a full inventory of numeric patterns with left/right context and label each as `must-keep` (code-relevant), `privacy`, or `neutral`.
- Dates that are clinically meaningful (surgery date, last colonoscopy) vs purely identifying.
- Medication/brand names that look like names or places. Product names (`dax`).
- Lowercase tokens that would be entities if cased (`i'm doctor isabella`).

### 6. Temporal information
Inventory of all temporal expressions and formats. Which can be generalized (year only, date-shift, bucketed age) with zero effect on ICD-10 codes, and which cannot (age-dependent codes, pregnancy weeks, acute vs chronic timing). List any code in the data whose correctness depends on age, sex, laterality, or a numeric threshold, and show where that information lives in the text.

### 7. Clinical content vs ICD-10 codes
For each encounter: lexical overlap between each `icd10_codes.description` and the texts (does the diagnosis appear verbatim, paraphrased, or only as symptoms/labs?). Find codes that are only inferable from numbers, from sex/age, from laterality, or from the note's ASSESSMENT section. Flag "sensitive-but-code-relevant" content (mental health, substance use, HIV, pregnancy, sexual history) — these need a deliberate policy, not a default. Report per-dataset.

### 8. Synthetic-data artifacts
The corpus is scripted/synthetic (ACI-Bench lineage). Hunt for what a model would overfit to: repeated boilerplate sentences across encounters, note sentences copied verbatim from dialogue, near-duplicate encounters (check `D2N181`/`D2N182`), templated openings (scribe preamble about `dax`), fixed name-position patterns, unrealistic uniformity (every patient introduced as `<name> is a <age>-year-old <sex>`), vocabulary that is implausibly clean or implausibly noisy. Check train/eval overlap (identical sentences, identical names, identical codes).

### 9. Typos and noise
Misspellings in names and medical terms, ASR-style errors, disfluencies, inconsistent abbreviations, casing errors inside notes, broken sentences, bracketed artifacts. Quantify and show examples. Say which ones are harmless and which could cause a false negative in PII detection.

### 10. Re-identification risk
Per encounter, list the rarest details (unique occupations, events, places, combinations). Estimate how many encounters are unique on `(age bucket, sex, set of codes)` and on `(age bucket, sex, codes, occupation)`. Identify what an attacker with the original text would use to re-link an anonymized encounter. This feeds the adversarial-attack bonus later.

### 11. Train vs eval shift
Compare splits on `dataset` mix, code distribution, text lengths, PII density, header sets, name overlap. State what this implies for the hidden test set.

## Deliverables

1. `outputs/eda/DATASET_ANALYSIS.md` — the report. Structure:
   - **Executive summary**: the 10 biggest risks for the anonymizer, ranked, one line each with a number.
   - One section per phase: findings with counts, per-dataset breakdown, verbatim examples with `encounter_id`, and the regex/verified tag.
   - **Trap catalogue**: a table `trap | example | why it fools a model | detection strategy (rule / model / both)`.
   - **Decisions**, written as recommendations with reasoning and trade-offs:
     a. Neural vs neuro-symbolic: which categories are solvable by rules with ~100% precision, which need a model, which need both.
     b. Pre-processing script: yes/no, what it should do (unicode cleanup, speaker-tag parsing, detokenization/truecasing), and how to keep an offset map to the original.
     c. Fine-tuning plan: label schema (categories), weak-labeling strategy (rules → model → manual review), expected label count per category, whether 140 encounters are enough, augmentation ideas (name swapping, date swapping), evaluation split.
     d. Policy proposals for the ambiguous classes (clinical numbers, dates, sensitive-but-code-relevant content).
   - **Open questions** we should ask the organizers or settle in the team.
2. `outputs/eda/hard_cases.jsonl` — one line per hard case: `{encounter_id, field, span_text, category, why_hard, expected_action}`. These become the test suite.
3. Machine-readable artifacts in `outputs/eda/`: `pii_inventory.json`, `name_lexicon.json` (all names with field/role/frequency), `collisions.json`, `numeric_patterns.json`, `temporal_expressions.json`, `headers_by_dataset.json`, `train_eval_shift.json`.
4. All scripts under `scripts/eda/`, each runnable with `uv run python scripts/eda/<file>.py`, plus `scripts/eda/README.md` with a one-line description of each.

## Rules

- Do not modify `resources/data/`. Do not write anything outside `outputs/eda/` and `scripts/eda/`.
- Do not start building the anonymizer. If you notice an obvious implementation idea, put it in the Decisions section.
- Prefer small, verifiable scripts over one giant notebook. Commit after each phase.
- When done, reply with: the executive summary, the path of the report, and the three decisions (neural vs neuro-symbolic / pre-processing / fine-tuning) in five lines each.