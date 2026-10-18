# CLAUDE.md — neural_wave

Patient-data anonymization challenge (Bosch @ Neural Wave Hackathon). **48h sprint.**
Full spec: `resources/specs/NeuralWaveTask_Anonymization of_patient_data_public.pdf` — read it once, then trust this file.

## Goal in one sentence

Anonymize `dialogue` and `note` in patient–doctor encounters so that private information is unrecoverable, while keeping every piece of clinical information needed to predict the encounter's ICD-10 codes.

## Hard constraints (non-negotiable)

1. **Inference must run on-prem**: no cloud/API models in the anonymization path. **Max 4 GB VRAM.** CPU-only is fine. Cloud models (incl. Claude) may be used only for *development* (data exploration, generating synthetic test data, writing code), never inside the pipeline that produces the submission.
2. **`icd10_codes` must be preserved byte-for-byte** (both `code` and `description`). Never touch, reorder or drop them.
3. **Never modify files under `resources/data/`.** Treat them as read-only inputs; write everything to `outputs/`.
4. Output must be a `.jsonl` with the **same schema** as the input (see below). Extra keys are allowed but put explanations/logs in a sidecar file by default so the organizers' eval pipeline is never confused.

## Repo layout

```
neural_wave/
├── resources/
│   ├── data/
│   │   ├── patient_doctor_encounters.jsonl        # train split, 140 encounters
│   │   └── eval_patient_doctor_encounters.jsonl   # eval split,   67 encounters
│   └── specs/…pdf                                 # task description
├── src/neural_wave/                               # package (src layout)
│   └── __init__.py
├── outputs/                                       # generated artifacts (gitignored) — create if missing
├── tests/                                         # pytest — create if missing
├── pyproject.toml / uv.lock / .python-version
└── README.md                                      # deliverable: how to run the tool
```

Planned modules (create as needed, keep flat, no premature abstraction):

```
src/neural_wave/
├── io.py            # load/save JSONL, schema validation
├── detect/          # PII detection: regex rules, NER model(s), medical-eponym disambiguation
├── transform/       # replace / generalize / perturb / date-shift, consistent per-encounter mapping
├── pipeline.py      # orchestrates detect -> transform, emits decisions log
├── eval/            # local proxy metrics: privacy leakage, ICD-10 F1, medical concept recall
└── cli.py           # `neural-wave anonymize|evaluate|attack`
```

## Environment & commands (uv)

```bash
uv sync                                   # create .venv and install deps from uv.lock
uv add <package>                          # add a runtime dependency
uv add --dev pytest ruff                  # dev tooling
uv run python -m neural_wave.cli --help   # run the tool
uv run pytest -q                          # tests
uv run ruff check . && uv run ruff format .
```

- Always use `uv run …` / `uv add …`; never `pip install` into the venv directly, never edit `uv.lock` by hand.
- Python version is pinned in `.python-version` — don't change it.
- Register the CLI in `pyproject.toml` under `[project.scripts]` (`neural-wave = "neural_wave.cli:main"`).
- Models (spaCy, HF, GGUF) must be downloaded once and cached locally; document the download step in `README.md`.

## Data schema (verified on the real files)

One JSON object per line, keys identical across all records:

| key | type | notes |
|---|---|---|
| `encounter_id` | str | e.g. `"D2N088"` |
| `dataset` | str | `aci` (77) / `virtassist` (35) / `virtscribe` (28) in train |
| `dialogue` | str | **anonymize**. Median ~6.2k chars, max ~14k |
| `note` | str | **anonymize**. Median ~2.6k chars, max ~5.7k |
| `icd10_codes` | list[{code, description}] | **preserve**. 1–6 codes per encounter (64 have exactly 1). 146 distinct codes in train, most frequent: `I10`, `E11.9`, `F32.A`, `K21.9`, `M25.561`, `I50.9` |

### `dialogue` format — quirks that matter

- Lines prefixed `[doctor]` / `[patient]`, one turn per line.
- **Entirely lowercase, with tokenized punctuation**: `hi , andrew . how are you ?`, `9/29/1957`, `mr . james`.
  → Cased NER models (standard spaCy/BERT) degrade badly here. Either true-case before NER, use a model robust to lowercase (e.g. GLiNER-style, or a lowercase-tolerant PII model), or rely heavily on rules + gazetteers. **Validate recall on dialogue separately from note.**
- Contains disfluencies (`um`, `uh`, `w- we're`). `dax` is the scribe product name, not a person.
- Patient name appears in the opening lines; doctor sometimes dictates a header: `today i am seeing jose james . date of birth 9/29/1957 …`.
- Doctor names appear as `i'm doctor isabella` / `dr . smith`.

### `note` format

- Clinical note with ALL-CAPS section headers: `CHIEF COMPLAINT`, `HISTORY OF PRESENT ILLNESS`, `REVIEW OF SYSTEMS`, `PHYSICAL EXAM(INATION)`, `RESULTS`, `VITALS`, `ASSESSMENT`, `PLAN`, `ASSESSMENT AND PLAN`, `INSTRUCTIONS`, `SOCIAL HISTORY`, `FAMILY HISTORY`, `MEDICAL HISTORY`, `MEDICATIONS`, `SURGICAL HISTORY`, `ALLERGIES`.
- Properly cased prose. Full name (`Andrew Campbell`) usually in the first HPI sentence, then `Mr. James` / `the patient`.
- Ages appear ~1.4× per encounter (`59-year-old`); DOB in ~15% of encounters; absolute dates in both fields.
- `SOCIAL HISTORY` / HPI carry quasi-identifiers: occupation, hobbies, family members, places, events (`house parties`, `barre classes`, `landscaping`).

### Cross-field consistency

The same patient appears in both `dialogue` and `note`. The same entity must map to the same placeholder/surrogate in both fields of an encounter (e.g. `[PATIENT]` everywhere, or one consistent fake name). Build the entity map per encounter, then apply it to both texts.

## What to remove vs. what to keep

**Remove / generalize (privacy)** — names (patient, doctor, relatives), DOB, exact ages (→ 5-year bucket or decade), absolute dates (→ year or date-shift, keep relative times like `five months ago`), locations, employers/occupations, specific hobbies/events, phone/email/IDs, rare distinctive details that act as quasi-identifiers.

**Keep (utility, drives ICD-10 F1 and concept recall)** — diagnoses, symptoms, review of systems, exam findings, lab values and vitals, medications and dosages, procedures, duration/frequency of symptoms, laterality (`left shoulder`), sex (many codes are sex/age dependent), relative timeline.

**Judgment calls** — only perturb a numeric value if it cannot flip a code (HbA1c, BP, eGFR, BMI *can* change the code: `E11.65`, `I10` vs `I11.0`, CKD stages, obesity classes). Default: keep clinical numbers unchanged; perturb only if a value is so specific it's a fingerprint.

**Eponym disambiguation** — `Alzheimer`, `Parkinson`, `Crohn`, `Hodgkin`, `Graves`, `Paget`, `Cushing`, `Addison`, … are diseases *and* surnames. Rule of thumb: disease if followed by `'s`, `disease`, `syndrome`, or if it matches an ICD-10 description for the encounter; person if preceded by `mr/ms/mrs/dr` or in a greeting position. Keep a medical-eponym whitelist and write a test for each case.

## Evaluation (organizers' metrics — equal weight)

1. **Privacy**: proportion of privacy-related information removed (anonymized vs original).
2. **Utility**: F1 of ICD-10 code prediction on anonymized text + medical concept recall.
3. **Originality & scalability** (jury).
4. **Bonus** (ranked, optional): adversarial re-identification attack, explainability of every decision, efficiency (runtime / peak RAM / peak VRAM), working without `icd10_codes` at inference, robustness to unseen formats.

Trade-off is explicit: more removal → less utility. Every pipeline change must be measured on both axes.

### Local proxy eval (build this in the first hours, `neural_wave/eval`)

- **Privacy proxy**: run an independent PII detector (different model than the one in the pipeline) + regex on the output; count surviving names/dates/ages. Also try to recover entities from the original by string matching against the anonymized text.
- **Utility proxy**: train/fit a cheap ICD-10 predictor (TF-IDF + linear, or embedding + kNN over the 146 train codes) on *original* text, evaluate on *anonymized* eval text; report micro/macro F1 vs the un-anonymized baseline. Zero-shot via `description` similarity works too.
- **Concept recall proxy**: extract medical terms from original and anonymized (scispaCy / a medical vocab / drug name list) and compute recall.
- Print one table per run: `privacy_removed | icd_f1 | concept_recall | runtime | peak_ram`. Save results to `outputs/eval/<run_name>.json`.

## Approach guidance

Preferred stack (all fits in 4 GB VRAM / CPU): rule layer (regex for dates, DOB, ages, MRN-like patterns, `[doctor]`/`[patient]` turn structure) + a small NER/PII model (spaCy / Presidio / GLiNER-type) + optional small local LLM (≤3B, 4-bit via llama.cpp/Ollama/transformers) only for hard cases (quasi-identifiers in free text, eponym disambiguation). Rules first, model second, LLM last — in that order of trust and cost.

Every detection produces a decision record `{field, span, text, category, action, reason, source}` → this is the explainability bonus for free and the input for the attack bonus.

## Working rules for the 48h

- **Hour 0–4: end-to-end baseline.** Load → regex + NER → placeholder replacement → write JSONL → run local eval. Ugly is fine; it must run on the full eval split.
- Iterate in small steps, re-run eval after each, keep a results log. Don't swap the whole architecture after hour 24.
- Determinism: fixed seeds, pinned model versions, no randomness in placeholder assignment unless seeded by `encounter_id`.
- No cloud calls in `src/neural_wave/pipeline.py` or anything it imports. If a function needs the network at inference, it is wrong.
- Prefer boring, well-known libraries over clever ones. One file per concern. Type hints and docstrings on public functions; no heavy class hierarchies.
- Test the nasty cases (`tests/`): lowercase names in dialogue, eponyms, `mr . james`, DOB variants, ages, dates like `9/29/1957` and `september 29th`, numbers that must not be touched (`hba1c 8.3`, `bp 140/90`, dosages).
- Log every skipped/unsure span instead of silently keeping it.
- Commit often with small messages. `outputs/` and model caches are gitignored.
- Use the mid-term evaluation slot (`Jonas.Bergdolt@bosch.com`) once the baseline is stable, not at the end.

## Deliverables (must exist before submission)

1. `outputs/anonymized_eval_patient_doctor_encounters.jsonl` — same schema, `icd10_codes` untouched.
2. Repo with source code + `README.md` describing: install (`uv sync`), model download, `neural-wave anonymize --input … --output …`, `neural-wave evaluate …`, hardware used, and a short description of the approach and trade-offs.
3. Nice-to-have: `outputs/decisions_*.jsonl` (explanations), eval table, attack script and results.

## When unsure

Ask before: changing the output schema, adding a dependency > 1 GB, or removing a clinical detail that might be code-relevant. Don't ask for: naming, file placement within the planned layout, or adding tests.