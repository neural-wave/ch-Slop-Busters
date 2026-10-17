# Evaluation: labels, splits and metrics

All labels in this project are **silver**: written by LLM agents at development time, checked by
scripts, and not reviewed by a human annotator. They are never read by the anonymization pipeline.
Label files contain substrings of patient text and live in the git-ignored `labels/` folder; this page
shows aggregates only.

## Splits

Defined in [configs/splits.json](../configs/splits.json) (encounter ids only).

| Split | Source | Encounters | Role |
|---|---|---:|---|
| `tune` | `data/patient_doctor_encounters.jsonl` (all train) | 140 | Development: error analysis, ablations, acceptance gate |
| `validation` | `data/eval_patient_doctor_encounters.jsonl` | 67 | Checked a few times per milestone. Both source teams used it while developing, so it is reported as validation, not as a test |
| `test` | `data/robustness/holdout.jsonl` | 24 | Untouched held-out hospital documents in 8 formats; scored once, at the end (M7) |
| `dev` | 40 train encounters | 40 | The upstream team's tuning subset; kept only to compare with earlier runs |

## Label sources

| Source | Format | Coverage | How it was made |
|---|---|---|---|
| Alind+Davide (upstream) | one JSON file per encounter: `pii` (field, text, category, risk) and `keep` traps | 40 train + 67 eval + test set | 12 LLM agents following [labeling_guidelines.md](labeling_guidelines.md); validator |
| Ketai | one JSONL line per encounter: `pii` (category, tier), medical concepts, ambiguous terms | 67 eval | LLM annotator pass plus an independent reviewer pass; audit script |
| M2 agents | upstream format | 100 train | 10 LLM annotation agents (10 encounters each) following the same guidelines; every file validated with 0 errors; about 0.97M tokens in total |

## Merged labels (`labels/merged/<split>/`)

Built by `scripts/labels/merge.py` with the logic in `src/unified_harness/eval/labels.py`.

- Categories use the upstream vocabulary. Ketai categories are mapped (`CLINICIAN_NAME` and
  `OTHER_PERSON_NAME` → `PERSON_NAME`, `CONTACT` → `ID`, `OTHER_QUASI` → `OTHER_QI`) and get the
  guideline's default risk.
- Two strings agree when their occurrences overlap in the original field (whole word, any case), so a
  full name in one source and name tokens in the other count as one finding. A string found by only one
  source is added with that source; duplicates are merged.
- A conflict is a string that one source calls private and the other calls clinical (an upstream keep
  trap, or a Ketai medical concept or non-private eponym). Validation had 26 conflicts. One LLM agent
  decided each from a short context window: 6 private, 9 clinical (code evidence), 11 neither.
- Every merged file passes `scripts/labels/validate.py` (231 files, 0 errors).

| Split | Encounters | PII strings | Keep traps | Agreement |
|---|---:|---:|---:|---|
| tune | 140 | 942 | 254 | single source (40 upstream, 100 M2 agents) |
| validation | 67 | 552 | 217 | 481 found by both teams, 32 upstream only, 33 Ketai only, 6 adjudicated |
| test | 24 | 1,246 | 313 | single source (upstream, independently audited) |

Risk mix: tune 721 high / 128 medium / 93 low; validation 379 / 96 / 77; test 853 / 353 / 40.

## Metrics

Computed by `unified-harness evaluate` (and `run`, which anonymizes first in a clean process).

| Metric | Definition |
|---|---|
| privacy | Share of silver-label PII occurrences no longer present (whole word, case-insensitive), overall and by risk tier, category, field and source dataset |
| keep | Share of keep-trap occurrences still present (over-removal check) |
| edit precision | Share of applied edits that overlap a labeled PII occurrence |
| concepts | Medical concept recall with a lexicon extractor independent of the pipeline's shield |
| evidence | Retention of the words supporting the encounter's own ICD-10 codes |
| ICD proxy | bge-m3 retrieval over the code space on original and anonymized text: F1 against the gold codes and agreement between the two |
| ms/enc, RAM, VRAM | Measured in the anonymization process |

## Acceptance gate

A change is kept only when, on `tune`: direct / high-risk removal does not drop; quasi or low-risk
removal rises; keep traps and medical concepts stay at or above 99.5%; ICD proxy agreement does not drop
beyond its run-to-run noise; all tests pass and `icd10_codes` stay byte-identical (acceptance gate above).
