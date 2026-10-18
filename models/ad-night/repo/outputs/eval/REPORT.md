# Results report

All numbers are produced by `scripts/final_runs.sh` on the current code; JSON details are in `outputs/eval/`.

## Submission: `outputs/anonymized_eval_patient_doctor_encounters.jsonl`

Balanced preset, rules + `obi/deid_roberta_i2b2` + GLiNER-PII, `icd10_codes` used for the shield.

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| submission_eval | 0.839 | 0.998 | 0.921 | 1.000 | 0.958 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 2,092.8 | 2,516.7 | 2,123.7 |

- Privacy (share of the 738 gold PII occurrences removed): **83.9%**; high-risk direct identifiers **99.8%**; high+medium **92.1%**.
- Over-removal: keep-trap survival **100.0%**, edit precision 95.8%, medical concept recall **100.0%**, code-evidence retention 100.0%.
- ICD-10 proxy F1 (bge-m3 retrieval, oracle k): original 0.2353 vs anonymized 0.2353, agreement 0.9281.
- ICD-10 proxy F1 (local Qwen 3.5 2B coder over retrieved candidates): original 0.2337 vs anonymized 0.2606, agreement 0.79.
- Efficiency: 2092.8 ms/encounter, peak RAM 2516.7 MB, peak VRAM 2123.7 MB (budget 4096 MB), placement {'rules': 'cpu', 'deid': 'mps', 'gliner': 'mps'}.
- Integrity: schema errors 0, shield/edit overlaps 0, residual cross-field leaks 0, explained decisions 100.0% of 775.

### Privacy by slice

| slice | removed | occurrences |
|---|---|---|
| cat:PATIENT_NAME | 100.0% | 302/302 |
| cat:AGE | 100.0% | 104/104 |
| cat:OTHER_QI | 28.1% | 25/89 |
| cat:DATE | 97.0% | 65/67 |
| cat:PERSON_NAME | 100.0% | 48/48 |
| cat:OCCUPATION | 26.1% | 12/46 |
| cat:LOCATION | 66.7% | 20/30 |
| cat:ORGANIZATION | 59.1% | 13/22 |
| cat:YEAR | 100.0% | 19/19 |
| cat:DOB | 100.0% | 9/9 |
| cat:ID | 100.0% | 2/2 |
| dataset:virtscribe | 87.3% | 240/275 |
| dataset:aci | 72.6% | 196/270 |
| dataset:virtassist | 94.8% | 183/193 |
| field:dialogue | 80.4% | 332/413 |
| field:note | 88.3% | 287/325 |
| risk:high+medium | 92.1% | 593/644 |
| risk:high | 99.8% | 528/529 |
| risk:medium | 56.5% | 65/115 |
| risk:low | 27.7% | 26/94 |

## Privacy / utility curve on `eval` (presets x codes, rules only vs with models)

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_eval_conservative | 0.740 | 0.991 | 0.845 | 1.000 | 0.984 | 1.000 | 1.000 | 0.235 | 0.235 | 0.935 | 82.3 | 98.4 | 0.000 |
| sweep_eval_balanced | 0.828 | 0.996 | 0.910 | 1.000 | 0.962 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 81.3 | 99.0 | 0.000 |
| sweep_eval_aggressive | 0.869 | 0.996 | 0.921 | 0.984 | 0.902 | 1.000 | 1.000 | 0.235 | 0.235 | 0.922 | 82.2 | 99.0 | 0.000 |
| sweep_eval_balanced_nocodes | 0.832 | 0.996 | 0.915 | 0.994 | 0.956 | 1.000 | 0.999 | 0.235 | 0.235 | 0.928 | 80.2 | 84.8 | 0.000 |
| sweep_eval_conservative_rules+deid+gliner | 0.745 | 0.998 | 0.851 | 1.000 | 0.984 | 1.000 | 1.000 | 0.235 | 0.235 | 0.935 | 2,210.9 | 2,137.2 | 2,139.7 |
| sweep_eval_balanced_rules+deid+gliner | 0.839 | 0.998 | 0.921 | 1.000 | 0.958 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 2,014.6 | 2,412.8 | 2,123.7 |
| sweep_eval_aggressive_rules+deid+gliner | 0.879 | 0.998 | 0.932 | 0.984 | 0.900 | 1.000 | 1.000 | 0.235 | 0.235 | 0.922 | 2,102.7 | 2,519.2 | 2,123.7 |
| sweep_eval_balanced_nocodes_rules+deid+gliner | 0.843 | 0.998 | 0.925 | 0.994 | 0.953 | 1.000 | 0.999 | 0.235 | 0.235 | 0.928 | 1,936.2 | 2,506.8 | 2,123.7 |

## Privacy / utility curve on `dev` (presets x codes, rules only vs with models)

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_dev_conservative | 0.753 | 0.982 | 0.878 | 1.000 | 0.982 | 1.000 | 1.000 | 0.244 | 0.256 | 0.987 | 86.0 | 98.2 | 0.000 |
| sweep_dev_balanced | 0.865 | 1.000 | 0.970 | 1.000 | 0.954 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 85.2 | 97.5 | 0.000 |
| sweep_dev_aggressive | 0.928 | 1.000 | 0.970 | 0.951 | 0.897 | 1.000 | 1.000 | 0.244 | 0.256 | 0.949 | 82.4 | 97.9 | 0.000 |
| sweep_dev_balanced_nocodes | 0.865 | 1.000 | 0.970 | 0.993 | 0.951 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 83.0 | 84.3 | 0.000 |
| sweep_dev_conservative_rules+deid+gliner | 0.758 | 0.985 | 0.881 | 1.000 | 0.976 | 1.000 | 1.000 | 0.244 | 0.256 | 0.987 | 2,021.2 | 2,240.5 | 2,123.7 |
| sweep_dev_balanced_rules+deid+gliner | 0.879 | 1.000 | 0.970 | 1.000 | 0.943 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 2,057.2 | 2,518.7 | 2,123.7 |
| sweep_dev_aggressive_rules+deid+gliner | 0.942 | 1.000 | 0.970 | 0.951 | 0.892 | 1.000 | 1.000 | 0.244 | 0.256 | 0.949 | 2,042.5 | 2,519.1 | 2,123.7 |
| sweep_dev_balanced_nocodes_rules+deid+gliner | 0.879 | 1.000 | 0.970 | 0.993 | 0.940 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 2,301.0 | 2,509.1 | 2,123.7 |

## Detector ablation on `eval` (balanced)

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_eval_balanced | 0.828 | 0.996 | 0.910 | 1.000 | 0.962 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 81.3 | 99.0 | 0.000 |
| ablation_eval_rules+deid | 0.836 | 0.998 | 0.919 | 1.000 | 0.961 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 1,139.5 | 2,524.5 | 1,040.7 |
| ablation_eval_rules+gliner | 0.833 | 0.996 | 0.915 | 1.000 | 0.959 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 1,122.1 | 1,943.0 | 1,059.7 |
| sweep_eval_balanced_rules+deid+gliner | 0.839 | 0.998 | 0.921 | 1.000 | 0.958 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 2,014.6 | 2,412.8 | 2,123.7 |
| ablation_eval_rules+llm | 0.835 | 0.996 | 0.918 | 0.994 | 0.954 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 5,925.0 | 271.0 | 1,581.1 |
| ablation_eval_rules+deid+gliner+llm | 0.842 | 0.998 | 0.924 | 0.994 | 0.953 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 8,351.4 | 1,898.5 | 3,702.3 |

## Detector ablation on `dev` (balanced)

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_dev_balanced | 0.865 | 1.000 | 0.970 | 1.000 | 0.954 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 85.2 | 97.5 | 0.000 |
| ablation_dev_rules+deid | 0.870 | 1.000 | 0.970 | 1.000 | 0.949 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 871.2 | 1,886.9 | 1,032.7 |
| ablation_dev_rules+gliner | 0.877 | 1.000 | 0.970 | 1.000 | 0.945 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 1,054.3 | 1,759.1 | 1,059.7 |
| sweep_dev_balanced_rules+deid+gliner | 0.879 | 1.000 | 0.970 | 1.000 | 0.943 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 2,057.2 | 2,518.7 | 2,123.7 |
| ablation_dev_rules+llm | 0.865 | 1.000 | 0.970 | 1.000 | 0.951 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 6,573.0 | 266.4 | 1,581.1 |
| ablation_dev_rules+deid+gliner+llm | 0.879 | 1.000 | 0.970 | 1.000 | 0.940 | 1.000 | 1.000 | 0.244 | 0.256 | 0.974 | 8,843.2 | 1,648.6 | 3,702.3 |

## Adversarial attacks on the submission (eval split)

Every attack also runs on the original text as a positive control.

| attack | original text | anonymized text |
|---|---|---|
| A1 independent NER (deid_roberta + GLiNER + rules) recovers gold PII (high+medium) | 85.4% | 0.4% |
| A2 roster dictionary attack (2236 names) finds person names | 100.0% | 0.0% |
| A4 local LLM guesses the patient's name | 95.2% (60/63) | 0.0% (0/63) |
| A4 local LLM guesses the exact age (+-1 y) | 100.0% (54/54) | 20.4% (11/54) |
| A4 local LLM guesses the place | 15.4% (2/13) | 7.7% (1/13) |
| A4 local LLM guesses employer/job | 34.8% (8/23) | 26.1% (6/23) |
| A3 rare identifier-like words (corpus-unique names/places/employers) still present | 13 | 1 |
| A5 TF-IDF re-link to the original record (attacker holds originals) | - | 100.0% |

### Same attacks on the rules-only balanced output

| attack | original text | anonymized text |
|---|---|---|
| A2 roster dictionary attack (2236 names) finds person names | 100.0% | 0.0% |
| A4 local LLM guesses the patient's name | 95.2% (60/63) | 0.0% (0/63) |
| A4 local LLM guesses the exact age (+-1 y) | 100.0% (54/54) | 20.4% (11/54) |
| A4 local LLM guesses the place | 15.4% (2/13) | 7.7% (1/13) |
| A4 local LLM guesses employer/job | 34.8% (8/23) | 26.1% (6/23) |
| A3 rare identifier-like words (corpus-unique names/places/employers) still present | 13 | 2 |
| A5 TF-IDF re-link to the original record (attacker holds originals) | - | 100.0% |

## Robustness: untuned first pass on the synthetic hospital documents

Measured once, before any rule was written for these formats (and scored with the earlier whole-string metric, which is more lenient than the strict token-level scoring used everywhere else). The `synthetic` set was then used to develop `formats.py`, so its later numbers are development numbers; the `holdout` set (24 documents in 8 other hospital formats, written by an agent that never saw the code) is the untouched test.

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic FIRST PASS (before hospital-format rules), rules | 0.745 | 0.824 | 0.749 | 0.963 | 0.955 | 0.975 | 0.974 | 0.339 | 0.375 | 0.732 | 22.1 | 94.4 | 0.000 |
| synthetic FIRST PASS, rules+deid+gliner | 0.921 | 0.968 | 0.926 | 0.933 | 0.935 | 0.970 | 0.974 | 0.339 | 0.393 | 0.750 | 485.7 | 2,527.3 | 2,059.7 |

## Robustness to unseen formats (rules only)

`holdout` = 24 untouched hospital documents (ICU, operative, psychiatry intake, paediatrics, ambulance handover, medication reconciliation, home health, portal chat); `discharge` = 500 Kaggle discharge summaries rendered in 3 layouts; `synthetic` = 24 hospital documents used to develop the format rules; `medconv` = 80 web medical conversations with injected PII.

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust_holdout2_balanced | 0.710 | 0.771 | 0.719 | 0.926 | 0.953 | 0.995 | 0.953 | 0.298 | 0.362 | 0.723 | 41.9 | 96.8 | 0.000 |
| robust_holdout2_balanced_nocodes | 0.714 | 0.777 | 0.723 | 0.921 | 0.951 | 0.995 | 0.906 | 0.298 | 0.319 | 0.723 | 35.9 | 82.7 | 0.000 |
| robust_holdout_balanced | 0.758 | 0.858 | 0.773 | 0.959 | 0.945 | 0.998 | 0.997 | 0.304 | 0.319 | 0.768 | 37.3 | 97.1 | 0.000 |
| robust_holdout_balanced_nocodes | 0.758 | 0.858 | 0.773 | 0.941 | 0.940 | 0.998 | 0.970 | 0.304 | 0.304 | 0.768 | 34.2 | 83.0 | 0.000 |
| robust_discharge_balanced | 0.992 | 1.000 | 0.992 | 1.000 | 0.994 | 0.996 | 1.000 | 0.444 | 0.632 | 0.640 | 5.6 | 97.8 | 0.000 |
| robust_discharge_balanced_nocodes | 0.992 | 1.000 | 0.992 | 1.000 | 0.994 | 0.996 | 1.000 | 0.444 | 0.632 | 0.640 | 5.5 | 84.0 | 0.000 |
| robust_synthetic_balanced | 0.958 | 0.987 | 0.964 | 1.000 | 0.984 | 0.997 | 0.980 | 0.339 | 0.393 | 0.750 | 26.5 | 96.8 | 0.000 |
| robust_synthetic_balanced_nocodes | 0.958 | 0.987 | 0.964 | 0.985 | 0.977 | 0.993 | 0.941 | 0.339 | 0.393 | 0.732 | 24.1 | 82.8 | 0.000 |
| robust_medconv_balanced_nocodes | 0.970 | 1.000 | 0.970 | - | 0.936 | 1.000 | - | - | - | - | 11.3 | 82.8 | 0.000 |

## Robustness to unseen formats (rules + models)

`holdout` = 24 untouched hospital documents (ICU, operative, psychiatry intake, paediatrics, ambulance handover, medication reconciliation, home health, portal chat); `discharge` = 500 Kaggle discharge summaries rendered in 3 layouts; `synthetic` = 24 hospital documents used to develop the format rules; `medconv` = 80 web medical conversations with injected PII.

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust_holdout2_balanced_rules+deid+gliner | 0.891 | 0.928 | 0.902 | 0.842 | 0.930 | 0.995 | 0.953 | 0.298 | 0.319 | 0.660 | 3,474.7 | 1,882.2 | 2,139.7 |
| robust_holdout2_balanced_nocodes_rules+deid+gliner | 0.895 | 0.934 | 0.907 | 0.837 | 0.929 | 0.995 | 0.906 | 0.298 | 0.277 | 0.660 | 1,366.1 | 2,166.9 | 2,139.7 |
| robust_holdout_balanced_rules+deid+gliner | 0.896 | 0.948 | 0.913 | 0.897 | 0.919 | 0.998 | 0.993 | 0.304 | 0.348 | 0.710 | 1,113.3 | 2,517.0 | 2,163.7 |
| robust_holdout_balanced_nocodes_rules+deid+gliner | 0.896 | 0.948 | 0.913 | 0.879 | 0.914 | 0.998 | 0.960 | 0.304 | 0.319 | 0.725 | 1,090.5 | 2,505.6 | 2,171.7 |
| robust_discharge_balanced_rules+deid+gliner | 0.997 | 1.000 | 0.997 | 1.000 | 0.957 | 0.996 | 1.000 | 0.444 | 0.634 | 0.642 | 207.9 | 2,518.8 | 2,097.7 |
| robust_discharge_balanced_nocodes_rules+deid+gliner | 0.997 | 1.000 | 0.997 | 1.000 | 0.957 | 0.996 | 1.000 | 0.444 | 0.634 | 0.642 | 201.0 | 2,505.0 | 2,097.7 |
| robust_synthetic_balanced_rules+deid+gliner | 0.970 | 0.994 | 0.976 | 0.970 | 0.955 | 0.997 | 0.980 | 0.339 | 0.411 | 0.768 | 864.2 | 2,518.6 | 2,131.7 |
| robust_synthetic_balanced_nocodes_rules+deid+gliner | 0.970 | 0.994 | 0.976 | 0.955 | 0.949 | 0.993 | 0.941 | 0.339 | 0.411 | 0.750 | 904.1 | 2,504.7 | 2,131.7 |
| robust_medconv_balanced_nocodes_rules+deid+gliner | 0.970 | 1.000 | 0.970 | - | 0.923 | 1.000 | - | - | - | - | 358.9 | 2,506.8 | 2,131.7 |
