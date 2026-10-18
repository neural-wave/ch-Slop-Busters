# Neural Wave: shield-first, code-aware anonymization of patient–doctor encounters

Bosch "Anonymization for Patient Data" challenge @ Neural Wave Hackathon. The tool anonymizes the `dialogue`
and `note` of each encounter, keeps `icd10_codes` byte-for-byte, runs fully on-prem (CPU-only, or under
4 GB of VRAM when the optional models are on the GPU), and explains every decision.

## The idea in one paragraph

Normal de-identification looks for PII and deletes it. We do the reverse first: **lock everything that must
survive** (the *medical shield*), then remove private information from what is left. The shield
protects clinical numbers (`bp 140/90`, `4/6 murmur`, `a1c seven . four`), eponyms in medical syntax
(`lyme titer`, `murphy's sign`, `Parkinson's disease`), drug names, and the **evidence for the encounter's
own ICD-10 codes** (the code descriptions plus every ICD-10-CM index synonym that leads to them, so
`Y93.53 Activity, golf` locks `golfing`, `L21.0` locks `dandruff`). Activities told as an injury mechanism
also stay. Then high-precision rules, an **identity graph** that links every anchored name across both
fields (`andrew` ↔ `Andrew Campbell` ↔ `Mr. Campbell`, typo variants such as `jaqueline` ↔ `Jacqueline`), and
optional local models remove the rest. Before writing, a verifier repairs any cross-field leak (it never
reverts an edit) and checks that no shielded span was touched.

```
            ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌───────────┐   ┌──────────┐
 encounter → normalize 1:1  → │ MEDICAL SHIELD│ → │ detect: rules,   │ → │ identity  │ → │ policy   │ → verify → JSONL
            │ (offsets kept)│   │ numbers,      │   │ quasi-identifier │   │ graph:    │   │ (preset) │   leak repair,
            │ speaker tags, │   │ eponyms, drugs│   │ gazetteers,      │   │ link +    │   │ [PATIENT]│   shield check,
            │ number words  │   │ code evidence │   │ deid/GLiNER/LLM  │   │ sweep both│   │ [50s] …  │   schema check
            └──────────────┘   └──────────────┘   └─────────────────┘   │ fields    │   └──────────┘   + decisions.jsonl
                                                                         └───────────┘
```

## Install

```bash
uv sync                      # core pipeline: Python stdlib only (no ML dependencies)
uv sync --all-extras         # + optional detectors (torch, transformers, gliner) and evaluation (numpy, scikit-learn)
```

Optional models, downloaded once and cached locally (inference then runs with `HF_HUB_OFFLINE=1`):

```bash
uv run python -c "from huggingface_hub import snapshot_download as d; d('obi/deid_roberta_i2b2'); d('knowledgator/gliner-pii-base-v1.0')"
ollama pull qwen3.5:2b-q4_K_M      # optional LLM detector (4-bit, 1.56 GB VRAM at num_ctx 4096)
ollama pull bge-m3                 # development-time ICD-10 utility proxy only (never used by the pipeline)
```

The lexicons under `src/neural_wave/data/` (2.2 MB) are built from public sources (ICD-10-CM 2026 tables
and index, FDA NDC, US Census surnames, US baby names, GeoNames, and word frequencies from the 100k-conversation
Kaggle medical corpus). They are committed, and `uv run python scripts/build_lexicons.py` rebuilds them
from the raw downloads.

## Usage

```bash
# Anonymize (the submission file): balanced preset, rules + deid_roberta + GLiNER (auto: GPU while < 4 GB VRAM)
uv run neural-wave anonymize -i resources/data/eval_patient_doctor_encounters.jsonl \
    -o outputs/anonymized_eval_patient_doctor_encounters.jsonl --preset balanced --detectors rules,deid,gliner

# Lightest profile: rules only, CPU, ~50 ms/encounter, ~100 MB RAM, 0 VRAM, no ML dependencies
uv run neural-wave anonymize -i in.jsonl -o out.jsonl

# Without ICD-10 codes at inference (bonus): the pipeline never sees icd10_codes (they are still copied unchanged)
uv run neural-wave anonymize -i in.jsonl -o out.jsonl --no-codes

# One harness row: anonymize a split in a clean process + evaluate privacy / utility / efficiency
uv run neural-wave run --split eval --preset balanced --detectors rules,deid,gliner --icd embed
uv run neural-wave sweep --split eval --detectors rules,deid,gliner     # presets x codes/no-codes curve
uv run neural-wave attack --original resources/data/eval_patient_doctor_encounters.jsonl \
    --anonymized outputs/anonymized_eval_patient_doctor_encounters.jsonl --gold outputs/gold/eval
uv run neural-wave robustness                                            # unseen formats (5 sets, committed with gold)
uv run neural-wave show --id D2N114                                      # one encounter + every decision
uv run pytest -q                                                         # 246 tests: hard cases, invariants, review regressions
```

Every `anonymize` call writes three files:

| file | content |
|---|---|
| `<out>.jsonl` | same schema, record order and keys as the input; `icd10_codes` byte-identical |
| `<out>.decisions.jsonl` | one explanation per decision: offsets, category, action (`REPLACE`/`GENERALIZE`/`KEEP`), replacement, detector + rule, confidence, detector reason, policy reason, entity id, and a readable `explanation`. The original text is **not** copied unless `--explain-text` is given (debug only), so the sidecar does not leak PHI |
| `<out>.summary.json` | runtime, ms/encounter, peak RAM, peak VRAM (measured, 4 GB budget enforced), shield integrity, residual leaks, schema errors, explanation coverage |

### Presets (the privacy/utility curve)

| preset | removes / generalizes |
|---|---|
| `conservative` | names (patient, clinicians, staff, relatives), DOB, IDs, contacts, absolute dates, years, exact ages (decade band) |
| `balanced` (default) | + places, facilities, employers/organizations, insurers, sports teams, occupations, named holidays, hobbies *unless* they are code evidence or the injury/exposure mechanism |
| `aggressive` | + other rare personal details (pets, family structure, brands, events, school grade), coarse ages (`[adult]`), lower name threshold, hobbies kept only as code evidence |

Policy highlights: typed placeholders (`[PATIENT]`, `[CLINICIAN_1]`, `[DOB]`, `[DATE]`, `[LOCATION]`); honorifics
kept (`mr . [PATIENT]`, sex can matter for coding); ages as bands that never contain the exact value (`[50s]`,
`[90+]`, `[infant]`); relative times, durations, laterality, every clinical number and diagnosis kept verbatim; no
"repair" of ASR errors or clinical contradictions; sensitive diagnoses kept (they are the codes).

### Hardware and VRAM

Developed on an Apple M4 Max (48 GB); all reported numbers were re-measured on an Apple M3 Pro (18 GB unified memory,
11 cores). The core pipeline is CPU-only. With
`--device auto` (default), the transformer detectors use the GPU only while the **total** stays under 4 GB.
VRAM is sampled after every model load and forward pass (before cache release) and the peak plus the Ollama
model size is reported in `<out>.summary.json`; models are loaded on the CPU in their final dtype and only then moved to
the GPU, so fp32 weights never sit on the GPU. The measured peak of each configuration is in the results tables below
(column `VRAM MB`). A runtime guard measures torch plus Ollama memory after every call and moves the transformer
models to the CPU (moving first, then upcasting) if the total would exceed 4 GB. Local Ollama calls bypass any HTTP
proxy (`127.0.0.1`, no proxy handler), and Hugging Face loading is forced offline (`local_files_only=True`).

## Local evaluation harness

Organizer metrics are reproduced locally with development-time resources only:

- **Privacy:** gold PII labels for the 67 eval and 40 train-dev encounters (`outputs/gold/`, guidelines in
  `docs/gold_annotation_guidelines.md`, 826 PII strings + 275 must-keep traps). The share of gold occurrences
  removed is reported by risk tier, category, field and dataset, together with keep-trap survival (eponyms,
  code evidence, clinical look-alike numbers) and edit precision.
- **Utility:** medical concept recall (ICD-10-CM index terms, drugs, labs, anatomy), code-evidence retention,
  and two ICD-10 prediction proxies run on original vs anonymized text: zero-shot retrieval with local bge-m3
  embeddings, and a local LLM choosing among the retrieved codes. The predictors never read the gold codes of the
  record they score, but they are given a closed label space (every code that occurs in the provided splits, plus the
  robustness sets' codes) and, for the retrieval proxy, k = the number of gold codes ("oracle k"). Absolute F1 is
  therefore optimistic; the comparison original vs anonymized (and their agreement) is what the proxy measures.
- **Efficiency:** runtime, peak RAM and peak VRAM, measured in a separate process.
- **Robustness:** five unseen-format sets with gold labels (`outputs/robustness/`, `outputs/gold/robust_*`): 500 Kaggle
  discharge summaries (3 layouts), 80 web medical conversations with injected PII, 24 synthetic hospital documents
  (used to develop `formats.py`), and two held-out hospital sets in 8 + 8 further formats written by agents that never saw
  the code. `holdout` (24 documents) was inspected by the code-review agents, so `holdout2` (16 documents, generated
  after all fixes) is the untouched test.
- **Strict leak scoring:** an occurrence also counts as leaked when one of its distinctive tokens (a name/place word or
  a digit run) survives, so a half-removed phone number or hospital name is not counted as removed.
- **Regression:** the 167 hard cases from the dataset analysis, invariant tests, and one or more tests for each of the
  40 defects confirmed by an adversarial review of the harness.

<!-- RESULTS -->
### Results (eval split, 67 encounters; full report: `outputs/eval/REPORT.md`)

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_eval_conservative | 0.740 | 0.991 | 0.845 | 1.000 | 0.984 | 1.000 | 1.000 | 0.235 | 0.235 | 0.935 | 82.3 | 98.4 | 0.000 |
| sweep_eval_balanced | 0.828 | 0.996 | 0.910 | 1.000 | 0.962 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 81.3 | 99.0 | 0.000 |
| sweep_eval_aggressive | 0.869 | 0.996 | 0.921 | 0.984 | 0.902 | 1.000 | 1.000 | 0.235 | 0.235 | 0.922 | 82.2 | 99.0 | 0.000 |
| sweep_eval_balanced_nocodes | 0.832 | 0.996 | 0.915 | 0.994 | 0.956 | 1.000 | 0.999 | 0.235 | 0.235 | 0.928 | 80.2 | 84.8 | 0.000 |
| SUBMISSION balanced+deid+gliner | 0.839 | 0.998 | 0.921 | 1.000 | 0.958 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 2,092.8 | 2,516.7 | 2,123.7 |
| ablation_eval_rules+deid+gliner+llm | 0.842 | 0.998 | 0.924 | 0.994 | 0.953 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 8,351.4 | 1,898.5 | 3,702.3 |

Robustness to unseen formats (`holdout2` = untouched 16-document hospital set, 8 formats nobody tuned on):

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust_holdout2_balanced | 0.710 | 0.771 | 0.719 | 0.926 | 0.953 | 0.995 | 0.953 | 0.298 | 0.362 | 0.723 | 41.9 | 96.8 | 0.000 |
| robust_holdout2_balanced_rules+deid+gliner | 0.891 | 0.928 | 0.902 | 0.842 | 0.930 | 0.995 | 0.953 | 0.298 | 0.319 | 0.660 | 3,474.7 | 1,882.2 | 2,139.7 |
| robust_holdout_balanced_rules+deid+gliner | 0.896 | 0.948 | 0.913 | 0.897 | 0.919 | 0.998 | 0.993 | 0.304 | 0.348 | 0.710 | 1,113.3 | 2,517.0 | 2,163.7 |
| robust_discharge_balanced_rules+deid+gliner | 0.997 | 1.000 | 0.997 | 1.000 | 0.957 | 0.996 | 1.000 | 0.444 | 0.634 | 0.642 | 207.9 | 2,518.8 | 2,097.7 |
| robust_synthetic_balanced_rules+deid+gliner | 0.970 | 0.994 | 0.976 | 0.970 | 0.955 | 0.997 | 0.980 | 0.339 | 0.411 | 0.768 | 864.2 | 2,518.6 | 2,131.7 |
| robust_medconv_balanced_nocodes_rules+deid+gliner | 0.970 | 1.000 | 0.970 | - | 0.923 | 1.000 | - | - | - | - | 358.9 | 2,506.8 | 2,131.7 |

Adversarial attacks on the submission:

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

<!-- /RESULTS -->

## Repository layout

```
src/neural_wave/
  text.py       1:1 normalization, speaker tags, sections, spelled numbers
  shield.py     medical shield (clinical numbers, eponyms, drugs, code evidence, injury context)
  rules.py      high-precision detectors: DOB/dates (6+ formats, spoken), MRNs, ages, name anchors
  quasi.py      places, facilities, employers, insurers, teams, occupations, holidays, activities
  identity.py   per-encounter identity graph: link names across fields, sweep, fuzzy variants
  models.py     optional detectors: deid_roberta, GLiNER, Qwen via Ollama; VRAM placement + guard
  policy.py     placeholders / generalization; pipeline.py orchestration; verify.py leak repair + checks
  cli.py        neural-wave anonymize | evaluate | run | sweep | attack | robustness | show
  eval/         privacy (gold), utility (concepts, ICD proxies), attacks, robustness sets, report
  data/         compact lexicons built by scripts/build_lexicons.py
tests/          hard-case regression suite + invariants
outputs/gold/   development-time gold labels; outputs/eval/ result tables; outputs/eda/ dataset analysis
```

## Limitations (stated honestly)

- `icd10_codes` must stay unchanged, and they alone make 80% of encounters unique in this corpus (EDA §10). The tool
  removes what the *text* reveals; it cannot provide k-anonymity, and anyone who already holds the original record
  can re-link it from the preserved clinical content (attack A5).
- Generalisation to unseen hospital formats is the weak spot: on the untouched held-out set the rules alone remove
  71% of the gold PII (77% of high-risk items); with deid_roberta + GLiNER this rises to 89% (93% high-risk), at the cost
  of more over-removal (keep-trap survival 0.84). On the challenge's own formats high-risk removal is 99.6–100%.
- Generic workplaces without a cue (`working in a bakery`), ASR-garbled places (`the rex center`), nicknames with
  no cue (`dino`) and clinician biography (`my first fall here`) are documented misses (`xfail` tests).
- The gold labels were produced at development time by an LLM annotator following written guidelines, and were
  validated programmatically. They were not reviewed by a second human annotator.
