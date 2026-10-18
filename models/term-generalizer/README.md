# Anonymization baseline harness

A zero-shot LLM baseline for the Neural Wave / Bosch patient-data anonymization task, plus a local
evaluator that approximates the organizers' metrics on `data/eval_patient_doctor_encounters.jsonl`.

Everything runs on-prem through [Ollama](https://ollama.com) (or vLLM for development, see below). Only the
Python 3.10+ standard library is needed. The only inputs are the customer files in `data/`; `../term-generalizer/gold/` is a
reference annotation that LLM agents derived from `data/`, not organizer labels (see [Evaluator](#evaluator-evaluatepy)).

The commands below run from `../qwen-finetune/`, which holds `data/`, `runs/`, the local model checkpoints and the
trained policy (`rl/`, needed by `--mode policy`).

## Quick start

```bash
ollama pull qwen3.5:2b      # anonymizer and ICD-10 coder (fits the 4 GB VRAM budget)

bash ../term-generalizer/run_baseline.sh qwen3.5:2b qwen3.5:2b   # anonymize + evaluate + summary table
cat runs/summary.md
```

Or step by step:

```bash
python3 ../term-generalizer/anonymize.py --mode extract --model qwen3.5:2b --out runs/my_run   # --limit 5 for a quick try
python3 ../term-generalizer/evaluate.py  --anon runs/my_run/anonymized.jsonl                    # --skip-icd for a fast pass
python3 ../term-generalizer/summarize.py runs/*
```

### vLLM backend (GPU node, fast baselines and RL rollouts)

`LLM_BACKEND=vllm` sends every call to an OpenAI-compatible `vllm serve` endpoint (`VLLM_URL`, default
`http://localhost:8000`) instead of Ollama, with the same settings (JSON-schema output, thinking off, `temperature=0`,
`presence_penalty=0`, `seed=0`). `--model`/`--coder-model` are the server's served name. Setup notes are in `serve_vllm.sh`.

```bash
bash ../term-generalizer/serve_vllm.sh                      # models/Qwen3.5-2B served as qwen3.5-2b on :8000 (~2.5 min to ready)
LLM_BACKEND=vllm python3 ../term-generalizer/anonymize.py --mode extract --model qwen3.5-2b --out runs/vllm-qwen3.5-2b_extract --workers 32
LLM_BACKEND=vllm WORKERS=32 CODER_WORKERS=16 bash ../term-generalizer/run_baseline.sh qwen3.5-2b qwen3.5-2b   # run dirs: runs/vllm-*
```

vLLM serves BF16 weights while Ollama serves a quantized GGUF, so numbers differ slightly between backends; ICD coder caches
are separate because the model names differ (`qwen3.5-2b` vs `qwen3.5:2b`).

**VRAM budget.** The task allows at most 4 GB of VRAM for anonymization at evaluation time (more is allowed during
development). The default `serve_vllm.sh` is a development setup: the BF16 checkpoint alone is 4.3 GB and the server
reserves half of the GPU for KV cache. `VRAM_GB=4 bash ../term-generalizer/serve_vllm.sh` serves the same model within 4 GiB:
FP8 weights (2.4 GiB), no CUDA graphs, at most 8 sequences and 2,048 prefill tokens per step, and a KV cache sized
to fit. vLLM does not count about 550 MiB (CUDA context, kernel workspaces) against `--gpu-memory-utilization`, so
the script holds back 600 MiB. Measured with `nvidia-smi` on an L40S with 8 concurrent encounters, the server used
3,942 MiB idle and 4,000 MiB at peak. Use this setup, or Ollama's quantized `qwen3.5:2b` (check `ollama ps`), to
produce the deliverable:

```bash
VRAM_GB=4 bash ../term-generalizer/serve_vllm.sh models/Qwen3.5-2B qwen3.5-2b 8001
LLM_BACKEND=vllm VLLM_URL=http://localhost:8001 python3 ../term-generalizer/anonymize.py --mode extract --model qwen3.5-2b \
  --out runs/vllm4gb-qwen3.5-2b_extract --workers 8
python3 ../term-generalizer/anonymize.py --mode generalize --model qwen3.5-2b --out runs/vllm4gb-qwen3.5-2b_generalize \
  --reuse-entities runs/vllm4gb-qwen3.5-2b_extract/anonymization_log.jsonl   # deliverable: .../anonymized.jsonl
```

The trained policy (SFT/PPO, `rl/`) runs through the same harness with `--mode policy`. It needs an exported checkpoint
(`rl/export.py`) served at fp8 and a context of 18,944 tokens:

```bash
VRAM_GB=3.8 MAX_MODEL_LEN=18944 bash ../term-generalizer/serve_vllm.sh ckpt/export/pilot_ppo nw-policy
LLM_BACKEND=vllm python3 ../term-generalizer/anonymize.py --mode policy --model nw-policy --out runs/vllm4gb-pilot-ppo_policy --workers 8
```

Repeated runs of the same checkpoint are not identical: concurrent requests are batched differently at fp8, and that
changed 14 of 67 records and PII removal by ~2.6 points between two runs of the pilot model. Treat differences of
that size between runs as noise.

The ICD-10 coder in `evaluate.py` stands in for the organizers' evaluation and is not part of the anonymizer, so it
is not bound by the budget.

Runs are resumable: `anonymize.py` keeps `anonymized.partial.jsonl` and skips finished encounters,
and ICD coder outputs are cached in `../term-generalizer/cache/` (so the original-text predictions are computed once).

## Anonymizer baselines (`anonymize.py`)

All LLM modes use a zero-shot prompt (task description and label set, no examples), `temperature=0`,
thinking off, and process each field in ~2,000-character chunks split on line boundaries.

| mode | what the LLM does | what the code does |
|---|---|---|
| `extract` | returns `{"entities":[{"text","type"}]}` (Ollama structured output) | replaces each returned span (case-insensitive, whole word) with `[TYPE]`; spans not found in the chunk and `[doctor]`/`[patient]` tags are ignored |
| `generalize` | same as `extract` | adds a regex pass for ages and dates (`generalize.py`) and replaces AGE/DATE spans with vague ranges instead of tags; other types keep `[TYPE]`; keeps spans made only of words of the encounter's ICD-10 descriptions; logs a reason for every decision |
| `rewrite` | rewrites the chunk with placeholders inlined | strips code fences, nothing else |
| `identity` | nothing | control run: utility upper bound, zero privacy |

Placeholders: `[PERSON] [DATE] [AGE] [LOCATION] [ORGANIZATION] [OCCUPATION] [CONTACT] [ID]`.
`icd10_codes` are copied unchanged. `presence_penalty` is pinned to 0 because the qwen3.5 default (1.5)
penalizes copying input tokens.

Output per run directory: `anonymized.jsonl` (the deliverable format), `anonymization_log.jsonl`
(raw model output, entities, tokens and latency per chunk; in `generalize` also `regex_spans`, the
`applied` spans with their `source` (`llm`, `regex` or `both`), `action` (`removed` or `generalized`),
`replacement` and `reason`, and the `retained` spans with the reason they were kept), `anonymize_meta.json`
(mode, model, backend, input path relative to the repo).

Truncated replies: the 2B model sometimes repeats one entity until the token limit (`done_reason: length`; 8-13 of
400 chunks on the eval split), which leaves the JSON unfinished. Every finished `{...}` item is still used
(`salvaged` in the chunk log); before this, the whole chunk went out unanonymized. Replays re-read the stored raw
reply, so old logs benefit too. The ICD coder salvages its codes the same way.

### Generalize mode (ages and dates as ranges)

`[AGE]` hides child vs adult vs elderly, which matters for ICD-10, and the 2B model misses many ages.
`generalize` keeps the extract spans, adds a deterministic regex pass, and writes ranges as words, so no
digit of the true value survives and the evaluator cannot count a range endpoint as a leaked gold string.

| age | written as | | date | written as |
|---|---|---|---|---|
| < 1 month | newborn | | with a year (`4/24/2021`, `october 2019`, `'21`) | `early/mid/late <decade>s` (year % 10: 0-3, 4-6, 7-9) |
| 1-11 months | infant | | month(s), no year (`july 8th`) | season(s) (`summer`) |
| 1-2 years | toddler | | other calendar date (`the 28th`, `monday`, `my birthday`, `6421`) | `[DATE]` |
| 3-4 / 5-12 / 13-17 / 18-19 | preschool-age / school-age / teenage / young-adult | | no calendar anchor (`two years ago`, `48 hours`, `today`) | kept as is |
| 20-89 | `forty-something` etc. | | | |
| >= 90 | ninety-plus (HIPAA Safe Harbor's "90 or older") | | | |

`an 8-year-old boy` → `a school-age boy`, `44-year-old female` → `forty-something female`, `age 7` → `school age`,
`since she was 4` → `since she was preschool-age`, `almost 40` → `thirty-something`. The regex pass only takes
unambiguous forms (N-year/month-old, N years old/of age, N y.o., age/aged N, since/when … was N, i'm/she is N,
turned N, numeric dates with a year, month + day, month + year). It skips guideline and gestational ages
(`under the age of 50`, `9 to 11 years old`, `34+3-week-old`), pain scores (`7/10`), blood pressure, durations
(`2 years ago`, `3-month follow-up`), decades (`in his 40s`) and bare month names or years (`may`, `2000 mg`),
which only the LLM can mark. Tests: `python3 ../term-generalizer/test_generalize.py`.

**Relative time is kept.** The 2B model labels many non-dates as DATE: durations (`about two years ago`, `the last
48 hours`), doses (`40 milligrams`), vitals (`ninety eight percent`, `180 over 95`) and pain scores (`four out of
ten`). On the vLLM eval run these were 188 of the 292 DATE spans. A DATE span with no calendar anchor (month, year,
numeric date, day ordinal, weekday, holiday, birthday, or a bare number that may be a garbled date) names no point in
time. It is kept and logged as `retained`, because onset, duration and findings matter for ICD-10 (acute vs. chronic)
and the task's own example keeps lab values. This raised reference concept recall from 93.4% to 94.2% and left
direct-identifier removal unchanged. The cost is that a quasi-identifier the model mislabels as DATE (`1% disability`)
is now kept: quasi removal fell from 67.1% to 66.3%.

**ICD-10 hints.** The task makes every field available at inference, `icd10_codes` included, and asks to tell
diagnoses apart from names (Alzheimer, Parkinson) and to preserve ICD-relevant information. So `generalize` drops an
LLM span (keeps the text) when every word of it occurs in the encounter's own ICD-10 descriptions, ignoring generic
words such as *type*, *unspecified* and *other*, with *left*/*right* also allowed. Ages, dates, contacts and IDs are
never kept this way. On the eval split the 2B model had tagged `diabetes` and `back pain` as ORGANIZATION and
`knee`, `right hip` and `right breast` as LOCATION; all of these are now retained. `--no-icd-hints` turns this off.
That is the bonus setting where codes are unknown, and `run_baseline.sh` writes it as `<tag>_generalize_noicd`.

**Explanations.** Every anonymization decision in `generalize` is logged with a reason (bonus task "Explainability"),
e.g. `{"text": "44-year-old", "type": "AGE", "action": "generalized", "replacement": "forty-something", "reason":
"exact age: a quasi-identifier; only the age band is kept (...)"}` or a `retained` entry for a diagnosis word.

### Replaying entities (`--reuse-entities`)

`--reuse-entities runs/<run>/anonymization_log.jsonl` replays the per-chunk entities of an earlier `extract` or
`generalize` run made with the same `--model` and `--chunk-chars`, so no model is called. Chunks are re-derived
deterministically; if any chunk is missing from the log the run stops before doing any work. `anonymize_meta.json`
records `entities_reused_from`. Replaying an extract log in `--mode extract` reproduces that run's
`anonymized.jsonl` byte for byte, so a `generalize` run built this way differs from its extract run only by the
regex pass and the range rendering.

## Evaluator (`evaluate.py`)

The organizers' pipeline and their PII labels are not public. The evaluator therefore reports two kinds of metrics:

- **data-only**: computed from `data/` alone (the records and their `icd10_codes`): ICD-10 F1, ICD description term
  retention, integrity. These need no annotation.
- **reference**: computed from a reference annotation of the 67 eval encounters (`../term-generalizer/gold/eval_gold.jsonl`). An
  annotator pass and an independent reviewer pass of Claude agents produced it from `data/` at development time.
  It is not ground truth. It holds 484 PII strings, 4,334 medical concepts and 173 ambiguous terms such as eponyms
  versus surnames. All spans are verbatim substrings of the source (`gold/build_gold.py` checks this).

`python3 ../term-generalizer/gold/audit_gold.py` cross-checks the annotation against `data/` without a model. It flags missing
encounters, non-verbatim spans, strings labelled both PII and medical, PII made of the encounter's ICD-10 description
words, and deterministic PII candidates no gold string covers: regex ages and dates, `dr./mr./mrs./ms. <name>`,
emails, phones and record numbers. On the current file it finds one candidate, `turn 50` in a colonoscopy guideline
sentence, which is correctly not PII. It also shows that 69 of the 484 PII strings are judgment calls (`OTHER_QUASI`,
`OCCUPATION`: hobbies, family facts, work context; `--verbose` lists them).

| metric | definition |
|---|---|
| *reference metrics* | |
| PII mention-level removal | `1 - Σ min(#left, #orig) / Σ #orig` over gold PII strings (case/spacing-insensitive whole-word match) |
| PII entity-level removal | share of gold PII strings with no remaining mention |
| name-token removal | mention-level removal per name token, so `[PERSON] Collins` still counts `Collins` as leaked |
| by tier | `direct` = names, dates, contacts, IDs; `quasi` = age, location, organization, occupation, other rare personal details |
| medical concept recall | share of gold medical concepts still present (all, and the ICD-relevant subset) |
| ambiguous terms | eponyms kept (non-PII ambiguous terms that are also annotated as medical concepts, e.g. *Cushing's*, *Lisfranc*) vs. person names that look like diseases removed. Product names and small talk the annotator also marked ambiguous (*dax*, *dragon*, *tennis*) are not counted |
| *data-only metrics* | |
| ICD term retention | mention-level share of the words of each encounter's own ICD-10 descriptions (*diabetes*, *knee*, *sprain*) still present after anonymization |
| ICD-10 F1 | a fixed zero-shot coder (default `qwen3.5:2b`) predicts codes from the original and from the anonymized text; micro P/R/F1 vs the `icd10_codes` in `data/` at full-code and 3-character category level, plus agreement F1 between the two predictions |
| integrity | `icd10_codes` unchanged, missing records, length ratio |

Outputs: `metrics.json`, `report.md`, `per_encounter.jsonl` (leaked PII, lost concepts, lost ICD terms and predicted
codes per encounter).

### Caveats
- The reference PII labels are model-generated and only approximate the organizers' definition. Quasi-identifiers
  in particular (`OTHER_QUASI`, `OCCUPATION`) involve judgment, so compare runs on the `direct` tier as well as
  overall, and rerun `audit_gold.py` after editing `gold/eval/*.json`.
- ICD term retention only sees the exact description words. It misses synonyms (*high blood pressure* for
  *hypertension*), so it is a lower bound on kept ICD information and is blind to paraphrasing anonymizers.
- String-match recall understates utility for rewrite-style anonymizers that paraphrase correctly, and it counts
  a concept as lost when an anonymizer perturbs a value (e.g. HbA1c 8.3% → 8.1%).
- ICD F1 depends on the coder. Compare anonymized against original F1 from the same coder, not across coders.
- The default coder is the 2B anonymizer model, so absolute ICD F1 is low (small models are weak coders);
  read it as the drop from original to anonymized text, not as an estimate of the organizers' F1.
