# ch-Slop-Busters: anonymization of patient data

Neural Wave hackathon, challenge “Anonymization of patient data”.

This repository holds **unified-harness**, an offline, diagnosis-aware anonymizer for patient–doctor
encounters, together with its evaluation harness. The anonymizer
removes or generalizes what identifies a person in the `dialogue` and `note` of each encounter and keeps
what a clinical coder needs for the ICD-10 codes. It runs on a laptop: rules plus one small local
language model through Ollama, about 1.6 GB of GPU memory, no cloud service.

> The full description of the design, the pipeline and every measured result is
> [`docs/solution.html`](docs/solution.html) (open it in a browser). Labels, splits and metrics are in
> [`docs/evaluation.md`](docs/evaluation.md).

## Contents

- [Results at a glance](#results-at-a-glance)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Running the anonymizer](#running-the-anonymizer)
- [Output files](#output-files)
- [Evaluating a run](#evaluating-a-run)
- [Docker](#docker)
- [How it works](#how-it-works)
- [Data and privacy rules](#data-and-privacy-rules)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Credits and license](#credits-and-license)

## Results at a glance

Default configuration (mode `llm`, preset `privacy`). Silver labels: shares of labeled occurrences that
were removed (privacy) or kept (keep traps, medical concepts).

| Split | All PII removed | High-risk removed | Keep traps kept | Medical concepts kept | Peak VRAM |
|---|---|---|---|---|---|
| tune (140 train encounters, used for design decisions) | 92.1% | 99.2% | 100.0% | 99.7% | 1.6 GB |
| validation (the 67 eval encounters) | 84.6% | 95.1% | 99.1% | 99.8% | 1.6 GB |
| test (24 held-out ICU notes, unseen format, scored once) | 85.7% | 89.7% | 95.3% | 95.6% | 1.6 GB |

On the validation split, with the same labels and scorer, the two earlier team harnesses reach 77.1%
(Ketai) and 81.5% (Alind and Davide) of PII removed, and keep 79.6% and 99.4% of keep traps. The
`icd10_codes` and every field other than `dialogue` and `note` are byte-identical in every output.

## Repository layout

| Path | Contents |
|---|---|
| `data/patient_doctor_encounters.jsonl` | 140 training encounters (tracked) |
| `data/eval_patient_doctor_encounters.jsonl` | 67 evaluation encounters (tracked) |
| `src/unified_harness/` | The anonymizer and the evaluation code (Python package, CLI `unified-harness`) |
| `src/unified_harness/models.toml` | Pinned models: the Ollama detector models (by digest) and, for `full` mode, the encoder weights |
| `src/unified_harness/data/` | Public lexicons (ICD-10-CM, drugs, names, places, word frequencies), compressed |
| `tests/` | Unit and integration tests on synthetic fixtures; data tests that run when `data/` and `labels/` exist |
| `docs/` | `solution.html` (the solution page), `evaluation.md`, `error_analysis.md`, `labeling_guidelines.md`, `docker.md` |
| `configs/` | `splits.json` (split ids) and the release-guard allowlist |
| `scripts/` | Model download, solution-page builder, release guard, scorer cross-check, error analysis, label tools (`labels/`) |
| `Dockerfile`, `compose.yaml`, `compose.gpu.yaml` | Container images and the offline Compose setup |
| Local only (git-ignored) | `labels/` (silver labels), `outputs/` (results, LLM cache), `models/` (encoder weights), `data/robustness/` (held-out sets) |

## Requirements

- **Python 3.12+** and [**uv**](https://docs.astral.sh/uv/).
- [**Ollama**](https://ollama.com) running locally with the detector model:
  `ollama pull qwen3.5:2b-q4_K_M` (1.9 GB download, about 1.6 GB of VRAM in use).
- Optional: `bge-m3` in Ollama (`ollama pull bge-m3`) for the ICD utility proxy in evaluations.
- Optional: **Docker Desktop** for the container run.
- No GPU is required; Ollama uses the GPU when one is available, within the 4 GB budget.

## Quickstart

```bash
ollama pull qwen3.5:2b-q4_K_M
uv sync
uv run unified-harness doctor
uv run unified-harness anonymize --input data/eval_patient_doctor_encounters.jsonl
uv run unified-harness validate --original data/eval_patient_doctor_encounters.jsonl \
    --anonymized outputs/<date>/anonymized-<date>-unified-harness.jsonl
```

`doctor` must end with `mode llm: ready`. The anonymized file appears in `outputs/<date>/`; `validate`
must report `0 contract violation(s)`.

## Running the anonymizer

```bash
uv run unified-harness anonymize --input <file.jsonl> [options]
```

The input is JSONL with one encounter per line: `encounter_id`, `dataset`, `dialogue`, `note`,
`icd10_codes` (a list of `{code, description}`). Any file in this schema works.

| Option | Default | Meaning |
|---|---|---|
| `--mode` | `llm` | `llm`: rules + the local Ollama model. `rules`: no model at all (fastest, CPU only). `full`: adds two PyTorch encoders (needs `uv sync --extra ml` and `scripts/download_models.py`) |
| `--preset` | `privacy` | How much to remove: `conservative`, `balanced`, `aggressive`, `privacy` (see below) |
| `--features` | from the preset | Extra feature flags, comma list: `F0`, `F2`, `F4` (see below) |
| `--detector-model` | `qwen3.5:2b-q4_K_M` | Another Ollama model for the detector; it must be pinned by digest in `models.toml` |
| `--no-codes` | off | Do not read `icd10_codes` at inference (bonus task) |
| `--out-root` | `outputs` | Results go to `<out-root>/<date>/` |
| `--date` | today | Date used in the output names (`YYYY-MM-DD`) |
| `--overwrite` | off | Replace an existing output of the same date (otherwise the run refuses) |
| `--ids`, `--limit` | all | Process only some encounters (a JSON list of ids, or the first N) |

**Presets**

| Preset | Removes | Typical use |
|---|---|---|
| `conservative` | Direct identifiers (names, dates of birth, IDs, contact data, dates; exact ages become bands) and years | Maximum utility |
| `balanced` | + places, organizations, occupations, facilities, teams, holidays, hobbies | Earlier default |
| `aggressive` | + low-risk personal details, coarser ages, no injury-activity protection | Maximum removal; loses some clinical keep traps |
| `privacy` | Everything aggressive removes, with balanced name and age settings and the F0 guard | **Default** |

**Feature flags**

| Flag | Effect | In the default |
|---|---|---|
| `F0` | An activity is kept as clinical evidence only when its sentence tells an injury mechanism (“hurt my knee skiing”), and then in every mention | yes |
| `F2` | The LLM reads only privacy-dense segments (work, home, family, travel, rare names) and also finds occupations; replies are cached | yes |
| `F4` | Low-risk details are removed only when two independent signals agree (higher precision, lower privacy) | no |

**Using another Ollama model.** Any model can replace the default detector if it fits the 4 GB budget
(the run measures its VRAM at load and stops above 4 GB). Small candidates in the Ollama library:
`llama3.2:3b` (2.0 GB), `granite4:3b` (2.1 GB), `phi4-mini` (2.5 GB), `ministral-3:3b` (3.0 GB),
`qwen3.5:4b` (3.3 GB), `gemma3:4b` (3.4 GB). Larger models such as `gemma4:latest` exceed the budget.

```bash
ollama pull llama3.2:3b
curl -s http://127.0.0.1:11434/api/tags      # copy the model's "digest"
# add a line under [detector_models] in src/unified_harness/models.toml:  "llama3.2:3b" = "<digest>"
uv run unified-harness doctor --detector-model llama3.2:3b
uv run unified-harness anonymize --input data/eval_patient_doctor_encounters.jsonl \
    --detector-model llama3.2:3b --out-root outputs/llama3.2 --progress --llm-log outputs/llama3.2/calls.jsonl
```

Each model keeps its own entries in the reply cache (the key includes the model digest). Compare models by
evaluating each output with `evaluate` and comparing the rows, or with `gate`.

**Monitoring a run.** `--progress` prints one line per encounter (time, decisions, LLM calls and cache
hits). `--llm-log <file>` appends one JSON line per LLM call: time, encounter id, field, segment offsets,
length, seconds, whether it came from the cache, and how many names, places, organizations and jobs the
model returned. It contains no text from the records. Follow it live with `tail -f <file>`.

**Environment variables**

| Variable | Default | Use |
|---|---|---|
| `UNIFIED_HARNESS_OLLAMA_URL` | `http://127.0.0.1:11434` | Where the Ollama server is |
| `UNIFIED_HARNESS_HOME` | current folder | Root for `outputs/`, `labels/`, `configs/` |
| `UNIFIED_HARNESS_MODELS` | `models` | Encoder weights for `full` mode |

**Safety behavior.** The run checks its requirements first and stops with an error if anything is
missing; it never falls back to a weaker mode. If any record fails, nothing is published. Output files
are written atomically, only after every record has passed the output contract.

## Output files

For a run on date `D`, in `<out-root>/D/`:

| File | Contents |
|---|---|
| `anonymized-D-unified-harness.jsonl` | The input records in the same order; only `dialogue` and `note` changed |
| `anonymized-D-unified-harness.decisions.jsonl` | One line per decision: position, category, action (replace, generalize, keep), detector and reason. No original text |
| `anonymized-D-unified-harness.summary.json` | Settings, model digests, record count, runtime, peak RAM and VRAM, LLM calls and cache hits, checks |

Replacements are tags such as `[PATIENT]`, `[CLINICIAN]`, `[DATE]`, `[LOCATION]`, `[OCCUPATION]` or
`[DETAIL]`; exact ages become decade bands such as `[50s]`.

## Evaluating a run

Scoring needs the silver labels in `labels/` (local only, not in git). Ask a teammate for the folder.

```bash
uv run --extra eval unified-harness evaluate \
    --original data/eval_patient_doctor_encounters.jsonl \
    --anonymized outputs/<date>/anonymized-<date>-unified-harness.jsonl \
    --decisions outputs/<date>/anonymized-<date>-unified-harness.decisions.jsonl \
    --summary outputs/<date>/anonymized-<date>-unified-harness.summary.json \
    --gold labels/merged/validation --name my_run
```

It prints one row and saves `outputs/eval/my_run.json`. Add `--icd none` when `bge-m3` is not in Ollama.

| Column | Meaning |
|---|---|
| privacy, high, high+med, low | Share of labeled PII occurrences removed: all, and by risk tier |
| direct, quasi | Removal of direct identifiers and of quasi-identifiers |
| name tok | Removal of each name token (a surname left behind counts as a leak) |
| keep | Share of clinical keep traps still present (eponyms, code evidence, look-alike numbers) |
| prec | Share of edits that hit a labeled PII string |
| concepts | Medical concept recall |
| ICD F1 / agree | Local ICD-10 retrieval proxy on original vs anonymized text, and their agreement |
| ms/enc, RAM, VRAM | Speed and memory of the anonymization run |

Other commands:

```bash
uv run --extra eval unified-harness run --split tune --mode llm --preset privacy   # anonymize a split and score it
uv run unified-harness gate outputs/eval/<new>.json outputs/eval/<old>.json   # acceptance gate
uv run --extra eval unified-harness attack --original <in.jsonl> --anonymized <out.jsonl> \
    --gold labels/merged/tune --skip-ner                        # re-identification attacks
uv run unified-harness show --id <encounter_id> --mode rules    # one encounter with every decision (prints text)
```

Splits for `--split`: `tune` (140 train encounters), `validation` (the 67 eval encounters), `test`
(24 held-out documents in `data/robustness/`). Any other set can be scored with `--gold <folder>`, one
`<encounter_id>.json` per encounter in the label format of [`docs/labeling_guidelines.md`](docs/labeling_guidelines.md).

## Docker

The harness image is small (147 MB, no PyTorch); Ollama runs in its own container. Both sit on an
internal network without internet access, and the host's Ollama models are mounted read-only.

```bash
docker compose build harness
docker compose run --rm harness doctor
docker compose run --rm -e TZ=Europe/Zurich harness anonymize \
    --input /data/eval_patient_doctor_encounters.jsonl --out-root /outputs/docker
docker compose run --rm harness validate --original /data/eval_patient_doctor_encounters.jsonl \
    --anonymized /outputs/docker/<date>/anonymized-<date>-unified-harness.jsonl
docker compose down
```

Inside the container `/data` is `./data` (read-only) and `/outputs` is `./outputs`. Pass `TZ`, or the
date in the output names is UTC. More, including the GPU override and the `full` image:
[`docs/docker.md`](docs/docker.md).

## How it works

1. **Normalize** the text with a one-to-one character view, so every detector reports positions in the
   original.
2. **Shield** what a coder needs: clinical numbers with units, drug names, eponyms, words that support
   the encounter's own ICD-10 codes, and activities told as an injury mechanism. Nothing may edit them.
3. **Detect** identifiers with rules and gazetteers, and with the local model on privacy-dense segments.
4. **Link** the mentions of each person across turns and fields, so a surname cannot survive elsewhere.
5. **Apply the policy**: replace, generalize or keep each category according to the preset.
6. **Verify**: search for any removed string still in the text and repair it; check the output contract.
7. **Publish** atomically, with the decisions file and the summary.

The design merges the two earlier team harnesses: the shield-first rules, identity graph and leak repair
of Alind and Davide, and the LLM extraction, age generalization and replay cache of Ketai.

## Data and privacy rules

- Never commit `labels/`, `outputs/`, `models/` or `data/robustness/`; all are git-ignored. The two
  challenge files in `data/` are tracked on purpose.
- Docker images contain no data, labels, outputs or weights; they are mounted at run time.
- The decisions file never contains original text (`--explain-text` is a debug option that does).
- Before publishing, run the release guard, which searches every tracked file for labeled identifiers:
  `uv run python scripts/check_no_patient_text.py`.
- Inference stays on-premises: no cloud models, at most 4 GB of VRAM.

## Development

```bash
uv sync --all-extras          # every optional dependency (ml, eval, download)
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
uv run python scripts/build_solution_page.py   # rebuild docs/solution.html from outputs/eval/
```


## Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor`: `Ollama is not reachable` | Start Ollama (`ollama serve` or the app), or set `UNIFIED_HARNESS_OLLAMA_URL` |
| `doctor`: `not pulled` | `ollama pull qwen3.5:2b-q4_K_M` |
| `doctor`: `not pinned in models.toml` | Add the model and its digest (`ollama list`) under `[detector_models]` |
| `... already exists` | Use another `--out-root` or `--date`, or `--overwrite` |
| ICD columns empty in `evaluate` | `ollama pull bge-m3`, or pass `--icd none` |
| Privacy columns missing in `evaluate` | The `labels/` folder is missing |
| A run on new text is slow | Expected: about 10–14 s per encounter for the LLM; reruns of the same text read `outputs/cache/` |

## Limitations

- All labels are silver (made by LLM agents and checked by script, not reviewed by a human).
- The unseen ICU-note format scores clearly lower than the training format (see the test row above).
- Some date forms are not targeted yet; they are the main high-risk gap on validation.
- A relative or clinician sometimes gets the `[PATIENT]` tag; the name itself is removed.
- VRAM was measured on Apple unified memory; a 4 GB CUDA card was not tested.

## Credits and license

Built on the work of Alind Xhyra and Davide Molinelli (shield-first pipeline, gold-label guidelines,
attacks and robustness sets) and Ketai (zero-shot LLM extraction, age and date generalization, entity
replay).

Licensed under the Apache License 2.0; see [`LICENSE`](LICENSE).
