# unified-harness

Offline, ICD-aware anonymization of patient-doctor encounters, with an evaluation harness built to
measure privacy and clinical utility honestly. It merges two earlier team solutions: the shield-first
rule pipeline by Alind Xhyra and Davide Molinelli and the LLM extraction harness by Ketai.

> **Version 1.0.0.** The design, the pipeline and all measured results are described in
> [docs/solution.html](docs/solution.html); labels, splits and metrics in [docs/evaluation.md](docs/evaluation.md).

## What it does

Each encounter has a `dialogue` and a `note`. The tool rewrites only those two fields: it removes or
generalizes what identifies the patient and keeps what supports the encounter's ICD-10 codes. Every
other field, the record order and `icd10_codes` stay byte-identical. Inference runs on-premises, on CPU
or within 4 GB of VRAM.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # rules mode: standard library only
uv sync --extra ml                        # full mode: encoder detectors
uv run --extra download python scripts/download_models.py    # pinned weights into ./models
ollama pull qwen3.5:2b-q4_K_M             # local LLM used by full mode
```

## Quickstart (local, no PyTorch)

```bash
ollama pull qwen3.5:2b-q4_K_M          # once; the local LLM detector (1.9 GB, about 1.6 GB VRAM)
uv sync                                # once
uv run unified-harness doctor          # must say "mode llm: ready"
uv run unified-harness anonymize --input data/eval_patient_doctor_encounters.jsonl
uv run unified-harness validate --original data/eval_patient_doctor_encounters.jsonl \
    --anonymized outputs/<date>/anonymized-<date>-unified-harness.jsonl
```

Defaults: mode `llm` (rules + `qwen3.5:2b-q4_K_M` via Ollama), preset `privacy`, features F0 and F2. On
new text expect about 10–14 s per encounter (LLM calls); a re-run of the same text reads the replay
cache in `outputs/cache/`. An existing output of the same date is never overwritten (`--overwrite`).


Put the challenge files in `data/` (git-ignored), then:

```bash
uv run unified-harness doctor --mode full                     # full mode: extras, pinned weights, Ollama model
uv run unified-harness anonymize --input data/eval_patient_doctor_encounters.jsonl --mode full
uv run unified-harness anonymize --input data/eval_patient_doctor_encounters.jsonl --mode rules
uv run unified-harness validate --original data/eval_patient_doctor_encounters.jsonl \
    --anonymized outputs/<date>/anonymized-<date>-unified-harness.jsonl
```

Modes: `rules` uses no ML and no GPU; `full` adds the deid_roberta and GLiNER-PII encoders and the local
Qwen 3.5 2B (Q4) model through Ollama. If anything `full` needs is missing, the run stops with an error;
it never falls back to `rules`. Presets: `conservative`, `balanced` (default), `aggressive`, `privacy`
(balanced settings plus low-risk details, with the code-evidence guard). `--no-codes` runs without
reading `icd10_codes`.

Feature flags (`--features`, comma list; each was gated on the tune split):

| Flag | What it does | Default in |
|---|---|---|
| `F0` | An activity is kept as injury evidence only when the sentence tells the mechanism; then every mention of it is kept | `privacy` |
| `F2` | The local LLM reads only privacy-dense segments and also finds occupations; replies are cached in `outputs/cache/` | full mode, opt-in |
| `F4` | Low-risk details are removed only when two independent signals agree (measured: lower privacy, higher precision) | opt-in |

### Evaluation harness

```bash
uv run --extra eval unified-harness run --split tune --mode rules --preset privacy    # one results row
uv run --extra eval unified-harness run --split tune --mode full --preset privacy --features F2
uv run --extra eval unified-harness sweep --split tune --mode full                     # presets x codes
uv run unified-harness gate outputs/eval/<new>.json outputs/eval/<current best>.json  # acceptance gate
uv run unified-harness show --id D2N071 --mode rules                                  # debug view (prints text)
```

Splits: `tune` (140 train encounters, for development), `validation` (the 67 eval encounters, for
checks only), `test` (24 held-out documents, scored once). Use `--icd none` to skip the ICD proxy when
Ollama with `bge-m3` is not running. Rows are appended to `outputs/eval/results.md`. Silver labels live in `labels/` (git-ignored); see
[docs/labeling_guidelines.md](docs/labeling_guidelines.md). The `dev` split ids are in
[configs/splits.json](configs/splits.json).

## Results

Silver labels, shares of labeled occurrences (details and how they were measured: [docs/solution.html](docs/solution.html)):

| Split | All PII removed | High risk | Keep traps | Concepts | VRAM |
|---|---|---|---|---|---|
| tune (140) | 92.1% | 99.2% | 100.0% | 99.7% | 1.6 GB |
| validation (67) | 84.6% | 95.1% | 99.1% | 99.8% | 1.6 GB |
| test, unseen ICU notes (24) | 85.7% | 89.7% | 95.3% | 95.6% | 1.6 GB |

Limitations: silver labels only; the unseen ICU-note format scores clearly lower (above); some date forms are not targeted yet; a relative or clinician can get
the [PATIENT] tag (the name is removed); VRAM measured on Apple unified memory, not on a CUDA card.

## Output

`outputs/<YYYY-MM-DD>/anonymized-<YYYY-MM-DD>-unified-harness.jsonl`, plus a `.decisions.jsonl` sidecar
(one explanation per decision, without original text) and a `.summary.json` (runtime, memory, checks).
Files are published atomically and only after the output contract passes.

## Docker

```bash
# rules mode: one container, no network
docker build --target rules -t unified-harness:rules .
docker run --rm --network none -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" unified-harness:rules \
    anonymize --input /data/eval_patient_doctor_encounters.jsonl --mode rules

# default llm mode: the harness and Ollama on an internal network without internet
docker compose run --rm harness doctor
docker compose run --rm harness anonymize --input /data/eval_patient_doctor_encounters.jsonl
```

Details, sizes and the GPU override: [docs/docker.md](docs/docker.md).

## Data and privacy

The challenge data, silver labels, outputs and model weights are never committed; `data/`, `labels/`,
`outputs/` and `models/` are git-ignored and excluded from Docker builds. Tests use synthetic fixtures.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

## Credits

Built on the work of Alind Xhyra and Davide Molinelli (shield-first pipeline, gold-label guidelines,
attacks and robustness sets) and Ketai (zero-shot LLM extraction, age and date generalization, entity
replay). License: to be decided before publication.
