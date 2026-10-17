# unified-harness

Offline, ICD-aware anonymization of patient-doctor encounters, with an evaluation harness built to
measure privacy and clinical utility honestly. It merges two earlier team solutions: the shield-first
rule pipeline by Alind Xhyra and Davide Molinelli and the LLM extraction harness by Ketai.

> **Status: milestone M1 (baseline ported).** The pipeline is the shield-first baseline; the privacy
> features of the plan come next. See [PLAN.md](PLAN.md) for the design and milestones and
> [STATUS.md](STATUS.md) for progress and measured results.

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

## Usage

Put the challenge files in `data/` (git-ignored), then:

```bash
uv run unified-harness doctor --mode full                     # extras, pinned weights, Ollama model
uv run unified-harness anonymize --input data/eval_patient_doctor_encounters.jsonl            # full mode
uv run unified-harness anonymize --input data/eval_patient_doctor_encounters.jsonl --mode rules
uv run unified-harness validate --original data/eval_patient_doctor_encounters.jsonl \
    --anonymized outputs/<date>/anonymized-<date>-unified-harness.jsonl
```

Modes: `rules` uses no ML and no GPU; `full` adds the deid_roberta and GLiNER-PII encoders and the local
Qwen 3.5 2B (Q4) model through Ollama. If anything `full` needs is missing, the run stops with an error;
it never falls back to `rules`. Presets: `conservative`, `balanced` (default), `aggressive`.
`--no-codes` runs without reading `icd10_codes`.

### Evaluation harness

```bash
uv run --extra eval unified-harness run --split dev --mode rules --preset balanced    # one results row
uv run --extra eval unified-harness sweep --split dev --mode full                      # presets x codes
uv run unified-harness show --id D2N071 --mode rules                                  # debug view (prints text)
```

Rows are appended to `outputs/eval/results.md`. Silver labels live in `labels/` (git-ignored); see
[docs/labeling_guidelines.md](docs/labeling_guidelines.md). The `dev` split ids are in
[configs/splits.json](configs/splits.json).

## Output

`outputs/<YYYY-MM-DD>/anonymized-<YYYY-MM-DD>-unified-harness.jsonl`, plus a `.decisions.jsonl` sidecar
(one explanation per decision, without original text) and a `.summary.json` (runtime, memory, checks).
Files are published atomically and only after the output contract passes.

## Docker

```bash
docker build --target rules -t unified-harness:rules .
docker run --rm --network none -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" unified-harness:rules doctor --mode rules
```

The `full` image and a Compose setup with a local Ollama service follow in milestone M6.

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
