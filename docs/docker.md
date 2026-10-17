# Docker

Two images from one multi-stage `Dockerfile`, and a Compose setup for full mode. No patient data, labels,
outputs or model weights are copied into an image; they are mounted at run time.

| Image | Contents | Size (arm64, measured 18 October 2026) |
|---|---|---|
| `unified-harness:rules` | Python 3.12, the package, standard library only; modes `rules` and `llm` (default) | 147 MB |
| `unified-harness:full` | + transformers, GLiNER and torch 2.14.1 (CPU build); mode `full`, not the default | 1.13 GB |

The lockfile pins torch from PyPI, whose Linux wheels depend on about 18 CUDA packages (several GB). The
`full` image installs every other package from the frozen lock and the same torch version from PyTorch's
CPU-only index. The encoders run on CPU in the container; the LLM runs in the Ollama container.

## Rules mode

```bash
docker build --target rules -t unified-harness:rules .
docker run --rm --network none -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" unified-harness:rules \
    anonymize --input /data/eval_patient_doctor_encounters.jsonl --mode rules
```

## Default (llm mode) with Compose

Prerequisite on the host, once: `ollama pull qwen3.5:2b-q4_K_M`. Compose mounts the host's Ollama store
read-only into the Ollama container (`OLLAMA_MODELS_DIR`, default `~/.ollama/models`), so the containers
never download anything.

```bash
docker compose build harness
docker compose run --rm harness doctor
docker compose run --rm harness anonymize --input /data/eval_patient_doctor_encounters.jsonl
```

Full mode (PyTorch encoders) uses the `full` image: build it with `docker build --target full -t
unified-harness:full .`, download the encoder weights with `scripts/download_models.py`, set the harness
service's image to `unified-harness:full`, mount `./models:/models:ro` and pass `--mode full`.

Both services are on the `offline` network (`internal: true`): they reach each other but not the internet.
The Ollama image is the version used for the measured runs (0.40.6), pinned by digest.

GPU (Linux with the NVIDIA container toolkit) for the LLM:
`docker compose -f compose.yaml -f compose.gpu.yaml run --rm harness ...`. Not tested here (no CUDA
hardware); VRAM on a 4 GB card is unverified.

The output folder and file names use the container's date, which is UTC unless you pass `TZ`
(`docker compose run --rm -e TZ=Europe/Zurich harness ...`).

## Checks done (18 October 2026, Docker 29.1.3, arm64)

| Check | Result |
|---|---|
| Rules output, local vs `rules` container, synthetic fixture | byte-identical |
| `doctor --mode full` in Compose | ready: weights at the pinned revisions, Qwen digest matches |
| Full-mode anonymize in Compose (`privacy`, `F2`), synthetic fixture | published, contract valid |
| HTTP request to the internet from the harness container | fails (no name resolution) |
