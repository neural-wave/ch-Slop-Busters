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

```bash
mkdir -p outputs                   # a folder Docker creates for the mount would belong to root
docker compose build harness
docker compose up ollama-pull      # downloads the detector model into the `ollama-models` volume
docker compose run --rm harness doctor
docker compose run --rm harness anonymize --input /data/eval_patient_doctor_encounters.jsonl
```

Three services:

| Service | Network | Role |
|---|---|---|
| `ollama-pull` | `fetch` (internet) | One-shot: starts Ollama, pulls `DETECTOR_MODEL` (default `qwen3.5:2b-q4_K_M`) into the volume, exits. Handles model weights only |
| `ollama` | `offline` (internal) | Serves the model to the harness; starts after `ollama-pull` succeeds |
| `harness` | `offline` (internal) | Reads `/data`, writes `/outputs`; cannot reach the internet |

`docker compose run` starts the two Ollama services automatically, but it does not show the download
progress of `ollama-pull`; `docker compose up ollama-pull` does, so run it once first. The volume keeps the
models between runs; `docker compose down -v` removes them. Another model: prefix every command with
`DETECTOR_MODEL=<model>` and pass `--allow-unpinned` to the harness when the model is not pinned in
`models.toml`.

Full mode (PyTorch encoders) uses the `full` image: build it with `docker build --target full -t
unified-harness:full .`, download the encoder weights with `scripts/download_models.py`, set the harness
service's image to `unified-harness:full`, mount `./models:/models:ro` and pass `--mode full`.

On Linux a container writes a host folder with its own uid. The `harness` service therefore runs as
`${HARNESS_UID:-1000}:${HARNESS_GID:-1000}`: the usual first user, who owns `./outputs` after `mkdir -p outputs`,
so the container can write there and the results stay that user's. Another user prefixes the commands with
`HARNESS_UID=$(id -u) HARNESS_GID=$(id -g)`. Output files are created 0644 (by the umask), so they are readable
on the host whichever uid wrote them. Docker Desktop (macOS, Windows) maps ownership itself.

The output folder and file names use the container's date, which is UTC unless you pass `TZ`
(`docker compose run --rm -e TZ=Europe/Zurich harness ...`).

## Checks done (18 October 2026, Docker 29.1.3, arm64)

| Check | Result |
|---|---|
| Rules output, local vs `rules` container, synthetic fixture | byte-identical |
| `doctor --mode full` in Compose | ready: weights at the pinned revisions, Qwen digest matches |
| Full-mode anonymize in Compose (`privacy`, `F2`), synthetic fixture | published, contract valid |
| HTTP request to the internet from the harness container | fails (no name resolution) |
| Fresh volume: `ollama-pull` downloads the default model, `doctor` ready (digest matches the pin) | done |

Linux (18 October 2026, Docker 29.7.2, Compose v5.6.0, x86_64, no NVIDIA container runtime):

| Check | Result |
|---|---|
| `rules` image | 122 MB |
| Rules output on the 67 eval encounters, local vs `rules` container (`--network none`) | byte-identical (records and decisions); `validate`: 0 contract violations |
| Fresh volume: `ollama-pull`, then `doctor` | ready, digest matches the pin |
| Compose before the fix: `./outputs` missing (Docker creates it as root) or owned by the user | anonymize fails with `PermissionError` on `/outputs`; files the image's uid 10001 wrote were 0600, unreadable on the host |
| Compose after the fix (`mkdir -p outputs`, harness as uid 1000, files 0644), `anonymize --limit 2` in llm mode | published, owned by the user, 0 contract violations; about 70 s per encounter with Ollama on the CPU |
| `compose.gpu.yaml` | not tested (needs the NVIDIA container toolkit) |
