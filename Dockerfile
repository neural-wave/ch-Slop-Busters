# syntax=docker/dockerfile:1
#
# Release images for unified-harness.
#   rules  standard-library pipeline, no PyTorch: modes rules and llm (the default; the LLM runs in a separate
#          Ollama container, compose.yaml)                       docker build --target rules -t unified-harness:rules .
#   full   + encoder detectors (CPU torch) and the local LLM client  docker build --target full -t unified-harness:full .
#          weights are mounted at /models; the LLM runs in a separate Ollama container (compose.yaml)
#
# No patient data, labels, outputs or model weights are copied into an image (.dockerignore); they are
# mounted at run time: /data (read-only input), /outputs (results), /models (weights, full only).

# python:3.12-slim, multi-arch index pinned by digest (resolved 2026-10-17)
ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_VERSION=0.12.1

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS builder-rules
ARG UV_VERSION
RUN pip install "uv==${UV_VERSION}"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --python /usr/local/bin/python3

FROM base AS rules
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data /outputs /models \
    && chown app:app /outputs
COPY --from=builder-rules /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    UNIFIED_HARNESS_HOME=/ \
    UNIFIED_HARNESS_MODELS=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /outputs
USER app
LABEL org.opencontainers.image.title="unified-harness" \
      org.opencontainers.image.description="Offline, ICD-aware anonymization of patient-doctor encounters (rules and llm modes)"
ENTRYPOINT ["unified-harness"]
CMD ["--help"]

# --- full: rules + deid_roberta + GLiNER-PII (CPU torch) + the Ollama client ------------------------------
# The lockfile pins torch 2.14.1 from PyPI, whose Linux wheels pull several GB of CUDA libraries. The image
# installs everything else from the frozen lock and the same torch version from PyTorch's CPU-only index.
FROM builder-rules AS builder-full
ARG TORCH_VERSION=2.14.1
RUN uv sync --frozen --no-dev --no-editable --extra ml --python /usr/local/bin/python3 \
        --no-install-package torch --no-install-package triton \
        --no-install-package cuda-bindings --no-install-package cuda-pathfinder --no-install-package cuda-toolkit \
        --no-install-package nvidia-cublas --no-install-package nvidia-cuda-cupti \
        --no-install-package nvidia-cuda-nvrtc --no-install-package nvidia-cuda-runtime \
        --no-install-package nvidia-cudnn-cu13 --no-install-package nvidia-cufft --no-install-package nvidia-cufile \
        --no-install-package nvidia-curand --no-install-package nvidia-cusolver --no-install-package nvidia-cusparse \
        --no-install-package nvidia-cusparselt-cu13 --no-install-package nvidia-nccl-cu13 \
        --no-install-package nvidia-nvjitlink --no-install-package nvidia-nvshmem-cu13 --no-install-package nvidia-nvtx \
    && uv pip install --python /app/.venv/bin/python --no-deps \
        --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

FROM rules AS full
USER root
COPY --from=builder-full /app/.venv /app/.venv
USER app
LABEL org.opencontainers.image.description="Offline, ICD-aware anonymization of patient-doctor encounters (full mode, CPU)"
