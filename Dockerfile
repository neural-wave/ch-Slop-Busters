# syntax=docker/dockerfile:1
#
# Release images for unified-harness.
#   rules  standard-library pipeline, no ML dependencies, no GPU     docker build --target rules -t unified-harness:rules .
#   full   + encoder detectors and the local LLM client (milestone M6, see PLAN.md)
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
      org.opencontainers.image.description="Offline, ICD-aware anonymization of patient-doctor encounters (rules mode)"
ENTRYPOINT ["unified-harness"]
CMD ["--help"]
