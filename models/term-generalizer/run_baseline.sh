#!/usr/bin/env bash
# Zero-shot baseline: anonymize the eval split with a small local LLM, then evaluate every run.
# Usage: bash ../term-generalizer/run_baseline.sh [anonymizer-model] [coder-model]
#   vLLM: LLM_BACKEND=vllm WORKERS=32 CODER_WORKERS=16 bash ../term-generalizer/run_baseline.sh qwen3.5-2b qwen3.5-2b
#         (models are the server's --served-model-name; run dirs get a vllm- prefix)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-qwen3.5:2b}"
CODER="${2:-qwen3.5:2b}"
WORKERS="${WORKERS:-4}"
CODER_WORKERS="${CODER_WORKERS:-2}"
TAG="${MODEL//[:\/]/-}"
IDENTITY=runs/identity   # its ICD-10 scores depend on the coder, so each backend keeps its own copy
if [[ "${LLM_BACKEND:-ollama}" == "vllm" ]]; then TAG="vllm-${TAG}"; IDENTITY=runs/vllm-identity; fi

python3 ../term-generalizer/anonymize.py --mode identity --model none   --out "$IDENTITY"
python3 ../term-generalizer/anonymize.py --mode extract  --model "$MODEL" --out "runs/${TAG}_extract" --workers "$WORKERS"
python3 ../term-generalizer/anonymize.py --mode rewrite  --model "$MODEL" --out "runs/${TAG}_rewrite" --workers "$WORKERS"
# generalize = extract's LLM spans (replayed, no model calls) + regex ages/dates rendered as ranges
# + diagnosis words of the record's own icd10_codes kept; _noicd is the bonus setting without codes.
# Replays are cheap, so --fresh: a stale partial output never hides a code change.
python3 ../term-generalizer/anonymize.py --mode generalize --model "$MODEL" --out "runs/${TAG}_generalize" \
  --reuse-entities "runs/${TAG}_extract/anonymization_log.jsonl" --fresh
python3 ../term-generalizer/anonymize.py --mode generalize --model "$MODEL" --out "runs/${TAG}_generalize_noicd" \
  --reuse-entities "runs/${TAG}_extract/anonymization_log.jsonl" --no-icd-hints --fresh

RUNS=("$IDENTITY" "runs/${TAG}_extract" "runs/${TAG}_rewrite" "runs/${TAG}_generalize" "runs/${TAG}_generalize_noicd")
for run in "${RUNS[@]}"; do
  python3 ../term-generalizer/evaluate.py --anon "$run/anonymized.jsonl" --coder-model "$CODER" --coder-workers "$CODER_WORKERS"
done

python3 ../term-generalizer/summarize.py runs/*/   # every run, so other backends keep their rows in runs/summary.md
