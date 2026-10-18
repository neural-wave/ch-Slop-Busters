#!/usr/bin/env bash
# Resumable: SFT data -> LoRA fine-tune -> merge -> GGUF (bf16) -> Q4_K_M. Finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LLAMA=${LLAMA_CPP:-$HOME/llama.cpp}
OUT=models/qwen35-2b-deid

[ -s outputs/sft/train.jsonl ] || $PY train/build_sft.py
$PY train/finetune.py all
[ -s $OUT-bf16.gguf ] || $PY $LLAMA/convert_hf_to_gguf.py models/qwen35-2b-deid-merged --outtype bf16 --outfile $OUT-bf16.gguf
[ -s $OUT-Q4_K_M.gguf ] || $LLAMA/build/bin/llama-quantize $OUT-bf16.gguf $OUT-Q4_K_M.gguf Q4_K_M
ls -la $OUT-*.gguf
echo "PIPELINE DONE"
