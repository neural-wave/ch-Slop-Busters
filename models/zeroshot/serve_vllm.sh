#!/usr/bin/env bash
# Serve a local HF checkpoint with vLLM on an OpenAI-compatible endpoint (default: models/Qwen3.5-2B on :8000).
# Usage: bash harness/serve_vllm.sh [model-dir] [served-name] [port]
#   VRAM_GB=4 bash harness/serve_vllm.sh   evaluation setup within the task's 4 GB VRAM limit: FP8 weights
#                                          (2.4 GiB instead of 4.3 GB BF16), no CUDA graphs, KV cache sized to fit.
#   Without VRAM_GB this is a development setup (BF16, half the GPU), which the task allows only for development.
# Setup (once): uv venv .venv-vllm --python 3.11 && uv pip install
#   "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
#   --torch-backend=cu129, then uv pip uninstall torchcodec (its PyPI build needs CUDA 13; vLLM treats it as optional)
#   hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${1:-models/Qwen3.5-2B}"
NAME="${2:-qwen3.5-2b}"
PORT="${3:-8000}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load amd/nvidia/cuda-12.9.2
source .venv-vllm/bin/activate
export HF_HUB_OFFLINE=1                # local checkpoint, no hub lookups
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer JIT sampling kernels fail on the node's 545 driver

if [[ -n "${VRAM_GB:-}" ]]; then
  # --gpu-memory-utilization is a fraction of the whole GPU for weights, profiled activations and KV cache.
  # The CUDA context and kernel workspaces come on top (measured ~550 MiB on an L40S with vLLM 0.31), so
  # 600 MiB of the budget is held back; check the real peak with nvidia-smi while a run is going.
  TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  UTIL=$(python3 -c "print(f'{(${VRAM_GB} * 1024 - 600) / ${TOTAL_MIB}:.4f}')")  # see the note above
  # small prefill batches keep the activation peak, which the utilization fraction does not cover, inside the budget
  MEM_ARGS=(--gpu-memory-utilization "$UTIL" --quantization fp8 --enforce-eager
            --max-num-batched-tokens 2048 --max-num-seqs 8)
  echo "[serve_vllm] VRAM budget ${VRAM_GB} GB of ${TOTAL_MIB} MiB -> --gpu-memory-utilization ${UTIL}" >&2
else
  MEM_ARGS=(--gpu-memory-utilization 0.5)
fi

# Text only: no image/video inputs, so the vision encoder gets no multimodal budget.
# 16k context matches evaluate.py --coder-ctx (whole encounters); the reasoning parser keeps
# <think> out of `content` when a caller enables thinking (evaluate.py --coder-think).
exec vllm serve "$MODEL_DIR" --served-model-name "$NAME" --port "$PORT" \
  --max-model-len 16384 "${MEM_ARGS[@]}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' --reasoning-parser qwen3
