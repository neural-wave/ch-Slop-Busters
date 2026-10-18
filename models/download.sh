#!/usr/bin/env bash
# Fetch what git does not carry for models/: the Qwen GGUF, the Hugging Face detector models and the two venvs.
# Run from anywhere: paths resolve against this script's folder. Every download is pinned to the revision we evaluated,
# and the GGUF is checked against its sha256. Finished steps are skipped, so the script can be rerun.
#
#   models/download.sh             # everything
#   models/download.sh --no-venv   # weights only
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=1
[ "${1:-}" = "--no-venv" ] && VENV=0

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
# huggingface_hub without installing anything globally (same major version as the venvs)
hf() { uv tool run --from 'huggingface_hub>=1.33,<2' hf "$@"; }

# 1. models/anonymous: Qwen 3.5 2B Q4_K_M (Unsloth) for the local llama-server
GGUF_REPO=unsloth/Qwen3.5-2B-GGUF
GGUF_REV=f6d5376be1edb4d416d56da11e5397a961aca8ae
BASE=$HERE/anonymous/models/base
for f in Qwen3.5-2B-Q4_K_M.gguf:aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223 \
         imatrix_unsloth.gguf_file:0661bad5020138c6da2d45d2223ba32f0e4e8329a2fbf8e2ae382e782348fbca; do
  name=${f%%:*} sum=${f#*:}
  if ! echo "$sum  $BASE/$name" | sha256sum -c --status 2>/dev/null; then
    echo "==> $GGUF_REPO/$name"
    hf download "$GGUF_REPO" "$name" --revision "$GGUF_REV" --local-dir "$BASE"
    echo "$sum  $BASE/$name" | sha256sum -c
  fi
done

# 2. Detector models (deid_roberta, GLiNER) into the Hugging Face cache: both pipelines load them with
#    local_files_only=True. ad-night uses knowledgator's GLiNER, anonymous uses urchade's (on an mDeBERTa backbone).
for m in obi/deid_roberta_i2b2:78f2152eb93ddd817290ce8dbe46f1a6685e09fc \
         knowledgator/gliner-pii-base-v1.0:61726e0ad791dcab3e29339bbec3ad42ded65641 \
         urchade/gliner_multi_pii-v1:1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d \
         microsoft/mdeberta-v3-base:a0484667b22365f84929a935b5e50a51f71f159d; do
  echo "==> ${m%%:*}"
  hf download "${m%%:*}" --revision "${m#*:}" >/dev/null
done

# 3. Optional Ollama model for ad-night's LLM detector and the harness judge
if command -v ollama >/dev/null; then
  ollama pull qwen3.5:2b-q4_K_M || echo "ollama pull failed (is the server running?), skipped" >&2
else
  echo "ollama not found: skipped qwen3.5:2b-q4_K_M" >&2
fi

[ "$VENV" = 1 ] || exit 0

# 4. venvs at the paths eval/models.toml calls
echo "==> ad-night venv"
(cd "$HERE/ad-night/repo" && uv sync --all-extras --frozen)

echo "==> anonymous venv"
(cd "$HERE/anonymous" && { [ -x .venv/bin/python ] || uv venv --python 3.14 .venv; } \
  && uv pip install --python .venv/bin/python -r requirements.txt)
