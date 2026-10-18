"""Export a trained policy for deployment: merge its LoRA into a full bf16 text-only checkpoint (TRAINING_PLAN.md §8 P4).

The merged checkpoint is served by ../term-generalizer/serve_vllm.sh with VRAM_GB set (online fp8). A PPO adapter is merged onto
the checkpoint it was trained from (ckpt/sft_merged, recorded in the step's state.json).

  .venv-train/bin/python rl/export.py --adapter ckpt/ppo/step_20/actor --out ckpt/export/pilot_ppo
  .venv-train/bin/python rl/export.py --dummy --out ckpt/export/dummy     gate 2: a random LoRA merged into the base
then
  VRAM_GB=3.8 MAX_MODEL_LEN=18944 bash ../term-generalizer/serve_vllm.sh ckpt/export/pilot_ppo nw-policy
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import train_common as T  # noqa: E402


def dummy_adapter(base: Path, out: Path, r: int = 32) -> Path:
    """A LoRA with small random weights on every target module (CPU), so the merge really changes the weights."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16)
    m = T.add_lora(m, r, 2 * r)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 1e-4)
    m.save_pretrained(str(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="LoRA directory (an SFT adapter or ckpt/ppo/<step>/actor)")
    ap.add_argument("--base", default=None, help="checkpoint the adapter was trained on (default: from state.json, "
                                                 "else models/Qwen3.5-2B)")
    ap.add_argument("--dummy", action="store_true", help="merge a random LoRA into the base (deployment gate)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    T.setup_env()
    base = Path(args.base) if args.base else None
    if base is None and args.adapter:
        state = Path(args.adapter).parent / "state.json"
        base = Path(json.loads(state.read_text())["base"]) if state.exists() else T.BASE_MODEL
    base = base or T.BASE_MODEL
    if args.dummy:
        tmp = Path(tempfile.mkdtemp(prefix="nw_dummy_lora_"))
        adapter = dummy_adapter(base, tmp)
    elif args.adapter:
        adapter = Path(args.adapter)
    else:
        ap.error("--adapter or --dummy")
    out = T.merge_and_save(base, adapter, Path(args.out))
    print(f"[export] {adapter} + {base} -> {out}")
    print(f"serve: VRAM_GB=3.8 MAX_MODEL_LEN=18944 bash ../term-generalizer/serve_vllm.sh {out} nw-policy")


if __name__ == "__main__":
    main()
