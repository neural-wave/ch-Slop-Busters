# rl/: SFT + PPO for the de-identification policy

Implementation of [train/TRAINING_PLAN.md](../train/TRAINING_PLAN.md). The policy (Qwen3.5-2B + LoRA) reads one
encounter and writes JSON decisions (`dx` → `evidence` → `entities` → `keep`, see `spec.py`). Deterministic code
renders them (`render.py`), so the model never rewrites text and `icd10_codes` stay unchanged.

| file | role | plan |
|---|---|---|
| `spec.py` | types, system prompt, output format, dx prefix, ICD-10-CM table, Y92 category words | §2 |
| `render.py` | decisions + regex pre-pass → anonymized record + explanation log (linking, `ctx`, span limits, keep vetoes) | §2 |
| `reward.py` | one reward per decision on its closing token; token mapping | §4 |
| `augment.py` | ICD-safe PII crossover, casing, punctuation, speaker tags, layout, sub-windows | §5 |
| `data.py` | splits, silver → target JSON, prompts, sampler (70% PII-dense, 50% prefilled) | §3, §5, §9 |
| `synth.py`, `synth/templates/` | 90 templates (PII filled in by code, `--seeds N` identities each): S001-S040 external-cause injuries, G0001-G0050 varied specialties/visits featuring quasi-identifiers and hard negatives | §3 |
| `synth_gen.py` | scenario matrix and spec for the G templates; `--check` validates/normalizes them, `--n` writes more with the API | §3 |
| `sft.py` | LoRA SFT → `ckpt/sft`, merged → `ckpt/sft_merged` | §6, P1 |
| `ppo.py` | custom PPO: vLLM rollouts with LoRA, actor/critic LoRA, GAE, KL, warm-up, validation, abort rules | §6-§8, P2-P3 |
| `export.py` | merge a LoRA into a text-only checkpoint for `../term-generalizer/serve_vllm.sh` | P4 |
| `infer.py` | anonymize with the served policy (prefill, windows, salvage) and score raw outputs | §2, P4 |
| `compare.py` | per-split metrics and paired bootstrap CIs between runs | §8, §9 |
| `gate_deploy.py` | gate 2: ≤ 3.8 GiB nvidia-smi peak at 18,944 context | §7 |
| `splits.json` | silver-val (14 train), train (126), dev (30 eval), test (37 eval) | §3, §9 |
| `resources/icd10cm_2026.tsv.gz` | CMS FY2026 ICD-10-CM order file (codes and headers), for the code-validity check | §3 |

## Environment

```bash
uv venv .venv-train --python .venv-vllm/bin/python3
echo "$PWD/.venv-vllm/lib/python3.11/site-packages" > .venv-train/lib/python3.11/site-packages/_vllm_base.pth
uv pip install --python .venv-train/bin/python --no-deps peft accelerate flash-linear-attention fla-core
source /etc/profile.d/modules.sh && module load amd/nvidia/cuda-12.9.2
```

Installed 2026-10-17: peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2 on torch 2.13 / transformers 5.17 /
vllm 0.31. `.venv-vllm` is unchanged. TensorBoard 2.21 was added the same way (`--no-deps tensorboard tensorboard-data-server absl-py
markdown werkzeug markupsafe`), so the numpy/protobuf/grpc of `.venv-vllm` stay in use.

## Order of work

Tests and data steps need no GPU:

```bash
python3 rl/data.py splits                    # rl/splits.json (already committed)
python3 rl/synth.py check && python3 rl/synth.py build --seeds 10    # 900 synthetic records
# SFT and PPO mix them in with --synth-share (default 0.4 of the samples/prompts; < 0: uniform over all records)
python3 -m unittest discover -s rl/tests     # reward §4 cells and knobs, render, augmentation invariance
.venv-train/bin/python -m unittest rl/tests/test_reward.py rl/tests/test_ppo_math.py   # + tokenizer mapping, GAE
.venv-train/bin/python rl/data.py audit      # prompt/target token lengths
```

GPU steps. Stop every `vllm serve` first; the scripts refuse to start without enough free memory.

```bash
# gate 2: deployment of a merged dummy LoRA under 3.8 GiB at 18,944 context
.venv-train/bin/python rl/export.py --dummy --out ckpt/export/dummy
VRAM_GB=3.8 MAX_MODEL_LEN=18944 bash ../term-generalizer/serve_vllm.sh ckpt/export/dummy nw-policy &
LLM_BACKEND=vllm python3 rl/gate_deploy.py --model nw-policy && python3 rl/infer.py --check-template  # then stop the server

# P1: SFT (arm 2)
.venv-train/bin/python rl/sft.py --dry-run     # sample and token statistics only
.venv-train/bin/python rl/sft.py               # -> ckpt/sft, ckpt/sft_merged

# gate 4: PPO smoke test (2 steps x 8 prompts; prints the token of every non-zero reward, s/step, peak VRAM)
.venv-train/bin/python rl/ppo.py --smoke

# P2 + P3: critic warm-up on 256 rollouts, then ~28 PPO steps, dev validation and checkpoints every 20 steps
.venv-train/bin/python rl/ppo.py --warmup-rollouts 256

# P4: export, serve at fp8, run test with and without the dx prefix (arm 3), compare against arms 1 and 2
.venv-train/bin/python rl/export.py --adapter ckpt/ppo/step_20/actor --out ckpt/export/pilot_ppo
VRAM_GB=3.8 MAX_MODEL_LEN=18944 bash ../term-generalizer/serve_vllm.sh ckpt/export/pilot_ppo nw-policy &
LLM_BACKEND=vllm python3 rl/infer.py --model nw-policy --split test --out runs/pilot_ppo
LLM_BACKEND=vllm python3 rl/infer.py --model nw-policy --split test --no-icd --out runs/pilot_ppo_noicd
python3 rl/compare.py runs/vllm4gb-qwen3.5-2b_generalize runs/pilot_sft runs/pilot_ppo --split test
```

Arm 2 is the same P4 sequence with `rl/export.py --adapter ckpt/sft`. Arm 1 on the splits comes from the existing
run: `python3 rl/compare.py runs/vllm4gb-qwen3.5-2b_generalize --split dev` (dev: PII 74.5, direct 83.7, quasi
59.8, concepts 96.0; test: 84.1 / 90.4 / 73.1 / 94.1). It still uses the old 2,000-character renderer. The P0 row with
the new renderer needs a policy-format model, so it comes with arm 2.

Ablation: `rl/ppo.py --init base --out ckpt/ppo_from_base` runs PPO without SFT, from a text-only copy of the base
model.

## Notes

- **TensorBoard.** `sft.py` and `ppo.py` mirror every numeric log value to `<out>/tb` (`train/*`, PPO `dev/*` and
  `critic_warmup/*`) and start TensorBoard on `0.0.0.0:6006` for the parent of `--out` (`ckpt/`), so SFT and PPO runs
  show side by side at `http://<node>:6006` (the URL is printed at startup). The server is detached and keeps running
  after training; a later run reuses it if the port is taken. `--tb-port 0` writes the event files without a server;
  `pkill -f tensorboard.main` stops it. Manually: `.venv-train/bin/tensorboard --logdir ckpt --host 0.0.0.0 --port 6006`.

- **Rollouts.** The actor LoRA is saved after every update and loaded by the in-process vLLM engine as a new
  `LoRARequest`. If vLLM LoRA fails on the Gated-DeltaNet projections in the smoke test, `--rollout-sync merged`
  writes the merged weights to `/dev/shm` and calls `reload_weights` instead (slower, same results).
- **Memory.** vLLM starts first at `--vllm-mem 0.30` with sleep mode, so it sleeps during training. The actor and
  critic are two bf16 copies with LoRA, both gradient-checkpointed (HF only checkpoints in `train()` mode, so the
  critic is kept there outside scoring). Scoring and updates run one sequence per forward. Log-probs are computed on
  response tokens only, in 1,024-token chunks. Each step logs `headroom_gb` (GPU memory unused at the step's peak,
  next to the sleeping vLLM) and warns below `--min-headroom-gb` (4).
- **Reward speed.** About 0.1 s per rollout, or 1-2 s for degenerate outputs with hundreds of entities. 16 spawned
  worker processes score a 64-rollout batch.
- **Not implemented yet.** The stage-B rarity term (large run, §4) is not built; it needs a corpus IDF table from the
  new datasets.
