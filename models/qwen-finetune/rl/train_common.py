"""Shared pieces of SFT and PPO: environment checks, model loading, LoRA, response log-probs, checkpoint merging.

Runs in .venv-train (the .venv-vllm stack plus peft, accelerate and flash-linear-attention, see rl/README.md).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
from common import ROOT  # noqa: E402

BASE_MODEL = ROOT / "models" / "Qwen3.5-2B"
CKPT = ROOT / "ckpt"
# every linear projection, including the Gated-DeltaNet in/out projections; lm_head is tied to the embeddings
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"]


def setup_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # FlashInfer JIT sampling fails on the 545 driver
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def check_free_vram(min_free_gb: float) -> None:
    """Fail before loading anything if another process (a `vllm serve`) holds the GPU."""
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < min_free_gb:
        try:
            procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        except OSError:
            procs = "?"
        raise SystemExit(f"only {free / 2**30:.1f} of {total / 2**30:.1f} GiB free (need {min_free_gb}); stop every "
                         f"`vllm serve` first. GPU processes:\n{procs}")


def load_tokenizer(path=BASE_MODEL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def im_end_id(tok) -> int:
    return tok.convert_tokens_to_ids("<|im_end|>")


def load_lm(path=BASE_MODEL, *, grad_ckpt: bool = True, device="cuda"):
    """Text-only Qwen3.5 causal LM (the vision tower and MTP head are not loaded), bf16."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16, attn_implementation="sdpa")
    m.config.use_cache = False
    if grad_ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
    return m.to(device)


def add_lora(model, r: int, alpha: int, dropout: float = 0.0, targets=None, adapter_path=None, name="default"):
    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter_path:
        return PeftModel.from_pretrained(model, str(adapter_path), adapter_name=name, is_trainable=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets or LORA_TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg, adapter_name=name)


def hidden_states(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Last hidden layer (B, T, H) without materializing the logits."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = base.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.last_hidden_state


def token_logprobs(model, hidden: torch.Tensor, targets: torch.Tensor, chunk: int = 1024,
                   with_entropy: bool = False):
    """log p(targets) from hidden states (N, H), computed in chunks: full logits at 7K x 248K would not fit."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.lm_head
    lps, ents = [], []
    for s in range(0, hidden.shape[0], chunk):
        logits = head(hidden[s:s + chunk]).float()
        lp = F.log_softmax(logits, dim=-1)
        lps.append(lp.gather(-1, targets[s:s + chunk, None]).squeeze(-1))
        if with_entropy:   # monitoring only: no graph, or the full-vocabulary tensor would be kept for backward
            with torch.no_grad():
                ents.append(-(lp.exp() * lp).sum(-1))
    lp = torch.cat(lps)
    return (lp, torch.cat(ents)) if with_entropy else lp


def response_logprobs(model, input_ids: torch.Tensor, prompt_len: int, chunk: int = 1024, with_entropy=False):
    """Log-probs of the response tokens input_ids[prompt_len:] of one sequence (1, T)."""
    h = hidden_states(model, input_ids)[0, prompt_len - 1:-1]
    return token_logprobs(model, h, input_ids[0, prompt_len:], chunk, with_entropy)


def copy_tokenizer(dst: Path, src: Path = BASE_MODEL) -> None:
    for f in TOKENIZER_FILES:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)


def merge_and_save(base_path, adapter_path, out: Path) -> Path:
    """base + LoRA -> full bf16 text-only checkpoint that vLLM serves as Qwen3_5ForCausalLM."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(base_path), dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, str(adapter_path)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(out), safe_serialization=True)
    copy_tokenizer(out, Path(base_path) if (Path(base_path) / "tokenizer.json").exists() else BASE_MODEL)
    (out / "merge_info.json").write_text(json.dumps({"base": str(base_path), "adapter": str(adapter_path)}, indent=1))
    return out


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def vram_headroom_gb() -> float:
    """GPU memory left at this process's peak reservation, given what other processes (a sleeping vLLM engine) hold
    now: total - others - max_memory_reserved."""
    free, total = torch.cuda.mem_get_info()
    others = total - free - torch.cuda.memory_reserved()
    return (total - others - torch.cuda.max_memory_reserved()) / 2**30


class JsonlLog:
    """Appends rows to a JSONL file; with tb_dir, also writes their numeric values as TensorBoard scalars.

    Tags are "<prefix>/<key>", where a row's "phase" field replaces the prefix (critic_warmup/ev_after); a nested dict
    becomes "<key>/<subkey>" (the PPO dev rows: dev/pii, ...). The x axis is the row's "step" (0 if absent).
    """

    def __init__(self, path: Path, tb_dir: Path | None = None, prefix: str = "train"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.tb = None
        if tb_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(tb_dir))

    def write(self, row: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.tb is None:
            return
        step = int(row.get("step", 0))
        prefix = row.get("phase", self.prefix)

        def scalars(d: dict, tag: str | None):
            for k, v in d.items():
                if isinstance(v, dict):
                    yield from scalars(v, f"{tag}/{k}" if tag else k)
                elif isinstance(v, (int, float)) and not isinstance(v, bool) and k != "step":
                    yield f"{tag or prefix}/{k}", v

        for tag, v in scalars(row, None):
            self.tb.add_scalar(tag, v, step)
        self.tb.flush()


def start_tensorboard(logdir: Path, port: int, host: str = "0.0.0.0") -> None:
    """Serve TensorBoard on host:port for logdir in a detached process that outlives training (port 0: do nothing).

    If the port is already taken, an earlier run's server is assumed to be there and nothing is started; it shows
    the new run too when it was started on the same logdir. Stop it with `pkill -f tensorboard.main`.
    """
    if not port:
        return
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"[tb] port {port} already in use: assuming TensorBoard runs there (logdir {logdir})")
            return
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / "tensorboard.log", "a") as f:
        p = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--host", host,
                              "--port", str(port), "--reload_interval", "15"],
                             stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    import time
    for _ in range(120):   # wait until it listens (or dies), so that the message below is true
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        if p.poll() is not None:
            print(f"[tb] TensorBoard exited with code {p.returncode}; see {logdir / 'tensorboard.log'}", flush=True)
            return
        time.sleep(0.5)
    print(f"[tb] TensorBoard pid {p.pid} on {host}:{port} -> http://{socket.gethostname()}:{port} (logdir {logdir})",
          flush=True)
