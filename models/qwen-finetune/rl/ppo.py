"""PPO with a token-level critic for the de-identification policy (TRAINING_PLAN.md §4, §6-§8).

A custom loop on the .venv-train stack (verl was ruled out by its version pins; see rl/README.md):
  rollouts   in-process vLLM on the merged SFT checkpoint; the current actor LoRA is loaded per step as a LoRARequest
             (--rollout-sync merged reloads a merged checkpoint instead, if vLLM LoRA fails on the hybrid layers)
  actor      HF model + PEFT LoRA (all linear projections incl. Gated-DeltaNet in/out)
  reference  the same model with the adapter disabled (= the merged SFT model)
  critic     a second copy of the merged SFT model + its own LoRA + a scalar value head
  reward     rl/reward.py, one reward per decision on the token that closes its JSON item, computed in a process pool
  advantage  GAE (gamma 1, lambda 0.95) on rewards minus beta*KL per token, whitened per batch

Phases (pilot clock in TRAINING_PLAN.md §8): optional critic warm-up on rollouts of the starting policy (P2, critic
only), then PPO steps with dev validation and checkpoints every --val-every / --ckpt-every steps. The run aborts
(exit code 3, see abort.json) on the plan's conditions: critic explained variance < 0.1 after 10 steps, JSON validity
below min(95%, starting validity - 5 points) at T=1.0, entities per encounter > 1.5x silver, KL > 20.

  .venv-train/bin/python rl/ppo.py --smoke                    2 steps x 8 prompts, logs the token of every reward
  .venv-train/bin/python rl/ppo.py --warmup-rollouts 256      pilot (P2 + P3) from ckpt/sft_merged
  .venv-train/bin/python rl/ppo.py --init base --out ckpt/ppo_from_base     PPO without SFT (ablation arm)
  .venv-train/bin/python rl/ppo.py --preset large --init-adapter ckpt/ppo/step_20/actor   large-run settings
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import data as D  # noqa: E402
import reward as W  # noqa: E402
import train_common as T  # noqa: E402


@dataclass
class Config:
    init: str = "sft"                       # sft: start from ckpt/sft_merged; base: from the untrained model
    base: str = ""                          # explicit starting checkpoint (overrides --init)
    init_adapter: str = ""                  # continue from a saved actor LoRA
    out: str = str(T.CKPT / "ppo")
    seed: int = 0
    steps: int = 28
    prompts_per_step: int = 16
    n: int = 4
    max_prompt: int = 6144
    max_resp: int = 1024
    p_prefill: float = 0.5
    synth: bool = True
    synth_share: float = 0.4                # share of prompts from synthetic records (< 0: uniform over all records)
    temperature: float = 1.0
    top_p: float = 1.0
    actor_r: int = 32
    actor_alpha: int = 64
    actor_lr: float = 2e-5
    critic_r: int = 32
    critic_alpha: int = 64
    critic_lr: float = 5e-5
    cosine: bool = False
    epochs: int = 1
    minibatch: int = 16                     # sequences per optimizer step
    clip: float = 0.2
    vclip: float = 0.2
    kl_coef: float = 0.01
    kl_adaptive: bool = False
    kl_target: float = 6.0                  # per-sequence KL target of the adaptive controller
    gamma: float = 1.0
    lam: float = 0.95
    max_grad_norm: float = 1.0
    warmup_rollouts: int = 0                # P2: critic-only warm-up on this many rollouts of the starting policy
    warmup_epochs: int = 2
    val_every: int = 10
    ckpt_every: int = 20
    vllm_mem: float = 0.30
    rollout_sync: str = "lora"              # lora | merged
    reward_workers: int = 16
    min_free_gb: float = 40.0
    min_headroom_gb: float = 4.0            # warn when a step leaves less GPU memory than this unused at its peak
    tb_port: int = 6006                     # TensorBoard on 0.0.0.0:PORT for the parent of --out (0: event files only)
    abort_after: int = 10
    abort_ev: float = 0.1
    abort_json: float = 0.95
    abort_entities: float = 1.5
    abort_kl: float = 20.0
    no_abort: bool = False
    smoke: bool = False
    reward: dict = field(default_factory=lambda: dataclasses.asdict(W.RewardConfig()))


PRESETS = {
    "large": dict(max_prompt=16384, max_resp=2560, prompts_per_step=24, steps=100, actor_r=64, actor_alpha=128,
                  actor_lr=1e-5, cosine=True, critic_r=64, critic_alpha=128, epochs=2, minibatch=32,
                  kl_adaptive=True, kl_target=6.0, warmup_rollouts=8 * 24 * 4, val_every=20, ckpt_every=50,
                  vllm_mem=0.35),
    "smoke": dict(steps=2, prompts_per_step=8, val_every=0, ckpt_every=0, warmup_rollouts=0, smoke=True,
                  no_abort=True),
}


# ---------------------------------------------------------------- rewards (process pool)

def _reward_job(job: tuple) -> dict:
    rec, gold, response, prefix, rcfg = job
    sv = W.prepare(rec, gold)
    res = W.compute(rec, sv, response, prefix, W.RewardConfig(**rcfg))
    return {"items": res.items, "parts": res.parts, "valid": res.valid,
            "n_silver": len(D.silver_target(rec, gold)["entities"]) if res.valid else None}


# ---------------------------------------------------------------- rollout engine

class Rollouts:
    def __init__(self, cfg: Config, base_path: Path, tok):
        from vllm import LLM
        self.cfg, self.tok, self.base_path = cfg, tok, base_path
        self.lora_id, self.lora_path = 0, None
        self.llm = LLM(model=str(base_path), tokenizer=str(base_path), dtype="bfloat16", seed=cfg.seed,
                       gpu_memory_utilization=cfg.vllm_mem, max_model_len=cfg.max_prompt + cfg.max_resp + 64,
                       enable_sleep_mode=True, enable_prefix_caching=True, generation_config="vllm",
                       enable_lora=cfg.rollout_sync == "lora", max_lora_rank=max(cfg.actor_r, 8), max_loras=1)
        self.awake = True

    def sleep(self):
        if self.awake:
            self.llm.sleep(level=1)
            self.awake = False

    def wake(self):
        if not self.awake:
            import torch
            torch.cuda.empty_cache()
            self.llm.wake_up()
            self.awake = True

    def set_policy(self, actor, out: Path, step: int):
        """Make the next rollouts use the current actor."""
        if self.cfg.rollout_sync == "lora":
            path = out / "rollout_adapter"
            if path.exists():
                shutil.rmtree(path)
            actor.save_pretrained(str(path))
            self.lora_id += 1
            self.lora_path = path
        else:   # merge the adapter into a full checkpoint in /dev/shm and reload it into the sleeping engine
            from safetensors.torch import save_file
            path = Path("/dev/shm") / f"nw_ppo_{os.getpid()}"
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)
            save_file(_merged_state_dict(actor), str(path / "model.safetensors"))
            shutil.copy(self.base_path / "config.json", path / "config.json")
            self.wake()
            self.llm.collective_rpc("reload_weights", kwargs={"weights_path": str(path)})
            self.sleep()

    def lora_request(self):
        if self.cfg.rollout_sync != "lora" or self.lora_path is None:
            return None
        from vllm.lora.request import LoRARequest
        return LoRARequest(f"actor{self.lora_id}", self.lora_id, str(self.lora_path))

    def generate(self, prompts: list[list[int]], n: int, temperature: float, max_tokens: int):
        from vllm import SamplingParams
        self.wake()
        sp = SamplingParams(n=n, temperature=temperature, top_p=self.cfg.top_p if temperature > 0 else 1.0,
                            max_tokens=max_tokens, stop_token_ids=[T.im_end_id(self.tok)], presence_penalty=0.0,
                            frequency_penalty=0.0, repetition_penalty=1.0, seed=random.randrange(2**31))
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sp, lora_request=self.lora_request(),
                                 use_tqdm=False)
        self.sleep()
        res = []
        for o in outs:
            group = []
            for c in o.outputs:
                ids = list(c.token_ids)
                if c.finish_reason == "stop" and (not ids or ids[-1] != T.im_end_id(self.tok)):
                    ids.append(T.im_end_id(self.tok))
                group.append({"ids": ids, "text": self.tok.decode(ids, skip_special_tokens=True),
                              "finish": c.finish_reason})
            res.append(group)
        return res


def _merged_state_dict(peft_model) -> dict:
    """W + B@A*scale for every LoRA layer, without modifying the model."""
    import torch
    sd = {}
    base = peft_model.get_base_model()
    with torch.no_grad():
        for name, mod in base.named_modules():
            if hasattr(mod, "base_layer") and hasattr(mod, "lora_A") and "default" in mod.lora_A:
                w = mod.base_layer.weight
                delta = (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()) \
                    * mod.scaling["default"]
                sd[f"{name}.weight"] = (w.float() + delta).to(w.dtype).cpu()
        for name, p in base.named_parameters():
            if "lora_" not in name and ".base_layer." not in name:
                sd.setdefault(name, p.detach().cpu())
    return sd


# ---------------------------------------------------------------- models

class Critic:
    def __init__(self, base_path: Path, cfg: Config, device: str):
        import torch
        self.model = T.add_lora(T.load_lm(base_path, device=device), cfg.critic_r, cfg.critic_alpha)
        hdim = self.model.get_base_model().config.hidden_size
        self.head = torch.nn.Linear(hdim, 1, dtype=torch.float32, device=device)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.params = [p for p in self.model.parameters() if p.requires_grad] + list(self.head.parameters())
        # from_pretrained returns eval mode, and HF skips gradient checkpointing outside train mode: without this the
        # critic keeps every layer's activations (OOM on a 44.5 GiB GPU at ~11K tokens, job 36208)
        self.model.train()

    def values(self, input_ids, attention_mask, prompt_lens, resp_lens):
        h = T.hidden_states(self.model, input_ids, attention_mask)
        out = []
        for b, (pl, rl) in enumerate(zip(prompt_lens, resp_lens)):
            # value of the state before each response token is generated
            out.append(self.head(h[b, pl - 1:pl - 1 + rl].float()).squeeze(-1))
        return out

    def save(self, path: Path):
        import torch
        self.model.save_pretrained(str(path))
        torch.save(self.head.state_dict(), path / "value_head.pt")

    def load_head(self, path: Path):
        import torch
        self.head.load_state_dict(torch.load(path / "value_head.pt"))


def group_tensors(seqs: list[dict], device):
    """Right-padded batch of one prompt group (padding after the content leaves earlier positions unchanged)."""
    import torch
    L = max(len(s["prompt"]) + len(s["ids"]) for s in seqs)
    ids = torch.full((len(seqs), L), 0, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        x = s["prompt"] + s["ids"]
        ids[b, :len(x)] = torch.tensor(x, device=device)
        mask[b, :len(x)] = 1
    return ids, mask, [len(s["prompt"]) for s in seqs], [len(s["ids"]) for s in seqs]


def actor_logprobs(actor, ids, mask, pls, rls, with_entropy=False):
    h = T.hidden_states(actor, ids, mask)
    out = []
    for b, (pl, rl) in enumerate(zip(pls, rls)):
        out.append(T.token_logprobs(actor, h[b, pl - 1:pl - 1 + rl], ids[b, pl:pl + rl], with_entropy=with_entropy))
    return out


def gae(rewards, values, gamma, lam):
    import torch
    T_ = len(rewards)
    adv = torch.zeros(T_, device=values.device)
    last = 0.0
    for t in reversed(range(T_)):
        nv = values[t + 1] if t + 1 < T_ else 0.0
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


# ---------------------------------------------------------------- the trainer

class Trainer:
    def __init__(self, cfg: Config):
        import torch
        self.cfg = cfg
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
        self.log = T.JsonlLog(self.out / "log.jsonl", tb_dir=self.out / "tb")
        self.samples_log = T.JsonlLog(self.out / "samples.jsonl")
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        T.check_free_vram(cfg.min_free_gb)
        T.start_tensorboard(self.out.parent, cfg.tb_port)
        # reward workers are spawned before CUDA is used in this process
        self.pool = ProcessPoolExecutor(max_workers=cfg.reward_workers, mp_context=multiprocessing.get_context("spawn"))
        self.base_path = self._base_path()
        self.tok = T.load_tokenizer(self.base_path)
        recs, golds = D.load(D.TRAIN_DATA, D.TRAIN_GOLD)
        ids, sids = D.splits()["train"], []
        if cfg.synth:
            srecs, sgolds = D.load(D.SYNTH_DATA, D.SYNTH_GOLD)
            recs.update(srecs)
            golds.update(sgolds)
            sids = sorted(sgolds)
            ids = ids + sids
        self.sampler = D.Sampler(ids, recs, golds, seed=cfg.seed, p_prefill=cfg.p_prefill,
                                 max_prompt_tokens=cfg.max_prompt, tokenizer=self.tok, synth_ids=sids,
                                 synth_share=cfg.synth_share)
        # vLLM first: it sizes its memory pool from the free memory at startup
        self.engine = Rollouts(cfg, self.base_path, self.tok)
        self.engine.sleep()
        dev = "cuda"
        self.actor = T.add_lora(T.load_lm(self.base_path, device=dev), cfg.actor_r, cfg.actor_alpha,
                                adapter_path=cfg.init_adapter or None)
        self.critic = Critic(self.base_path, cfg, dev)
        self.a_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.a_opt = torch.optim.AdamW(self.a_params, lr=cfg.actor_lr, weight_decay=0.0)
        self.c_opt = torch.optim.AdamW(self.critic.params, lr=cfg.critic_lr, weight_decay=0.0)
        self.kl_coef = cfg.kl_coef
        self.device = dev
        self.json_baseline = None   # JSON validity of the starting policy (warm-up batch, else step 1)
        if cfg.init_adapter:
            self.engine.set_policy(self.actor, self.out, 0)

    def _base_path(self) -> Path:
        if self.cfg.base:
            return Path(self.cfg.base)
        if self.cfg.init == "sft":
            p = T.CKPT / "sft_merged"
            if not (p / "config.json").exists():
                raise SystemExit(f"{p} missing: run rl/sft.py first (or --init base)")
            return p
        p = T.CKPT / "base_text"   # text-only copy of the untrained model, so vLLM and HF load the same weights
        if not (p / "config.json").exists():
            from transformers import AutoModelForCausalLM
            import torch
            m = AutoModelForCausalLM.from_pretrained(str(T.BASE_MODEL), dtype=torch.bfloat16)
            p.mkdir(parents=True, exist_ok=True)
            m.save_pretrained(str(p))
            T.copy_tokenizer(p)
            del m
        return p

    # ---- rollouts + scoring
    def collect(self, n_prompts: int, temperature: float):
        cfg = self.cfg
        samples = [self.sampler.draw() for _ in range(n_prompts)]
        prompts = [self.tok(s.prompt_text(self.tok), add_special_tokens=False)["input_ids"] for s in samples]
        t0 = time.time()
        gens = self.engine.generate(prompts, cfg.n, temperature, cfg.max_resp)
        t_gen = time.time() - t0
        jobs, seqs = [], []
        for s, p, group in zip(samples, prompts, gens):
            for g in group:
                jobs.append((s.record, s.gold, g["text"], s.prefix(), cfg.reward))
                seqs.append({"sample": s, "prompt": p, "ids": g["ids"], "text": g["text"], "finish": g["finish"],
                             "group": len(seqs) // cfg.n})
        results = list(self.pool.map(_reward_job, jobs, chunksize=2))
        for sq, r in zip(seqs, results):
            ends = W.token_char_ends(self.tok, sq["ids"])
            res = W.RewardResult(r["valid"], r["items"], r["parts"])
            sq["rewards"] = W.to_token_rewards(res, ends, len(sq["sample"].prefix()))
            sq["parts"], sq["valid"], sq["n_silver"], sq["items"], sq["ends"] = \
                r["parts"], r["valid"], r["n_silver"], r["items"], ends
        return seqs, t_gen

    def score_old(self, seqs):
        """old log-probs (actor), reference log-probs (adapter off) and values, without gradients."""
        import torch
        self.actor.eval()
        self.critic.model.eval()
        with torch.no_grad():
            # one sequence per forward, as in update(): a padded group of 4 x 18K tokens peaks far higher than 1
            for g in ([s] for s in seqs):
                ids, mask, pls, rls = group_tensors(g, self.device)
                old = actor_logprobs(self.actor, ids, mask, pls, rls)
                with self.actor.disable_adapter():
                    ref = actor_logprobs(self.actor, ids, mask, pls, rls)
                vals = self.critic.values(ids, mask, pls, rls)
                for s, o, r, v in zip(g, old, ref, vals):
                    s["old_lp"], s["ref_lp"], s["values"] = o.float(), r.float(), v.float()
        self.actor.train()
        self.critic.model.train()   # train mode = gradient checkpointing in update()

    def _groups(self, seqs):
        gs = {}
        for s in seqs:
            gs.setdefault(s["group"], []).append(s)
        return list(gs.values())

    def advantages(self, seqs, use_kl=True):
        import torch
        kls = []
        for s in seqs:
            r = torch.tensor(s["rewards"], device=self.device, dtype=torch.float32)
            kl = s["old_lp"] - s["ref_lp"]
            kls.append(kl.sum().item())
            if use_kl:
                r = r - self.kl_coef * kl
            s["adv"], s["ret"] = gae(r, s["values"], self.cfg.gamma, self.cfg.lam)
        allv = torch.cat([s["adv"] for s in seqs])
        mu, sd = allv.mean(), allv.std().clamp_min(1e-6)
        for s in seqs:
            s["adv"] = (s["adv"] - mu) / sd
        rets, vals = torch.cat([s["ret"] for s in seqs]), torch.cat([s["values"] for s in seqs])
        ev = 1 - (rets - vals).var() / rets.var().clamp_min(1e-8)
        return sum(kls) / len(kls), ev.item()

    # ---- updates
    def update(self, seqs, train_actor=True, epochs=None):
        import torch
        cfg = self.cfg
        groups = self._groups(seqs)
        per_mb = max(1, cfg.minibatch // cfg.n)
        stats = {"pg_loss": [], "v_loss": [], "clipfrac": [], "entropy": [], "approx_kl": [], "a_gn": [], "c_gn": []}
        for _ in range(epochs or cfg.epochs):
            random.shuffle(groups)
            for i in range(0, len(groups), per_mb):
                mb = groups[i:i + per_mb]
                n_tok = sum(len(s["ids"]) for g in mb for s in g)
                # one sequence per forward/backward: a padded group of 4 x 6K tokens does not fit next to vLLM
                for g in ([s] for grp in mb for s in grp):
                    ids, mask, pls, rls = group_tensors(g, self.device)
                    if train_actor:
                        new = actor_logprobs(self.actor, ids, mask, pls, rls, with_entropy=True)
                        loss = 0.0
                        for s, (lp, ent) in zip(g, new):
                            ratio = torch.exp(lp.float() - s["old_lp"])
                            a = s["adv"]
                            pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
                            loss = loss + pg.sum() / n_tok
                            stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                            stats["entropy"].append(ent.mean().item())
                            stats["approx_kl"].append((s["old_lp"] - lp.float()).mean().item())
                        loss.backward()
                        stats["pg_loss"].append(loss.item())
                    vals = self.critic.values(ids, mask, pls, rls)
                    vloss = 0.0
                    for s, v in zip(g, vals):
                        vc = s["values"] + (v - s["values"]).clamp(-cfg.vclip, cfg.vclip)
                        vl = 0.5 * torch.max((v - s["ret"]) ** 2, (vc - s["ret"]) ** 2)
                        vloss = vloss + vl.sum() / n_tok
                    vloss.backward()
                    stats["v_loss"].append(vloss.item())
                if train_actor:
                    stats["a_gn"].append(torch.nn.utils.clip_grad_norm_(self.a_params, cfg.max_grad_norm).item())
                    self.a_opt.step()
                    self.a_opt.zero_grad(set_to_none=True)
                stats["c_gn"].append(torch.nn.utils.clip_grad_norm_(self.critic.params, cfg.max_grad_norm).item())
                self.c_opt.step()
                self.c_opt.zero_grad(set_to_none=True)
        return {k: sum(v) / len(v) for k, v in stats.items() if v}

    # ---- metrics
    def batch_metrics(self, seqs) -> dict:
        n = len(seqs)
        valid = [s for s in seqs if s["valid"]]
        m = {"json_valid": len(valid) / n, "return": sum(s["parts"]["return"] for s in seqs) / n,
             "resp_len": sum(len(s["ids"]) for s in seqs) / n,
             "truncated": sum(s["finish"] == "length" for s in seqs) / n,
             "prefill_share": sum(s["sample"].prefill for s in seqs) / n}
        for k in ("dP", "dU", "fp", "kill_evidence", "miss", "type", "keep", "dx", "evidence", "rejected", "P", "U"):
            m[k] = sum(s["parts"].get(k, 0.0) for s in valid) / max(1, len(valid))
        ents = sum(s["parts"].get("n_entities", 0) for s in valid)
        silver = sum(s["n_silver"] or 0 for s in valid)
        m["entities_per_enc"] = ents / max(1, len(valid))
        m["silver_entities_per_enc"] = silver / max(1, len(valid))
        m["entities_vs_silver"] = ents / silver if silver else None
        m["rejected_rate"] = sum(s["parts"].get("n_rejected", 0) for s in valid) / max(1, ents)
        return m

    def log_reward_tokens(self, seqs, k=2):
        """Smoke test: the decoded token at every non-zero reward of the first k rollouts."""
        for s in seqs[:k]:
            print(f"--- rollout ({'prefill' if s['sample'].prefill else 'free'}, valid={s['valid']}, "
                  f"return {s['parts']['return']:.3f})")
            for j, r in enumerate(s["rewards"]):
                if r:
                    print(f"  tok {j:4d} {self.tok.decode(s['ids'][j:j + 1])!r:>12} reward {r:+.4f}")

    # ---- validation on dev (eval reference labels; selection only, never reward)
    def validate(self, step: int) -> dict:
        import infer
        recs = [r for r in infer.load_eval() if r["encounter_id"] in set(D.splits()["dev"])]
        prompts, prefixes = [], []
        for r in recs:
            s = D.Sample(r["encounter_id"], r, {}, True, ["dialogue", "note"])
            prompts.append(self.tok(s.prompt_text(self.tok), add_special_tokens=False)["input_ids"])
            prefixes.append(s.prefix())
        gens = self.engine.generate(prompts, 1, 0.0, self.cfg.max_resp)
        outputs = [p + g[0]["text"] for p, g in zip(prefixes, gens)]
        m = infer.score_outputs(recs, outputs)
        m["step"] = step
        self.log.write({"dev": m, "step": step})
        print(f"[ppo] dev step {step}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                       for k, v in m.items()}), flush=True)
        return m

    def save(self, step: int, tag: str | None = None):
        d = self.out / (tag or f"step_{step}")
        self.actor.save_pretrained(str(d / "actor"))
        self.critic.save(d / "critic")
        T.copy_tokenizer(d / "actor")
        (d / "state.json").write_text(json.dumps({"step": step, "kl_coef": self.kl_coef,
                                                  "base": str(self.base_path)}))

    # ---- phases
    def warmup_critic(self):
        cfg = self.cfg
        n_prompts = math.ceil(cfg.warmup_rollouts / cfg.n)
        t0 = time.time()
        seqs, _ = self.collect(n_prompts, cfg.temperature)
        self.score_old(seqs)
        _, ev0 = self.advantages(seqs)
        stats = self.update(seqs, train_actor=False, epochs=cfg.warmup_epochs)
        self.score_old(seqs)
        _, ev1 = self.advantages(seqs)
        row = {"phase": "critic_warmup", "rollouts": len(seqs), "ev_before": ev0, "ev_after": ev1,
               "seconds": round(time.time() - t0, 1), **self.batch_metrics(seqs), **stats}
        self.json_baseline = row["json_valid"]
        self.log.write(row)
        print(f"[ppo] critic warm-up on {len(seqs)} rollouts: explained variance {ev0:.3f} -> {ev1:.3f} "
              f"({row['seconds']}s)", flush=True)

    def run(self):
        import torch
        cfg = self.cfg
        if cfg.warmup_rollouts:
            self.warmup_critic()
        sched = None
        if cfg.cosine:
            sched = torch.optim.lr_scheduler.LambdaLR(self.a_opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / cfg.steps)))
        for step in range(1, cfg.steps + 1):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            seqs, t_gen = self.collect(cfg.prompts_per_step, cfg.temperature)
            if cfg.smoke:
                self.log_reward_tokens(seqs)
            t1 = time.time()
            self.score_old(seqs)
            kl, ev = self.advantages(seqs)
            stats = self.update(seqs)
            if sched:
                sched.step()
            self.engine.set_policy(self.actor, self.out, step)
            m = {"step": step, **self.batch_metrics(seqs), **stats, "kl": kl, "kl_coef": self.kl_coef,
                 "explained_var": ev, "t_gen": round(t_gen, 1), "t_train": round(time.time() - t1, 1),
                 "t_step": round(time.time() - t0, 1), "peak_vram_gb": round(T.peak_vram_gb(), 2),
                 "headroom_gb": round(T.vram_headroom_gb(), 2)}
            if self.json_baseline is None:
                self.json_baseline = m["json_valid"]
            self.log.write(m)
            for s in seqs[:: cfg.n][:4]:
                self.samples_log.write({"step": step, "eid": s["sample"].eid, "ops": s["sample"].ops,
                                        "prefill": s["sample"].prefill, "return": s["parts"]["return"],
                                        "text": s["text"][:4000]})
            print(f"[ppo] step {step}/{cfg.steps} return {m['return']:+.3f} json {m['json_valid']:.2f} "
                  f"ents {m['entities_per_enc']:.1f}/{m['silver_entities_per_enc']:.1f} kl {kl:.2f} ev {ev:.2f} "
                  f"clip {stats.get('clipfrac', 0):.3f} len {m['resp_len']:.0f} {m['t_step']}s "
                  f"(gen {m['t_gen']}s) peak {m['peak_vram_gb']} GiB headroom {m['headroom_gb']} GiB", flush=True)
            if m["headroom_gb"] < cfg.min_headroom_gb:
                print(f"[ppo] WARNING: only {m['headroom_gb']} GiB GPU headroom at this step's peak "
                      f"(< {cfg.min_headroom_gb}); a longer batch may OOM", flush=True)
            if cfg.kl_adaptive:
                err = max(-0.2, min(0.2, kl / cfg.kl_target - 1))
                self.kl_coef *= 1 + err * len(seqs) / 1000
            if cfg.ckpt_every and step % cfg.ckpt_every == 0 and step < cfg.steps:   # the last step is saved as final
                self.save(step)
            if cfg.val_every and step % cfg.val_every == 0:
                self.validate(step)
            reason = None if cfg.no_abort else self.abort_reason(step, m)
            if reason:
                self.save(step, "aborted")
                (self.out / "abort.json").write_text(json.dumps({"step": step, "reason": reason, "metrics": m}))
                print(f"[ppo] ABORT at step {step}: {reason}. Ship the SFT model (arm 2).", flush=True)
                sys.exit(3)
        self.save(cfg.steps, "final")

    def abort_reason(self, step: int, m: dict) -> str | None:
        c = self.cfg
        # the SFT policy samples ~86% valid JSON at T=1.0 (smoke test, 2026-10-17), so an absolute 95% would stop the
        # run before PPO acts on the -1 reward; abort when PPO makes validity worse than where it started
        floor = min(c.abort_json, (self.json_baseline or 1.0) - 0.05)
        if m["json_valid"] < floor and step >= 3:
            return f"JSON validity {m['json_valid']:.2f} < {floor:.2f} (start {self.json_baseline:.2f})"
        if m["entities_vs_silver"] and m["entities_vs_silver"] > c.abort_entities:
            return f"entities per encounter {m['entities_vs_silver']:.2f}x silver > {c.abort_entities}x"
        if m["kl"] > c.abort_kl:
            return f"KL {m['kl']:.1f} > {c.abort_kl}"
        if step >= c.abort_after and m["explained_var"] < c.abort_ev:
            return f"critic explained variance {m['explained_var']:.3f} < {c.abort_ev} after {step} steps"
        return None


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--smoke", action="store_true", help="= --preset smoke")
    for f in dataclasses.fields(Config):
        if f.name in ("reward", "smoke"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(flag, type={"int": int, "float": float, "str": str}[str(f.type)], default=None)
    for k, v in dataclasses.asdict(W.RewardConfig()).items():
        ap.add_argument("--r-" + k.replace("_", "-"), type=type(v) if not isinstance(v, bool) else None,
                        default=None, dest="r_" + k, help=f"reward: {k} (default {v})")
    a = ap.parse_args()
    cfg = Config()
    preset = "smoke" if a.smoke else a.preset
    for k, v in PRESETS.get(preset, {}).items():
        setattr(cfg, k, v)
    for f in dataclasses.fields(Config):
        v = getattr(a, f.name, None)
        if v is not None and f.name not in ("reward", "smoke"):
            setattr(cfg, f.name, v)
    for k in cfg.reward:
        v = getattr(a, "r_" + k, None)
        if v is not None:
            cfg.reward[k] = v
    if cfg.smoke and cfg.out == Config.out:
        cfg.out = str(T.CKPT / "ppo_smoke")
    return cfg


def main():
    T.setup_env()
    cfg = parse_args()
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
