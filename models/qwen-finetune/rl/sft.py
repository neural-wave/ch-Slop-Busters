"""SFT of the policy on silver targets (TRAINING_PLAN.md §6, pilot step P1): LoRA r=32, lr 1e-4, 3 augmented passes.

Each pass visits every train encounter (silver-val held out) once and enough synthetic records (rl/synth.py) for
--synth-share of the pass, rotating through them across passes, each with a fresh augmentation; half the samples have
the dx prefix prefilled (prompt, no loss). The loss covers the target JSON and the closing <|im_end|>. After training, the LoRA is merged into the base (ckpt/sft_merged): that checkpoint is the
PPO starting point and the arm-2 model.

  .venv-train/bin/python rl/sft.py --dry-run          build the samples and print token statistics (no GPU)
  .venv-train/bin/python rl/sft.py --export runs/sft_dataset.jsonl   also write the samples as JSONL (no GPU)
  .venv-train/bin/python rl/sft.py                    train -> ckpt/sft (adapter) and ckpt/sft_merged
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import augment as A  # noqa: E402
import data as D  # noqa: E402
import train_common as T  # noqa: E402


def build_samples(args, tok) -> tuple[list[dict], list[dict]]:
    recs, golds = D.load(D.TRAIN_DATA, D.TRAIN_GOLD)
    srecs, sgolds = D.load(D.SYNTH_DATA, D.SYNTH_GOLD)
    sp = D.splits()
    synth_ids = sorted(sgolds) if args.synth else []
    if args.synth_share >= 0 and synth_ids:   # per pass: every real encounter + a share of the synthetic records
        n_syn = min(len(synth_ids), round(len(sp["train"]) * args.synth_share / (1 - args.synth_share)))
    else:
        n_syn = len(synth_ids)
    syn_order = random.Random(args.seed + 1).sample(synth_ids, len(synth_ids))   # rotated, so passes differ
    recs.update(srecs)
    golds.update(sgolds)
    pools = A.build_pools(recs, golds)
    rng = random.Random(args.seed)

    def encode(s: D.Sample) -> dict | None:
        prompt = tok(s.prompt_text(tok), add_special_tokens=False)["input_ids"]
        target = tok(s.target(), add_special_tokens=False)["input_ids"] + [T.im_end_id(tok)]
        if len(prompt) > args.max_prompt or len(target) > args.max_resp:
            return None
        return {"eid": s.eid, "prompt": prompt, "target": target, "prefill": s.prefill, "ops": s.ops,
                "prompt_text": s.prompt_text(tok), "target_text": s.target()}

    train, skipped = [], 0
    for p in range(args.passes):
        start = (p * n_syn) % max(1, len(syn_order))
        for eid in sp["train"] + (syn_order + syn_order)[start:start + n_syn]:
            if p == 0 and not args.augment_first_pass:
                rec, g, meta = recs[eid], golds[eid], {"sections": ["dialogue", "note"], "known": [], "ops": []}
            else:
                rec, g, meta = A.augment(recs[eid], golds[eid], rng, pools)
            s = D.Sample(eid, rec, g, rng.random() < args.p_prefill, meta["sections"], meta["known"], meta["ops"])
            x = encode(s)
            if x is None:
                skipped += 1
                continue
            train.append(x)
    val = [x for eid in sp["silver_val"] for pf in (False, True)
           if (x := encode(D.Sample(eid, recs[eid], golds[eid], pf, ["dialogue", "note"]))) is not None]
    rng.shuffle(train)
    print(f"[sft] {len(train)} train samples ({skipped} over the length limits), {len(val)} silver-val samples")
    return train, val


def loss_of(model, x: dict, device) -> tuple:
    import torch
    ids = torch.tensor([x["prompt"] + x["target"]], device=device)
    lp = T.response_logprobs(model, ids, len(x["prompt"]))
    return -lp.mean(), len(x["target"])


def evaluate(model, val: list[dict], device) -> float:
    import torch
    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for x in val:
            l, k = loss_of(model, x, device)
            tot += l.item() * k
            n += k
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(T.BASE_MODEL))
    ap.add_argument("--out", default=str(T.CKPT / "sft"))
    ap.add_argument("--merged", default=str(T.CKPT / "sft_merged"))
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--augment-first-pass", action="store_true", help="also augment pass 1 (default: original text)")
    ap.add_argument("--no-synth", dest="synth", action="store_false")
    ap.add_argument("--synth-share", type=float, default=0.4,
                    help="share of synthetic samples per pass (records rotate across passes); < 0: every synthetic "
                         "record in every pass")
    ap.add_argument("--p-prefill", type=float, default=0.5)
    ap.add_argument("--max-prompt", type=int, default=6144)
    ap.add_argument("--max-resp", type=int, default=1024)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=8, help="samples per optimizer step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=30)
    ap.add_argument("--tb-port", type=int, default=6006,
                    help="serve TensorBoard on 0.0.0.0:PORT for the parent of --out (0: only write the event files)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--export", default=None, metavar="JSONL",
                    help="write the train and silver-val samples (text and token counts) and exit, like --dry-run")
    args = ap.parse_args()
    T.setup_env()
    tok = T.load_tokenizer(args.base)
    train, val = build_samples(args, tok)
    lens = sorted(len(x["prompt"]) + len(x["target"]) for x in train)
    print(f"[sft] tokens/sample mean {sum(lens) / len(lens):.0f} max {lens[-1]}; "
          f"target tokens {sum(len(x['target']) for x in train)}; prefill share "
          f"{sum(x['prefill'] for x in train) / len(train):.2f}")
    if args.export:
        rows = [{"split": split, "eid": x["eid"], "prefill": x["prefill"], "ops": x["ops"],
                 "prompt_tokens": len(x["prompt"]), "target_tokens": len(x["target"]),
                 "prompt_text": x["prompt_text"], "target_text": x["target_text"]}
                for split, xs in (("train", train), ("silver_val", val)) for x in xs]
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"[sft] wrote {len(rows)} samples -> {args.export}")
    if args.dry_run or args.export:
        return

    import torch
    T.check_free_vram(args.min_free_gb)
    torch.manual_seed(args.seed)
    device = "cuda"
    model = T.add_lora(T.load_lm(args.base, device=device), args.r, args.alpha)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    steps = math.ceil(len(train) / args.accum)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    out = Path(args.out)
    log = T.JsonlLog(out / "train_log.jsonl", tb_dir=out / "tb")
    T.start_tensorboard(out.parent, args.tb_port)
    print(f"[sft] silver-val loss before: {(v0 := evaluate(model, val, device)):.4f}")
    log.write({"step": 0, "val_loss": v0})
    t0, step = time.time(), 0
    model.train()
    for i in range(0, len(train), args.accum):
        batch = train[i:i + args.accum]
        n_tok = sum(len(x["target"]) for x in batch)
        tot = 0.0
        for x in batch:   # token-weighted mean over the batch
            l, k = loss_of(model, x, device)
            (l * k / n_tok).backward()
            tot += l.item() * k
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        row = {"step": step, "loss": tot / n_tok, "grad_norm": gn, "lr": sched.get_last_lr()[0],
               "elapsed_s": round(time.time() - t0, 1), "peak_vram_gb": round(T.peak_vram_gb(), 2)}
        if step % 10 == 0 or step == steps:
            print(f"[sft] step {step}/{steps} loss {row['loss']:.4f} gn {gn:.2f} {row['elapsed_s']}s "
                  f"peak {row['peak_vram_gb']} GiB", flush=True)
        log.write(row)
    v1 = evaluate(model, val, device)
    print(f"[sft] silver-val loss after: {v1:.4f} ({time.time() - t0:.0f}s)")
    log.write({"step": step, "val_loss": v1})
    model.save_pretrained(str(out))
    T.copy_tokenizer(out)
    (out / "sft_args.json").write_text(json.dumps(vars(args), indent=1))
    del model, opt
    torch.cuda.empty_cache()
    T.merge_and_save(args.base, out, Path(args.merged))
    print(f"[sft] adapter -> {out}, merged -> {args.merged}")


if __name__ == "__main__":
    main()
