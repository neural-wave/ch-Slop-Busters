"""LoRA fine-tune of Qwen3.5-2B on teacher PHI labels (chunk -> JSON spans), then merge.

Resumable: re-running continues from the last checkpoint in models/qwen35-2b-deid-lora/.
"""
import glob
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("NW_BASE_MODEL", "Qwen/Qwen3.5-2B")
LORA_DIR = ROOT / "models/qwen35-2b-deid-lora"
MERGED_DIR = ROOT / "models/qwen35-2b-deid-merged"

tok = AutoTokenizer.from_pretrained(BASE)


def encode(row):
    prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
    completion = row["messages"][-1]["content"] + "<|im_end|>\n"
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(completion, add_special_tokens=False).input_ids
    return {"input_ids": p + c, "labels": [-100] * len(p) + c}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(batch), n), pad)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train():
    rows = [json.loads(line) for line in open(ROOT / "outputs/sft/train.jsonl")]
    data = [encode(r) for r in rows]
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model.config.use_cache = False
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(LORA_DIR), num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=8, weight_decay=0.0, bf16=True,
        logging_steps=5, save_steps=40, save_total_limit=2, report_to=[], dataloader_num_workers=2,
        remove_unused_columns=False, seed=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate)
    resume = bool(glob.glob(str(LORA_DIR / "checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(LORA_DIR / "final")
    json.dump(trainer.state.log_history, open(LORA_DIR / "log_history.json", "w"), indent=1)


def merge():
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, LORA_DIR / "final").merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("train", "all") and not (LORA_DIR / "final/adapter_model.safetensors").exists():
        train()
    if step in ("merge", "all") and not (MERGED_DIR / "config.json").exists():
        merge()
