#!/usr/bin/env bash
# Reproduce every number in outputs/eval/REPORT.md from the current code (development machine: ~45 min).
# Requires: uv sync --all-extras, cached HF models, Ollama with qwen3.5:2b-q4_K_M and bge-m3.
set -euo pipefail
cd "$(dirname "$0")/.."
NW="uv run neural-wave"
EVAL=resources/data/eval_patient_doctor_encounters.jsonl

# 1. Submission file (balanced, rules + deid_roberta + GLiNER, auto GPU placement under the 4 GB budget)
$NW anonymize -i "$EVAL" -o outputs/anonymized_eval_patient_doctor_encounters.jsonl --preset balanced --detectors rules,deid,gliner
$NW evaluate --original "$EVAL" --anonymized outputs/anonymized_eval_patient_doctor_encounters.jsonl --gold outputs/gold/eval \
  --decisions outputs/anonymized_eval_patient_doctor_encounters.decisions.jsonl \
  --summary outputs/anonymized_eval_patient_doctor_encounters.summary.json --icd embed --name submission_eval
$NW evaluate --original "$EVAL" --anonymized outputs/anonymized_eval_patient_doctor_encounters.jsonl --gold outputs/gold/eval \
  --summary outputs/anonymized_eval_patient_doctor_encounters.summary.json --icd llm --name submission_eval_icdllm

# 2. Privacy/utility curve: presets x codes/no-codes, rules only and with models (eval and dev)
$NW sweep --split eval --icd embed --detectors rules,deid,gliner
$NW sweep --split dev --icd embed --detectors rules,deid,gliner

# 3. Detector ablation (balanced, eval and dev)
for split in eval dev; do
  for det in rules,deid rules,gliner rules,llm rules,deid,gliner,llm; do
    $NW run --split $split --preset balanced --detectors $det --icd embed --name ablation_${split}_${det//,/+}
  done
done

# 4. Adversarial attacks on the submission and on the rules-only output
$NW attack --original "$EVAL" --anonymized outputs/anonymized_eval_patient_doctor_encounters.jsonl --gold outputs/gold/eval --name attack_submission
$NW attack --original "$EVAL" --anonymized outputs/runs/sweep_eval_balanced/anonymized.jsonl --gold outputs/gold/eval --name attack_balanced_rules --skip-ner

# 5. Robustness to unseen formats (rules only and with models)
$NW robustness --build --preset balanced
$NW robustness --preset balanced --detectors rules,deid,gliner

# 6. Tests
uv run pytest -q
