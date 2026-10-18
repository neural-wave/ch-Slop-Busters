# NeuraWave

Patient-data anonymization for the Neural Wave / Bosch hackathon task (`task/`). Inputs are the customer files in
`data/`; everything runs on-prem.

- How to run the anonymizer and the local evaluator: [../term-generalizer/README.md](../term-generalizer/README.md) (run its commands from this directory)
- Deliverable within the task's 4 GB VRAM limit: `generalize` mode on the `VRAM_GB=4` vLLM server (or Ollama's
  quantized `qwen3.5:2b`), see "VRAM budget" in the term-generalizer README. Output: `runs/<run>/anonymized.jsonl`, same schema
  as the input with `icd10_codes` unchanged.
- Current results: [runs/summary.md](runs/summary.md)
