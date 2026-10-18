# SFT dataset: status, schema, and two fixes before training

## Context
The user asked whether the SFT dataset is ready and what its schema is. Inputs are ready: 126 train encounters
(silver labels, `../term-generalizer/gold/train_gold.jsonl`), 40 synthetic injury encounters (`data/synth_external_cause.jsonl` +
`../term-generalizer/gold/synth_gold.jsonl`), 14 silver-val encounters (`rl/splits.json`). The SFT samples themselves are not a
file: `rl/sft.py: build_samples()` builds them in memory at train time (3 passes x 166 encounters, fresh augmentation
per pass, 50% dx-prefilled). Printing one sample showed a label bug that should be fixed before SFT.

## Schema (one in-memory sample)
```
{"eid": "D2N070", "prefill": true, "ops": ["crossover", "despace"],
 "prompt": [token ids],   # chat template: system (spec.SYSTEM_PROMPT) + user (DIALOGUE / PHYSICIAN NOTE sections,
                          # optional carry-over identifier list) + "<think>\n\n</think>\n\n" + dx prefix if prefill
 "target": [token ids]}   # silver JSON continuation + <|im_end|>; loss only on these tokens
```
Target JSON (`data.silver_target`, serialized by `spec.dump_output`):
`{"dx": [{"code","desc"}], "evidence": [{"code","spans":[...]}], "entities": [{"text","ctx"?,"type","reason"}],
"keep": [{"text","reason"}]}`. With prefill, `{"dx": [...], "evidence": [` is in the prompt and the target starts
at the first evidence item. Measured: prompts mean ~2.6K / max 5.1K tokens, targets mean ~400 / max 816.

## Problems found in the printed sample
1. Stale reasons after PII crossover: `at home working` was swapped to `collects antique clocks` but the entity keeps
   the old annotator comment ("works from home on weekdays"). Swapped OTHER_QUASI/OCCUPATION text also reads oddly in
   context ("i'm collects antique clocks").
2. Surnames in `augment.COMMON` (e.g. `walker`, `baker`) are never swapped, so the real silver surname stays in the
   augmented text (`Mr. Bernard Walker`). Acceptable (still labelled PII) but worth knowing.

## Changes
- `rl/augment.py` `Crossover.apply`: when an entry's text changed and its category is OTHER_QUASI or OCCUPATION,
  replace `comment` with a neutral reason ("personal detail" / "person's job"); keep comments for names/ages/dates
  (they stay true). Restrict OTHER_QUASI swaps to strings of similar shape (both noun phrases <= 4 words), otherwise
  leave the original string unchanged.
- `rl/sft.py`: add `--export PATH` that writes the built samples as JSONL for inspection, one row per sample:
  `{"eid","prefill","ops","prompt_text","target_text","prompt_tokens","target_tokens","split":"train|silver_val"}`
  (no GPU, exits before model loading, like `--dry-run`).

## Verification
- `python3 -m unittest discover -s rl/tests` (augmentation invariance must still pass).
- `.venv-train/bin/python rl/sft.py --export runs/sft_dataset.jsonl` then spot-check 20 rows: reasons match the
  entity text, swapped quasi strings read naturally, every entity text occurs in `prompt_text`.
- No training is run (standing instruction).
