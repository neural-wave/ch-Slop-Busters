# RL training plan: PPO (actor + critic) on Qwen3.5-2B for clinical de-identification

Target: Bosch / Neural Wave task (`task/NeuralWaveTask_Anonymization of_patient_data_public.pdf`).
Hardware: 1x L40S (46 GB). Budgets: **pilot ≤ 1 h GPU time** on `data/`, then **4-5 h** on the larger datasets.

This version merges the first plan, its revision (`plans/why-2k-char-chunk-tranquil-firefly.md`) and the review
(`train/ppo-training-plan.md`). Both of those are now superseded; §13 lists what changed.

## 0. Input and output

### The task

| | |
|---|---|
| **Input** | One JSON line per encounter: `encounter_id`, `dataset` (source: aci / virtassist / virtscribe), `dialogue` (doctor-patient transcript with `[doctor]`/`[patient]` tags), `note` (physician note, SOAP-style), `icd10_codes` (list of `{code, description}`). Every field is available at inference. |
| **Output** | The same records in the same schema: `dialogue` and `note` anonymized, `icd10_codes` byte-identical. Deliverables: the anonymized eval set as `.jsonl`, plus a repo with source code and README. |
| **Scored on** (equal weights) | Privacy: share of private information removed. Utility: ICD-10 prediction F1 and medical-concept recall on the anonymized text. Originality. Scalability. The organizers compute these on a hidden set with the same structure. |
| **Hard constraint** | Anonymization runs on-prem in ≤ 4 GB VRAM. Development and training may use more. |
| **Bonus** | Adversarial attacks, explanations, efficiency, no ICD codes at inference, robustness to other formats. |

### Our system

```
record ─► prompt builder ─► Qwen3.5-2B + merged LoRA (fp8, vLLM, ≤ 4 GB) ─► JSON decisions ─► renderer (code) ─► anonymized record
          dialogue + note,              ▲                                                      │
          no codes in the prompt        └ assistant prefix: dx prefilled from icd10_codes      └► explanation log, 1 entry per decision
```

| stage | input | output |
|---|---|---|
| Prompt builder | One record, or a ≤ 15K-token window plus a known-identifiers list | System prompt (label spec) + user turn (dialogue + note). **ICD codes are never in the prompt.** |
| Policy (the model being trained) | The prompt, plus an assistant prefix with `dx` filled from `icd10_codes` when codes are given | One JSON object: `dx` → `entities` → `keep` (§2) |
| Renderer (deterministic: `../term-generalizer/anonymize.apply_entities` + `../term-generalizer/generalize`) | Original text + regex spans + the JSON decisions | Anonymized `dialogue`/`note` and the explanation log. `dx` is dropped; `icd10_codes` is copied |
| SFT | Prompt → silver target JSON | LoRA adapter (arm 2 in §8, and the fallback submission) |
| PPO | Prompts; silver labels feed the reward only | LoRA adapter (arm 3) |
| Export | LoRA + `models/Qwen3.5-2B` | Merged bf16 text-only checkpoint, served by `../term-generalizer/serve_vllm.sh` with `VRAM_GB` set (online fp8) |
| Evaluator (`../term-generalizer/evaluate.py`) | Original and anonymized `.jsonl` + reference labels | PII removal (mention, entity, name token, by tier), concept recall, ICD term retention, ICD-10 F1 |

## 1. Facts that shape the plan (checked against the files)

| fact | consequence |
|---|---|
| Privacy, utility, originality and scalability are weighted equally, and the first two trade off | The reward charges both under- and over-redaction, per decision (§4). It is a proxy: equal rubric weights do not tell us how the organizers combine metrics. |
| 4 GB VRAM applies only at inference. Measured on the untrained model (`logs/vllm_4gb_test.log`): fp8 online quantization, `--max-model-len 16384`, eager mode, weights 2.36 GiB, KV cache 0.55 GiB = 38,967 tokens (2.38 × 16K). nvidia-smi peak at `VRAM_GB=4`: **4,000 MiB** | Export = merge the LoRA, then serve through the same vLLM fp8 path. Target an nvidia-smi peak ≤ 3.8 GiB for margin, and measure it again on the trained model at the final context length. GGUF is dropped: it is not needed, and llama.cpp support for the hybrid architecture is unverified. |
| `data/patient_doctor_encounters.jsonl`: 140 encounters, **no PII labels**, mean 2,219 / max 4,530 tokens | Silver labels are needed (§3). A whole encounter fits in one pass, so the harness's 2,000-character chunks are dropped. |
| `data/eval_…jsonl`: 67 encounters, reference labels in `../term-generalizer/gold/eval_gold.jsonl` (484 PII strings, 4,334 concepts, 173 ambiguous terms), annotated by LLM agents, not by the organizers | Never used for the reward or SFT. Split into dev and test (§9). |
| Dialogues entirely lowercase: 138/140 train, 67/67 eval, with spaced punctuation (`hi , andrew .`). Notes are cased | Capitalization heuristics only work on notes. All matching is case-insensitive, and training augments casing (§5). |
| ICD codes: train 285 (2.04 per encounter), eval 153 (2.28). Only **46%** of eval code occurrences appear in train (61% at 3 characters) | Learning a closed set of codes will not transfer, so `dx` is open-vocabulary (§2). |
| **External-cause (V-Y) codes: 0 of 285 in train, 24 of 153 in eval** (e.g. `W18.49XA`, `Y93.53` golf) | Their only evidence (activity, place, mechanism) looks like a quasi-identifier. Needs a spec rule and synthetic examples (§3). |
| Current best at deployment precision (`runs/vllm4gb-qwen3.5-2b_generalize`): PII removed 79.8% (direct 87.5, quasi 67.1), name tokens 87.0, concept recall 94.9 (ICD-relevant 94.4), ICD term retention 98.8, 17.6 s/encounter | This is arm 1 of the comparison (§8). The old 75.2% zero-shot `extract` number is no longer the bar. |
| `models/Qwen3.5-2B`: post-trained with a chat template. Hybrid Gated-DeltaNet linear attention plus full attention every 4th layer, with a vision tower | Training needs `flash-linear-attention` and `causal-conv1d`. Load the text-only model. |
| One vLLM engine currently holds ~23 GB of the GPU (checked 2026-10-17) | Stop every `vllm serve` before training. Training scripts check free VRAM at startup. |

**Starting checkpoint:** the post-trained `Qwen3.5-2B`, because it follows JSON and has measured baselines. Switch to
`Qwen3.5-2B-Base` only if PPO output diversity collapses (all n samples identical). SFT is mandatory in both cases.

## 2. Policy: the model decides, code rewrites

For each encounter the actor emits one JSON object, and deterministic code applies it. The model never rewrites text,
so medical wording cannot be paraphrased away, and `icd10_codes` stay unchanged by construction.

```json
{"dx": [{"code": "E11.9", "desc": "type 2 diabetes mellitus without complications"}],
 "evidence": [{"code": "E11.9", "spans": ["type 2 diabetes", "hemoglobin a1c of 8.3"]}],
 "entities": [
   {"text": "martha", "type": "PATIENT_NAME", "reason": "patient first name"},
   {"text": "50-year-old", "type": "AGE", "reason": "exact age"},
   {"text": "mount baker", "type": "LOCATION", "reason": "named place visited"},
   {"text": "parkinson", "ctx": "mr. parkinson", "type": "OTHER_PERSON_NAME", "reason": "surname of a relative"}],
 "keep": [{"text": "parkinson's disease", "reason": "diagnosis eponym"}]}
```

- **Order `dx` → `evidence` → `entities` → `keep`.** The model names the diagnoses and their evidence first, then
  redacts around them. Evidence is a separate list (one item per code) rather than nested in each `dx` item, because
  an assistant prefix can only fix a prefix of the output: with nesting, the second `dx` item could not be prefilled.
- **ICD codes stay out of the prompt.** When codes are available (the main task), `dx` is **prefilled** as an
  assistant prefix from `icd10_codes` (`{"dx": [...], "evidence": [`), and the model writes the evidence and
  continues. When codes are unavailable (the bonus), the model writes `dx` itself. Training mixes both: 50% of
  rollouts are prefilled. The prefix is part of the prompt, so it gets no reward and no loss.
- **Types** are the 11 reference categories (`PATIENT_NAME, CLINICIAN_NAME, OTHER_PERSON_NAME, DATE, CONTACT, ID,
  AGE, LOCATION, ORGANIZATION, OCCUPATION, OTHER_QUASI`). The type sets the rendering: AGE and DATE become ranges
  (`generalize.render`), quasi spans that are ICD evidence become category words (§3), and everything else becomes
  `[TYPE]`.
- **Occurrences (`ctx`).** Without `ctx`, every whole-word, case-insensitive occurrence in the **whole encounter**
  (both fields) is replaced. This is encounter-level linking: a name found once is removed everywhere. With `ctx` (a
  longer verbatim string containing `text`), only occurrences inside a `ctx` match are replaced. `ctx` is required
  when `text` is also a silver medical concept or an ICD description word (eponym collisions). Without it, the span
  is dropped by `is_protected`, as today.
- **Span limits.** The renderer rejects a span that is not found verbatim, is longer than 6 words, crosses a line,
  or is a speaker tag, and the reward charges for it. This blocks whole-chunk deletion.
- **The regex pre-pass** (`generalize.find_spans`) runs first. The model can veto a regex span by listing it in
  `keep` (a clinical date, or a dose that looks like a year), and the veto is scored like any other decision.
- **Reasons** are at most 8 words and are not rewarded for content (§4). The logged explanation combines the rule
  that fired (`generalize.explain`), the type, and the model's reason.
- Thinking is off. Sampling uses T=1.0 and top_p=1.0 with penalties off (`presence_penalty` 0, as in the harness).

### Input length

- **One pass for up to 16,384 prompt tokens** (system prompt ~400 + content). Every current encounter takes one pass
  (the maximum is 4,530 tokens).
- **Beyond 16K:** windows of 15K content tokens with a 512-token overlap, cut on line boundaries. Each window carries
  a known-identifiers list of up to 64 entries (~400 tokens), and every string found is applied to the whole document.
- **Max response 2,560 tokens.** If the JSON is unclosed, salvage the finished items as the harness does, then split
  the input in half and re-run. Measure every prompt with the real tokenizer, and fail loudly instead of truncating.
- **Inference settings:** `--max-model-len 18944`, fp8, chunked prefill at 2,048 tokens. Only 16,384 has been
  measured so far, so re-measure KV capacity at this length (§7, gate 2).

## 3. Silver labels and label spec (CPU/API time, before the GPU clock)

Annotate the 140 train encounters with the eval-gold pipeline: an annotator pass, then an independent reviewer pass
(larger or cloud models are allowed during development). Fields: `pii`, `medical_concepts` (+ `icd_relevant`),
`ambiguous_terms`, plus **evidence links** from each `icd10_codes` entry to the concept spans that support it (needed
for `dx.evidence`). Add a `--split train` flag to `build_gold.py` to validate them and write
`../term-generalizer/gold/train_gold.jsonl`, then run `audit_gold.py`.

These spec additions decide what SFT and RL can learn:

- **Context for external-cause codes and Z codes is `icd_relevant`, never `OTHER_QUASI`.** "i was golfing" is the
  only evidence for `Y93.53`. Removing it loses one of about 2 codes, roughly a 50% F1 hit on that record. Such place
  spans are rendered as a **category word** from a closed list built from the Y92/Y93/Z57 titles ("mount baker" →
  "a mountain"), not as `[LOCATION]`, and activity words stay verbatim. Because train has no V-Y codes, add **~40
  synthetic injury encounters** (falls, sports, vehicles, workplace) generated by a larger model, with codes set by
  construction and PII inserted programmatically.
- **Case:** the annotator sees the lowercased dialogue, and all matching (`find_spans`, gold, reward) is
  case-insensitive.
- **Eponym/surname collisions:** when one encounter uses a term in both roles, label it in `ambiguous_terms` with a
  `ctx`.
- **Audit sample:** check about 60 labels by hand in the categories most likely to differ from the organizers'
  interpretation: OCCUPATION, OTHER_QUASI, eponyms, activities, places.
- **Resource:** the ICD-10-CM code table (public CMS file, used offline) for the code-validity check in the reward.
- **New datasets (large run):** pre-annotate with the pilot policy, then have a strong model review and correct the
  labels. Labeling, not GPU time, is the bottleneck for new data.

Hold out 14 train encounters (10%, stratified by source) as **silver-val** to check overfitting.

## 4. Reward: one reward per decision, on its closing token

Reuse the evaluator's matching (`common.norm`, `common.count_mentions`, `evaluate.name_tokens`) so the reward
optimizes what is measured. All quantities compare the rendered encounter against silver labels:

- `P` = share of weighted PII mentions removed (direct tier w=2, quasi w=1). Name tokens count separately, so
  `[PERSON] collins` still leaks `collins`.
- `U` = share of weighted medical-concept mentions kept (`icd_relevant` w=2, eponyms included).
- A decision's ΔP and ΔU are the marginal changes when its span is applied. With no silver PII, ΔP = 0; with no
  concepts, ΔU = 0.

| decision | reward, placed on the token that closes the item |
|---|---|
| `entities[i]`, applied | `w_P·ΔP + w_U·ΔU − λ_fp·(#removed words not covered by silver PII)`; **−0.5** if it removes the last surviving evidence of a gold code (stands in for ICD F1, which is too slow for the loop); +0.02 if the type matches silver |
| `entities[i]`, rejected (§2 span limits, missing `ctx`) | −0.1 |
| `keep[j]` | +0.02 for a silver non-PII ambiguous term or a correctly vetoed regex span; −0.1 if it is silver PII |
| `]` closing the entity list | `−w_miss·(1 − P)`: charges omissions at the point where the model stops |
| `dx[k]` (free mode only) | Content-word F1 of `desc` against the best-matching gold description, plus code credit (exact 1.0 / 3-char 0.5 / chapter 0.1; **−0.2** if the code is not in the ICD-10-CM table), weighted by 1/√(train frequency); −0.3 per false-positive `dx` |
| `evidence[k]` (both modes) | +0.05 per span that is verbatim in the text, matches a silver concept, and survives rendering |
| whole output | −1 if the JSON does not parse (nothing is applied) |

Start values: `w_P = w_U = w_miss = 1`, `λ_fp = 0.03`. PPO with a token-level critic (GAE, γ=1, λ=0.95) does the
credit assignment. Entity rewards depend only on silver PII and concept labels, **not on predicted codes**, so a
wrong or unseen diagnosis cannot make redacting real evidence pay.

**Direction of the knobs** (this fixes two sign errors in the previous version):

| to favor | change | effect |
|---|---|---|
| privacy | raise `w_P` or `w_miss` | each removed PII mention pays more; each omission costs more |
| utility | raise `w_U` or `λ_fp` | destroying concepts and redacting non-PII words cost more |

If utility drops during PPO, **raise** `λ_fp` or `w_U`. If privacy stalls, raise `w_miss`.

**Rarity term (large run, stage B only):** a non-medical word with corpus IDF in the rarest 2% that lies within ±10
words of a silver direct identifier is exempt from `λ_fp` when removed, and costs −0.01 at `]` if it survives. This
targets the brief's "rare words" attack surface without forcing removals elsewhere.

**Not rewarded:** reason wording, which would teach plausible text rather than faithful reasons. The in-loop attacker
is removed: a frozen 2B model guessing deleted names from one chunk recovers ~0% from the first step, so the penalty
never fires. Attacks move offline (§11).

**Reward tests (write these before training, using eval gold as if it were silver).** Cover four cells (PII present
or absent × concepts present or absent), each with these candidate outputs: identity, silver labels, redact every
word, selective over-redaction (every non-PII, non-medical word as a 1-word span), whole-line deletion, a list of 200
random words, and invalid JSON.
- The silver output has the highest return in every cell.
- Identity returns 0 on PII-free inputs (the best possible there) and `−w_miss` when PII is present.
- Selective over-redaction and whole-line deletion return ≤ −1. Invalid JSON returns −1.
- An invented code is penalized, and a prefilled `dx` scores 0.
- Direction: take output A (one more PII mention removed) and output B (one more concept kept). Raising `w_P` must
  increase R(A) − R(B), and raising `λ_fp` or `w_U` must decrease it.
- Entity rewards sum to S(final) − S(original) regardless of order, except where spans overlap.

## 5. Training data

**Augmentations** are deterministic and drawn per sample, so each visit to an encounter looks different:

| augmentation | probability | rule |
|---|---|---|
| PII crossover | 1.0 | Replace the encounter's PII with PII from other encounters' silver labels (50%) or a fake-identity list (50%), following the ICD-safe rules below |
| True-casing | 0.5 | Capitalize sentence starts, "I", and silver names and places |
| De-spaced punctuation | 0.5 | " , " → ", " and " . " → ". " |
| Speaker-tag style | 0.5 | `[doctor]` → `Doctor:`, `Dr. <surrogate>:`, `D:`, `Clinician:`, `PHYSICIAN -`, or no tags |
| Field layout | 0.3 | Note first, dialogue only, or note only |
| Sub-window + carry-over | 0.3 | A random window of 512 tokens up to the full sample, with a carry-over list from preceding silver labels (50% of entries dropped) |

**ICD-safe PII crossover rules:**
- **Names:** a consistent per-token map within the encounter, sex-consistent (sex taken from pronouns and
  honorifics), skipping names that are common words or medical terms ("may", "bill", "grace").
- **Ages:** stay in the same `generalize.age_band`, within ±4 years.
- **Dates:** one per-encounter shift of ±1-3 years plus up to 7 days. Relative durations are never touched.
- **Locations, organizations, contacts and IDs** are swapped freely within their category.
- **Occupations and OTHER_QUASI** are swapped only if they have no evidence link to a code.
- **Medical concepts, eponyms and ambiguous terms** never change.

Multi-visit concatenation keeps one surrogate identity and takes the union of codes. Splicing segments between
encounters is not done, because the text would stop matching its codes.

**Sampling:**
- About **70% of prompts are PII-dense.** PII-free windows (~30%) teach the model to leave clean text alone.
- **Pilot:** `data/` train (126 encounters after the holdout) plus the ~40 external-cause synthetics, with a fresh
  augmentation on every visit. Prompts ≤ 6K tokens.
- **Large run:** ~60% real silver data (the new datasets plus `data/`), ~25% synthetic styles (discharge summaries,
  ER triage, radiology and lab reports, referral letters, telephone notes, 8-16K multi-visit records), and ~15% hard
  negatives (eponym used as a patient name in an encounter without that disease, external-cause injuries, clinical
  dates and ages such as gestational age, "7/10 pain", guideline ages). Long samples make up 15%, sub-windows 30%.
- Eval data is never augmented for training.

**Invariance tests** (unit tests, no model, every train encounter × 20 seeds):
- Every silver concept, ambiguous term and evidence span is present with the same count.
- `icd10_codes` is identical.
- No original PII string survives, except tokens shared with a concept.
- Every new PII string is present with its recorded count.
- Sex and age band are unchanged.
- Every silver span can still be located after the style transforms.

## 6. Training stack and hyperparameters

- **Framework:** a custom PPO loop (`rl/ppo.py`): GAE with a separate critic, in-process vLLM rollouts with the actor
  LoRA loaded per step, and a reward function that writes each decision's reward at its closing token. verl 0.9.1 was
  dropped because it pins transformers < 5.11, vllm 0.24, torch 2.11 and a CUDA 13 flash-attn wheel, and the node's
  driver is 545 (decision of 2026-10-17).
- **Initialization:** the SFT LoRA is merged into the base before PPO. The actor gets a fresh PPO LoRA on the merged
  model. The reference is the merged SFT model (PPO adapter disabled). The critic starts from the merged SFT model
  with its own LoRA and a scalar value head.
- **No second implementation.** If the smoke test (§7) fails on the hybrid architecture, ship the SFT model (arm 2)
  rather than starting a second implementation.
- **Environment:** `.venv-train` layers `peft accelerate flash-linear-attention` over `.venv-vllm` (torch 2.13,
  transformers 5.17, vllm 0.31) through a `.pth` file, so the serving venv stays untouched (`rl/README.md`).
  `causal-conv1d` is not installed: transformers falls back to its torch convolution, which is cheap at kernel size 4.
- **Memory:** compute log-probs over response tokens only, in chunks (full logits at 18.9K × 248K vocab would be about
  18 GB in fp32); gradient checkpointing; micro-batches sized by token count; vLLM colocated with sleep mode. LoRA is
  needed because the actor, critic and vLLM share 46 GB, not because of the 4 GB rule.

| | pilot (`data/`) | large run |
|---|---|---|
| max prompt / response | 6,144 / 1,024 | 16,384 / 2,560 |
| rollouts per step | 16 prompts × n=4 = 64 (~185K tokens) | 24 × n=4 = 96 (~480K tokens) |
| est. step time / steps | 60-80 s / ~28 | 120-150 s / ~90-110 |
| SFT | LoRA r=32, lr 1e-4, 3 augmented passes (~380 samples). Targets: `dx` (prefilled or free) + silver evidence, entities and keeps, with the label comment as the reason | Refresh from the pilot LoRA on the new silver data |
| actor | LoRA r=32, α=64, all linear projections (incl. linear-attention in/out), lr 2e-5; raise to 3e-5 if KL < 0.5 after 10 steps | r=64, α=128, lr 1e-5 → cosine |
| critic | LoRA r=32 + value head, lr 5e-5, pre-warmed on SFT rollouts (§8, P2) | r=64, lr 5e-5, 8 warm-up steps |
| PPO | 1 epoch, mini-batch 16, clip 0.2, value clip 0.2 | 2 epochs, mini-batch 32 |
| KL to reference | fixed, 0.01 | adaptive, target 6 |
| validation / checkpoints | dev split every 10 steps / every 10 steps | every 20 steps / every 25 steps |

Step times are estimates until the smoke test replaces them.

## 7. Gates before the GPU clock starts

1. **Reward and spec fixed.** Gate: the §4 tests pass (four cells, knob directions, sum of entity rewards), and the
   external-cause rule is in the spec.
2. **Deployment proven.** Merge a dummy LoRA into the text-only checkpoint and serve it with
   `VRAM_GB=3.8 ../term-generalizer/serve_vllm.sh` at `--max-model-len 18944`. Run the 5 longest and 5 most PII-dense encounters
   and one 16K synthetic record. Gate: valid output, nvidia-smi peak ≤ 3.8 GiB, runtime settings logged.
3. **Labels audited, evaluation frozen.** Gate: train silver labels built and audited, holdouts reserved (§9), and
   arm 1 measured on dev and test with per-category recall, encounters with a leaked direct identifier, and absolute
   ICD F1.
4. **Data, reward code, smoke test.** Build `rl/data.py` (records → prompts, augmentations, silver spans) and
   `rl/reward.py`, and audit token lengths. Smoke test: 2 PPO steps with 8 prompts, recording s/step, peak VRAM and
   the JSON-valid rate, and logging the decoded token at every non-zero reward. Gate: the step runs, and the batch is
   resized so a step takes ≤ 80 s.
5. Stop every `vllm serve` process (one engine held ~23 GB on 2026-10-17).

## 8. Pilot: 60 GPU minutes on `data/`

The run compares three arms on identical data, rules, rendering and deployment precision: **arm 1** = current model +
rules, **arm 2** = SFT + rules, **arm 3** = SFT + PPO + rules. RL earns its place only if arm 3 beats arm 2.

| clock | step | output |
|---|---|---|
| 0:00-0:04 | **P0.** Arm 1 with the new renderer (linking, `ctx`, category words), fp8, on dev | baseline row in `runs/summary.md` |
| 0:04-0:14 | **P1.** SFT (§6), then a dev eval at fp8 → arm 2 | `ckpt/sft`. **Send SFT output on the 67 eval encounters for the organizers' one mid-term evaluation** |
| 0:14-0:18 | **P2.** Critic warm-up on 256 SFT rollouts (critic only) | explained variance logged |
| 0:18-0:52 | **P3.** PPO, ~28 steps, dev validation every 10 steps | `ckpt/ppo_step*` |
| 0:52-1:00 | **P4.** Merge the best checkpoint, serve it at fp8 under 3.8 GiB, run test with and without `dx` prefill → arm 3 | `runs/pilot_ppo/`, `runs/pilot_ppo_noicd/` |

ICD F1 with the large offline coder runs for all arms after the clock.

**Monitor:** mean return and each part separately (ΔP, ΔU, `λ_fp` cost, miss charge, `dx`); JSON-valid rate;
entities per encounter vs. silver; rejected-span rate; response length; KL; critic explained variance; clip fraction.

**Abort and ship SFT if:**
- critic explained variance stays below 0.1 after 10 steps;
- JSON validity drops below 95%;
- entities per encounter exceed 1.5× silver (reward hacking);
- KL exceeds 20.

**Go/no-go for the 4-5 h run.** Measure everything on the test split with the exported fp8 model, using paired
encounter-level bootstrap 95% CIs:
- Arm 3 beats arm 2 on PII removal (the CI of the difference excludes 0), with concept recall down ≤ 0.5 points and
  ICD F1 down ≤ 1 point. If not, the large run is SFT on the new data, with PPO only to fix a specific failure.
- Against arm 1 (79.8 / direct 87.5 / quasi 67.1 / concept recall 94.9): PII removal ≥ 85%, direct ≥ 93%, concept
  recall ≥ 94%.
- JSON validity ≥ 99%, no truncation, nvidia-smi peak ≤ 3.8 GiB, and s/encounter no worse than arm 1.
- The organizers' mid-term feedback has arrived and has been compared with the local metrics, so the 4-5 h run does
  not optimize label definitions that differ from theirs.

## 9. Evaluation hygiene

- **Splits:** the 67 eval encounters split into **dev (30)** for checkpoint selection and **test (37)**, reported
  once per arm. Stratify by source and by V-Y presence, and never split an encounter. Silver-val is the 14 held-out
  train encounters.
- **The reference labels are a proxy:** two model passes share systematic errors. Compare runs rather than reading
  absolute numbers. Report per-category recall, the share of encounters where any direct identifier survives, and
  paired bootstrap CIs.
- **Absolute ICD F1** comes from the strongest offline coder (`qwen3.8:27b`; identity scores 25.1 at code level and
  46.6 at category level), not only retention against the original text.
- **Only certify the deployed model.** Every reported number comes from the exported fp8 model on the 4 GB server.
  Pin the runtime, quantization, context limit and batch settings in the README.

**Scalability tests** (deterministic, built from eval and its reference labels; first after the pilot, final in the
large run):

| test | construction | pass criterion vs. plain eval |
|---|---|---|
| Long documents | Concatenate eval encounters into multi-visit records with one surrogate identity: 8-16K (single pass) and 20-27K (windowed) | PII removal and concept recall within 2 points; no truncation; nvidia-smi peak ≤ 3.8 GiB at 16K |
| Style shift | Casing, punctuation and speaker-tag combinations **not** used in training | within 3 points |
| Unseen source | Per-source metrics; in the large run, leave-one-source-out (train without virtscribe) | within 3 points of the mean |
| Unseen codes | Split eval codes into seen (46%) and unseen in train, and report V-Y codes separately | ICD-relevant concept recall within 2 points of seen; ICD F1 retained ≥ 97% on both |

## 10. Large run: 4-5 h on the extended datasets

| clock | step |
|---|---|
| 0:00-0:15 | Baseline of the pilot checkpoint on dev and on held-out synthetic styles |
| 0:15-0:40 | SFT refresh on the new silver and synthetic data, starting from the pilot LoRA |
| 0:40-2:00 | PPO stage A: real data plus long samples. Validate every 20 steps |
| 2:00-3:50 | PPO stage B: full mix (+ synthetic styles + hard negatives) with the rarity term on |
| 3:50-4:30 | Top 3 checkpoints on test (with and without `dx` prefill) and the §9 scalability tests. Select by PII removal at equal or better concept recall and ICD F1 |
| 4:30-5:00 | Export the selected model, measure latency, RAM and VRAM, write the deliverable `anonymized.jsonl` and the explanation log |

A checkpoint every 25 steps means a crash costs at most about 25 minutes.

## 11. Bonus tasks and originality

| bonus | how the plan covers it | evidence to submit |
|---|---|---|
| Adversarial privacy attacks | Offline, with a larger model and encounter-level context, given everything the evaluator sees including ICD metadata: (1) residual extraction, (2) attribute inference against a prior-only baseline, (3) record linkage of anonymized records to a candidate pool built from combined clues. Never used in training | recovery rate per attack, before and after RL, against the prior-only baseline |
| Explainability | Every decision is logged with the rule that fired, the type and the model's reason. `keep` explains retained ambiguous terms. The log describes the transformation and makes no compliance claim | % of decisions explained (target 100%), plus a manual correctness check of 50 decisions |
| Efficiency | Tag instead of rewrite, one pass per encounter, fp8 2B model, regex pre-pass | s/encounter, peak RAM, nvidia-smi peak VRAM |
| No ICD codes at inference | The model writes `dx` itself in 50% of training rollouts; evaluated both ways | metrics with and without prefill, seen vs. unseen codes |
| Robustness | Augmentations and synthetic styles; held-out style combinations; leave-one-source-out | per-style and per-source metrics |

**Originality** (25% of the grade): encounter-level linking, ICD-safe PII crossover, category words for
external-cause context, the open-vocabulary dx-then-redact policy, and the linkage attack. Optional: deterministic
perturbation of lab values that are not code evidence, as in the organizers' HbA1c 8.3 → 8.1 example. It stays off by
default, because string-match concept recall counts a perturbed value as lost. Test it locally first.

## 12. Main risks

| risk | mitigation |
|---|---|
| vLLM LoRA does not support the Qwen3.5 hybrid layers | Smoke-test gate (§7); `--rollout-sync merged` reloads merged weights instead; the fallback is the SFT model |
| Silver-label noise gets optimized as truth | Reviewer pass, manual audit, KL to the reference, evaluation on the reference labels rather than silver, mid-term check |
| V-Y codes absent from train | Spec rule, synthetic injuries, category-word rendering, separate V-Y reporting |
| Reward hacking (redact everything, or emit nothing) | Precision term, span limits, four-cell tests, entities per encounter monitored |
| The critic is fragile | Pre-warmed on SFT rollouts, explained-variance abort, n=4 samples per prompt |
| The trained model exceeds 4 GB at 18.9K context | Gate 2 before the clock, re-measured after export |
| Overfitting 126 encounters in the pilot | Augmentation on every visit (~3.5 visits per encounter); compare silver-val against dev |
| GPU shared with the vLLM servers | Stop them; scripts check free VRAM at startup |

## 13. What changed from the previous version

| area | before | now | source |
|---|---|---|---|
| Input unit | 2,000-character chunks | Whole encounter up to 16K tokens, windows beyond | revision |
| ICD codes | In the prompt for 50% of samples | Never in the prompt. `dx` is prefilled as a prefix when codes are given, and generated otherwise | revision |
| Output | `entities` + `keep` | `dx` → `entities` (+ `ctx`) → `keep` | revision + review |
| Reward | Per-chunk harmonic mean, β stated in the wrong direction | Per-decision rewards, a precision term against selective over-redaction, span limits, knob directions corrected | revision + review |
| Labels | No external-cause rule | External-cause rule, category words, synthetic injuries, evidence links, audit | review |
| Training | SFT optional, custom PPO fallback | SFT mandatory, three-arm comparison, critic pre-warmed, n=4, lr 2e-5; PPO is a custom loop because of verl's version pins | review + 2026-10-17 |
| Deployment | Tested in the last 4 minutes, GGUF | Gate before the clock, fp8 vLLM only, nvidia-smi peak ≤ 3.8 GiB | review |
| Attacker | Frozen 2B in the loop | Rarity term in the loop, three offline attacks | review |
| Evaluation | Eval set used for both selection and reporting | Dev/test split, bootstrap CIs, absolute ICD F1, scalability tests | review + revision |
| Baseline | 75.2% (`extract`) | 79.8% (`generalize` at fp8) | `runs/summary.md` |

## 14. Open questions for the team

- [ ] Who owns `rl/reward.py` and its tests, and signs off on the start values (`λ_fp`, `w_miss`)?
- [ ] Who owns the critic warm-up and the explained-variance abort?
- [ ] Who audits the categories most likely to differ from the organizers' view (eponyms, activities, places,
      occupations), and who writes the ~40 external-cause synthetics?
- [ ] Is the fallback submission confirmed as SFT + rules (arm 2) if PPO, export or integration fails?
- [ ] Should the mid-term evaluation use SFT output (as planned), or wait for the PPO pilot?
