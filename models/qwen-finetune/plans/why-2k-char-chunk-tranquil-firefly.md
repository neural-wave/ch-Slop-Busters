# Revision of TRAINING_PLAN.md: a scalable input unit, per-decision rewards, open-vocabulary ICD knowledge

## Context

Your feedback so far:
- 4 GB VRAM applies only at inference, and the model is exported in 8-bit.
- Chunk sizes from the harness should not be trusted.
- Every decision should get its own reward.
- ICD-10 knowledge is learned in training and never put in the prompt.
- **The design must scale:** longer inputs, unseen conversations, and ICD codes absent from training.

Measurements (read-only: tokenizer and JSON only, no model run) that the design has to answer:

| finding | numbers | risk for scaling |
|---|---|---|
| Encounter length | train mean 2,219 / max 4,530 tokens | Inputs longer than the trained range |
| PII per encounter | 6.4 unique strings (p90 14, max 28) | Output grows with document length and may be truncated |
| **All dialogues 100% lowercase**, `[doctor]/[patient]` tags, 2 of 3 sources with spaced punctuation (" , ") | aci / virtassist / virtscribe; notes are cased SOAP | Real conversations are cased with normal punctuation, so the model would overfit to one surface form |
| ICD long tail | 146 distinct train codes, 108 seen once. **Only 46% of eval code occurrences seen in train** (61% at 3-char) | A closed-set code classifier won't generalize and would bias the policy toward I10 and E11.9 (24% of train codes) |

## 1. Input unit: one pass up to 16K tokens, windows beyond that

**16K fits the 4 GB inference budget. This is already measured.** Your run `~/NeuraWave/logs/vllm_4gb_test.log`
(vLLM 0.31, `--quantization fp8`, `--max-model-len 16384`, `--enforce-eager`, `--max-num-batched-tokens 2048`,
`--gpu-memory-utilization 0.0759` = 3.38 GiB) reported:
- weights 2.36 GiB;
- weights + non-torch 2.71 GiB;
- peak activation 0.12 GiB;
- **KV cache 0.55 GiB = 38,967 tokens, "maximum concurrency for 16,384 tokens per request: 2.38x"**.

That is ~15 KB per token (6 full-attention layers × 2 KV heads, plus the fixed linear-attention state), so one
16K + 2.5K-token request needs ~0.28 GiB. **Caveat:** nvidia-smi counts the CUDA context on top of vLLM's budget.
The earlier 0.0861 setting showed 4,226 MiB in nvidia-smi. Size the budget so the **nvidia-smi peak is ≤ 3.8 GiB**,
and report that number.

**Design:**
- **Single pass up to 16,384 prompt tokens** (system ~400 + content ~15.5K). Every current encounter (max 4,530) is a
  single pass.
- **Beyond 16K:** windows of 15K content tokens with a 512-token overlap, cut on line boundaries. Each window carries
  a **known-identifiers list** of up to 64 entries (~400 tokens). Every found string is applied to the **whole**
  document.
- **Max response 2,560 tokens.** A 16K multi-visit record can hold ~60-80 unique identifiers plus ~10 diagnoses, with
  reasons capped at 8 words.
- **Truncation fallback:** if the JSON is unclosed, split the input in half and re-run.
- **vLLM inference settings:** `--max-model-len 18944`, fp8 weights, chunked prefill at 2,048 tokens. This fits the
  measured KV capacity with headroom.

**Train for 16K.** The memory fits, but quality at 16K only holds if training covers that length. Current data
reaches only 4.5K tokens.
- **Long samples (large run, 15% of prompts):** multi-visit records of 8K-16K tokens. Concatenate 2-6 training
  encounters under **one consistent surrogate patient identity**, and merge their silver labels and codes
  (deterministic, exact labels). Use only training encounters; eval stays for the §5 tests.
- **Partial documents (30% of prompts):** random sub-windows of 512 tokens up to the full sample. They start
  mid-conversation, with a carry-over list from the preceding silver labels (50% dropped at random).
- **The pilot stays ≤ 6K,** because `data/` has nothing longer and long samples cost ~6x per rollout. The 16K
  capability is a large-run deliverable.
- **Training memory for 18.9K-token sequences:**
  - compute log-probs **only over response tokens**, chunked (full-sequence logits would be 18.9K × 248K vocab ≈
    18 GB in fp32);
  - gradient checkpointing;
  - token-based micro-batches of ~20K tokens, so one long sample per micro-batch.

## 2. Decisions and per-decision rewards (unchanged from the last revision, schema extended in §3)

Each output item is a decision: a diagnosis, a PII span with its category (which sets the rendering), or a
deliberate keep. Each gets a reward on its closing token: the **marginal change in Privacy + Utility** when that span
is applied, measured against silver labels with `../term-generalizer/common.norm` and `../term-generalizer/common.count_mentions`. The `]`
that closes the entity list is charged the remaining unremoved PII share. Invalid JSON gives −1. The rewards
telescope to S(final) − S(original), and PPO with a token-level critic (GAE γ=1, λ=0.95) does the credit assignment.

**Scaling property:** entity rewards depend only on silver PII and medical-concept labels, **not on the predicted
codes**. A wrong or unseen diagnosis cannot make it pay to redact real evidence.

## 3. ICD knowledge, open-vocabulary so it carries over to unseen codes

What has to generalize is "find the diagnosis evidence and protect it", not "emit code X". The first section of the
output becomes:

```json
{"dx": [{"desc": "type 2 diabetes mellitus without complications", "code": "E11.9",
         "evidence": ["type 2 diabetes", "hemoglobin a1c of 8.3"]}],
 "entities": [...], "keep": [...]}
```

| part | reward (closing token of each `dx` item) | why it handles unseen codes |
|---|---|---|
| `desc` | content-word F1 against the best-matching gold `icd10_codes.description` | Free text, so it works for any code; it builds on the model's pretrained medical vocabulary |
| `code` | hierarchical credit: exact 1.0 / 3-char 0.5 / chapter letter 0.1. **−0.2 if the code is not in the ICD-10-CM table** (public CMS file, offline) | Partial credit gives a signal on codes never seen in training; the validity check stops invented codes |
| `evidence` | +0.2 per span that is verbatim in the text, matches a silver medical concept **and survives anonymization** | Ties ICD knowledge to the redaction policy in a code-agnostic way |
| balance | each gold code's reward is weighted by 1/√(train frequency); each false-positive `dx` costs −0.3 | Keeps the policy from collapsing onto I10 and E11.9 |

**At inference:**
- With no codes available, the model writes `dx` itself.
- With codes available, the `dx` section is **prefilled** from `icd10_codes` (desc + code). The model fills in
  evidence and continues.
- `dx` is dropped from the deliverable, and `icd10_codes` is copied through unchanged.

## 4. Generalization to unseen conversations (deterministic augmentations, no model needed)

Applied per sample, independently, so each pass over the 140 encounters looks different:

| augmentation | probability | rule |
|---|---|---|
| **PII crossover** (§4a) | 1.0 | Swap PII between encounters, ICD-safe |
| True-casing | 0.5 | Capitalize sentence starts, "I", and silver names/places. The evaluator matches case-insensitively, so labels stay valid |
| De-spaced punctuation | 0.5 | " , " → ", " and " . " → ". " |
| Speaker-tag style | 0.5 | `[doctor]` → one of `Doctor:`, `Dr. <surrogate>:`, `D:`, `Clinician:`, `PHYSICIAN -`, or no tags |
| Field layout | 0.3 | note-first, dialogue-only, or note-only |
| Sub-window + carry-over | 0.3 | Described in §1 |

The large run adds synthetic documents in other styles (discharge, ER, radiology, referral, multi-visit records), with
PII inserted programmatically so the labels are exact.

### 4a. PII crossover between encounters (ICD-invariant)

**Idea:** the clinical content of encounter A stays byte-identical, and its PII is replaced by PII harvested from other
encounters' silver labels (50%) or from an external fake-identity list (50%, for name and place diversity beyond the
data). The codes and medical text are unchanged, so the `dx` targets are unchanged. The model sees the same
content → code mapping under many identities, which teaches it that **PII is irrelevant to codes**.

**Per-category rules (what keeps ICD prediction intact):**

| category | crossover rule | ICD reason |
|---|---|---|
| PATIENT / CLINICIAN / OTHER_PERSON_NAME | Consistent **per-token** map within the encounter (martha→linda, collins→park), so full name, first name, "mrs. collins" and the possessive all change together. **Sex-consistent:** encounter sex comes from pronoun/honorific counts, and names are drawn from the same-sex pool | Sex-specific codes (O, N40, C61, N80…) |
| AGE | Swap only within the **same `generalize.age_band`** and ±4 years, rendered in the same surface form ("50-year-old" → "53-year-old") | Pediatric, adult and geriatric codes; age-based screening |
| DATE | Per-encounter shift by **whole years (±1-3) + ≤7 days**, applied to every absolute date incl. DOB (parsed with `generalize.DATE_NUMERIC`/`DATE_MONTH`) | Keeps intervals, season and age-at-visit; relative durations ("5 months ago") are never touched, because they set acute/chronic and initial/subsequent 7th characters |
| LOCATION / ORGANIZATION / CONTACT / ID | Free swap from the pool, same category | No code depends on them |
| OCCUPATION / OTHER_QUASI | Swap **only if the silver label has no evidence link to a code**; linked ones stay as they are | Occupational exposure (Z57), place of occurrence (Y92) |
| Medical concepts, eponyms, ambiguous terms | **Never changed** | They are the ICD evidence |

**Exclusions:**
- **Common-word names:** name tokens that are also common English words or medical terms ("may", "bill", "rose",
  "grace", "bell") are not crossed over. A whole-word replace would corrupt clinical text.
- **Eponym collisions as hard negatives:** in the large run, a surname that is also an eponym (Parkinson, Graves,
  Cushing, Crohn, Addison, Wilson…) is injected as a patient name **only into encounters without that disease**. The
  model has to keep the disease and remove the person. These samples are labeled `ambiguous_terms`.

**Labels follow automatically:** new PII strings get the original category and mention counts, while the
medical-concept, ambiguous-term and `icd10_codes` labels are copied unchanged.

**Two crossovers that keep codes correct, and one that doesn't:**
1. **PII crossover (above):** codes unchanged.
2. **Multi-visit concatenation (§1):** codes = union of the parts, with one surrogate identity across visits.
3. **Not done:** splicing dialogue or note segments between encounters. The text would no longer match its codes,
   and dialogues would lose coherence.

**Volume:**
- **Pool size:** pools come from 140 encounters (est. ~250 names, ~100 ages, ~150 dates, ~150
  org/location/occupation strings) plus external lists.
- **Pilot:** generate a fresh variant at every prompt visit, so ~450 visits produce ~450 distinct samples from 140
  encounters.
- **Large run:** generate on the fly in the dataloader, with seeds logged for reproducibility.
- **Eval is never augmented for training.** The §5 style-shift set is test-only.

**Invariance checks (deterministic, no model; part of the unit tests):**
- Every silver medical concept, ambiguous term and ICD-evidence span is present with the **same count** in the
  augmented text.
- `icd10_codes` is identical.
- No original PII string survives, except tokens shared with a medical concept.
- Every new PII string is present with its recorded mention count.
- Encounter sex and age band are unchanged.
- Later, only when you allow model runs: the harness ICD coder on 20 original/augmented pairs gives agreement F1
  ≥ 95%.

**Labeling cost at scale** is the bottleneck for new datasets, not GPU time. Pre-annotate with the pilot policy, then
use a strong-model reviewer to correct, instead of annotating from scratch.

## 5. Scalability checks (built from eval + gold, deterministic; run after the pilot)

| test | construction | pass criterion vs. plain eval |
|---|---|---|
| **Long documents** | Concatenate eval encounters into multi-visit records with one consistent surrogate identity, at two sizes: 8-16K (single pass) and 20-27K (windowed). Gold is remapped | PII removal and concept recall within 2 pts at both sizes; no truncation; nvidia-smi peak ≤ 3.8 GiB at 16K |
| **Style shift** | Eval with true-casing, de-spaced punctuation and alternate speaker tags, using combinations **not** used in training | Within 3 pts |
| **Unseen source** | Per-source metrics; in the large run, a leave-one-source-out validation (train without virtscribe) | Within 3 pts of the mean |
| **Unseen codes** | Split eval by codes seen vs. unseen in train (46% / 54%). Measure ICD-relevant concept recall and `dx` desc-F1 / 3-char-F1 per split | Concept recall for unseen-code evidence within 2 pts of seen. ICD F1 retained (anon vs. orig, harness coder) ≥ 97% on both splits |

## 6. Numbers

| setting | pilot (`data/`, 60 min) | large (4-5 h) |
|---|---|---|
| max prompt / max response | 6,144 / 1,024 (data max 4,530) | 16,384 / 2,560 (15% long 8-16K records, 30% sub-windows) |
| rollouts / step | 16 prompts × n=4 = 64 (~185K tokens) | 24 × n=4 = 96 (~480K tokens, mean ~5K/rollout) |
| est. step time / steps | ~60-80 s / ~25-30 | ~120-150 s / ~90-110 |
| inference export | fp8 (vLLM), `--max-model-len 18944`, nvidia-smi peak ≤ 3.8 GiB | same |
| sampling | T=1.0, top_p=1.0, penalties off, thinking off | same |
| actor LoRA | r=32, α=64, all linear layers, lr 2e-5 | r=64, α=128, lr 1e-5 cosine |
| critic LoRA + value head | r=32, lr 5e-5, 3 warm-up steps | r=64, lr 5e-5, 8 warm-up steps |
| PPO | 1 epoch, mini-batch 16, micro-batches by tokens (~16K), clip 0.2, value clip 0.2 | 2 epochs, mini-batch 32 |
| KL to reference | 0.01 fixed | adaptive, target 6 |
| SFT warm-up | 1 epoch, targets = `dx` (from `icd10_codes` + silver evidence) + silver entities/keeps with `comment` as reason | from the pilot LoRA |

LoRA is used because the actor, the critic and vLLM must share 46 GB, not because of the 4 GB inference rule.
Step times are estimates until the 2-step smoke test replaces them.

## 7. Edits to TRAINING_PLAN.md (after approval)

1. **§0:** findings table above; 4 GB applies only at inference (8-bit export).
2. **§1:** window, carry-over and truncation fallback; output schema `dx` → `entities` → `keep`; no ICD in the prompt;
   ICD prefill at inference.
3. **§2:** silver labels include evidence-to-code links. Add the ICD-10-CM code table as a resource. Add the
   pre-annotate + review flow for new datasets.
4. **§3:** per-decision token-level rewards and the open-vocabulary `dx` rewards.
5. **§4, §6, §7:** numbers from §6 above. The augmentation table goes into data building, and the §5 tests go into
   P3 / final eval.
6. **§8:** the no-ICD bonus and the robustness bonus are backed by the §5 tests.
7. Add a "Starting checkpoint" section: post-trained `Qwen3.5-2B`, with Base only if output diversity collapses.

## Verification

- **Reward unit tests**, using eval gold as if it were silver:
  - The gold output's return equals S(gold) − S(orig).
  - The empty output receives only the `]` penalty.
  - "Redact all words" gives a strongly negative return.
  - Invalid JSON gives −1.
  - A code that is not in the ICD-10-CM table is penalized.
- **Augmentation tests:** the §4a invariance checks on every training encounter × 20 seeds. In addition, every
  silver span stays locatable (case-insensitive) after the style transforms.
- **Smoke test** (only when you allow GPU use): 2 PPO steps, logging the decoded token at every nonzero reward.
- **After the pilot:** `../term-generalizer/evaluate.py` on all 67 eval encounters (codes prefilled and no codes), plus the four
  §5 scalability tests.
