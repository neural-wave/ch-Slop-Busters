# Merged review of the PPO training plan — clinical de-identification


## Verdict

Keep the tag-not-rewrite architecture, fix the reward before any training step, prove the 4 GB deployment before the clock starts, and make SFT the baseline that PPO has to beat. Both reviews agree on that spine; they differ in what they caught. Review 1 found the data-specific holes (selective over-redaction, external-cause codes, sparse signal, lowercase dialogue). Review 2 found the arithmetic errors (beta and over-redaction weight point the wrong way) and the process risks (deployment tested last, eval set doubling as dev set, custom PPO fallback).

This merged report keeps every point that survives scrutiny, drops two that were wrong or redundant, and corrects one factual slip in Review 1 after checking the uploaded data. It is a design review, not a reproduced run: baseline numbers, GPU timings and export compatibility remain the team's claims.

## Data facts checked against the uploaded files

Review 1's dataset numbers hold, with one correction: the golf encounter (`Y93.53`, `W18.49XA`) is in the **eval** split, not train. More important, the train split contains **no** external-cause codes at all, while the eval split has 24 of them across 153 codes. A silver annotator calibrated on train will never see a V–Y chapter code, and the hidden test set has the same structure as eval.

| Fact | Train | Eval |
| --- | --- | --- |
| Encounters | 140 | 67 |
| ICD codes per encounter | 2.04 | 2.28 |
| Median chars (dialogue + note) | 8,800 | 9,100 |
| External-cause (V–Y) codes | 0 of 285 | 24 of 153 |
| Z codes | 15 | 7 |
| Dialogues entirely lowercase | 138 of 140 | 67 of 67 |

The lowercase, spaced-punctuation dialogue (`hi , andrew .`) is confirmed, so any capitalization-based name heuristic works on notes only. Review 2 made no data claims; its findings are reasoning and process checks.

## Must fix before training: the reward

Five changes, in order of severity. The first two are arithmetic errors (Review 2); the third and fourth are the same loophole found from two directions (Review 1 and Review 2); the fifth is edge-case hygiene (Review 2).

1. **Beta points the wrong way.** With R = (1 + beta²) P U / (beta² U + P), beta → 0 gives R → U and beta → ∞ gives R → P. So beta > 1 favors privacy and beta < 1 favors utility, the opposite of what the plan states. The verified example P = 0.80, U = 0.95 gives R = 0.916 at beta 0.5, 0.869 at beta 1, 0.826 at beta 2. Raising beta when utility drops moves the balance the wrong way.
2. **Over-redaction weight points the wrong way.** The term is subtracted, so lowering its weight when utility drops makes over-redaction cheaper, not costlier.
3. **Selective over-redaction is unpunished.** `over_redaction` fires only on spans overlapping a silver medical concept. Redacting every non-medical, non-PII token (verbs, fillers, "golfing") scores P = 1, U = 1, no hallucination penalty, and loses at most the 0.10 explain term. The "redact every token" sanity test misses this because that test also kills the concepts.
4. **Whole-chunk deletion on PII-only chunks scores 1.0.** U = 1 when a chunk has no annotated concept, so deleting the chunk gives P = U = 1. Whether the renderer accepts such a span is unverified; the reward itself does not reject it.
5. **Edge rules are missing.** P = U = 0 needs a zero-denominator rule. The identity output should score well on a PII-free chunk, so the universal identity-to-zero test is wrong. U must say how eponyms are normalized so it stays on scale.

**One fix covers 3 and 4:** add a precision term over applied spans (spans not matching silver PII cost reward), or make P an F-measure against silver PII rather than pure recall. Then split the sanity tests into four cells (PII present or absent × concepts present or absent) and write numerical tests that show the direction each reward parameter moves a pair of candidate outputs.

Two residual limits stay as monitoring items, not reward terms: preserving disease words does not preserve negation, severity, timing or who has the disease; and a type-correct 2–15-word reason does not make an explanation correct.

## Labeling spec and renderer

The silver labels decide what SFT and RL can learn, so these spec gaps matter more than any hyperparameter.

- **External-cause rule (Review 1, strengthened by the data check).** Context that supports a V–Y chapter code or a Z code is `icd_relevant`, never `OTHER_QUASI`. "i was golfing" is the only evidence for `Y93.53`; removing it costs one of \~2 codes on that record, a 50% F1 hit. Because train has zero such codes, add synthetic or hand-written external-cause examples to the silver set or the annotator will treat every activity and place as quasi.
- **Occurrence-specific spans (Review 2).** The JSON names text strings without positions. If "Parkinson" appears as a surname and in a disease name in one chunk, the renderer cannot know which to edit. Emit (text, occurrence index) or character offsets, and validate them.
- **Case (Review 1).** `find_spans` and gold matching must be case-insensitive, and the silver annotator must have seen the lowercased dialogue.
- **Encounter-level linking (both).** A `PATIENT_NAME` tagged in chunk 1 and missed in chunk 7 leaks the encounter. Propagate direct identifiers across chunks at render time; it is deterministic and raises P without touching the model. Also overlap chunk boundaries.
- **Regex pre-pass is irreversible (Review 2).** If rule redactions are always applied, the model cannot undo a false positive (a clinical date, a dose that looks like a year). Add clinically aware exceptions or let the model veto.
- **Token budget (Review 2).** 2,000 chars plus label spec plus ICD hints may not fit 1,024 prompt tokens, and dense entity lists can overflow the response. Measure with the real tokenizer; fail loudly rather than truncate.

## Training strategy: SFT first, RL conditional

Both reviews reach the same conclusion from different premises. Review 1: silver span labels make this a supervised extraction task, so SFT does 90% of the work and \~2,900 noisy rollouts over 716 chunks (\~4 epochs) can add a few points, not transform the model. Review 2: there is no evidence yet that PPO is the highest-return hour, and beating the old zero-shot baseline proves nothing when prompts, labels and rendering changed too.

1. **SFT on the tag format is mandatory**, not optional (both). It is also the fallback submission if training, export or integration fails.
2. **Run the three-arm comparison** on identical data, rules, rendering and deployment precision (Review 2): current model + rules; SFT + rules; SFT + RL + rules. Only arm 3 beating arm 2 justifies RL.
3. **The pilot uses PPO** with a learned critic, as the team decided. Review 1's critic-free alternative (GRPO/RLOO) is not taken, so its costs must be managed instead. The critic costs \~4 GB, 5–10 warm-up steps of \~45, and an explained-variance debugging loop; the plan's own "check the critic first" shows it is the fragile part. Mitigations: warm up the critic on SFT-model rollouts before the clock starts, log explained variance from step 1 and abort if it stays near zero after 10 steps, and raise samples per prompt from 2 to 4 so the baseline does not rest on the critic alone.
4. **Oversample PII-bearing chunks** (about 70/30) (Review 1). Many exam and plan chunks have zero PII, where P = 1 trivially and the batch is noise.
5. **Raise the LoRA learning rate** from 1e-5 to 2–5e-5 for a \~45-step run, checked against the SFT rate (Review 1).
6. **Drop the 400-line custom PPO fallback** (Review 2). Framework failure should return the team to the SFT submission, not open a second implementation project.

RL earns its place only for what SFT cannot express: the P/U trade-off knob, the hallucination penalty and format robustness.

## Deployment gate and schedule

The 4 GB VRAM constraint is the only hard pass/fail in the brief, and the plan tests it in the last four minutes. Review 2 flags this as a submission blocker; Review 1 flags the same minutes as the least realistic part of the schedule. Together they say: move export and deployment before the clock, then resize the pilot.

**Before the clock (no trained weights needed):**

- Load the intended quantized model (fp8 or GGUF Q8\_0) in the intended backend under a 4 GB cap, on long and identifier-dense inputs. Quantized weight size is not runtime memory; context length, batch size, caches and CUDA graphs all count.
- Run the whole export pipeline (merge adapter, drop vision tower, quantize, convert) on the untrained model. `convert_hf_to_gguf` support for the Qwen3.5 hybrid Gated-DeltaNet architecture is not something to discover with four minutes left.
- Load a small untrained adapter through the same path.
- Run P0 (pure inference baseline) and benchmark one full RL iteration and one full eval + export cycle.

**Schedule corrections (Review 1 specifics):**

| Item | Plan | Reality |
| --- | --- | --- |
| P4 export + quantize + GGUF | 4 min | Pre-run; keep 4 min only for the trained adapter |
| P3 two full evals of 67 encounters + ICD coder | 8 min | Coder is too slow for the loop; run once or offline |
| vLLM sleep/wake per cycle | not counted | 30–90 s × 45 steps |
| Validation every 10 steps on 85 chunks | not counted | Comes out of the PPO window |
| Real RL steps in 60 min | 45 | \~30 |

**After training:** rerun privacy and utility on the exported, quantized model. A bf16 checkpoint does not certify the deployment. Pin runtime, quantization, context limit and batch settings used for the submitted numbers.

## Evaluation hygiene

This section is Review 2's alone; Review 1 only praised the eval-gold firewall. The plan's "gold" is two model passes plus review, which is a reference set of uncertain accuracy: two passes share systematic errors, and training and evaluating against the same annotator rewards agreement with the annotator, not anonymization. The 67 eval encounters are also used for checkpoint selection, so they are development data, not a test set.

- Reserve an encounter-level holdout; keep an encounter's chunks together.
- Manually audit a targeted annotation sample: names, occupations, quasi-identifiers, eponyms.
- Report paired uncertainty on model comparisons, resampled at encounter level.
- Report per-category recall and the fraction of encounters with any surviving direct identifier; an 83% mention-level average can hide records that leak.
- Report absolute ICD prediction F1, not only retention relative to the unanonymized baseline. High retention of a weak coder's output is weak evidence of utility.
- Spend the one organizer mid-term evaluation on the pilot output (both reviews), and use it to check whether local metrics track the organizers' metrics.

The local harmonic reward is a proxy. Equal weights in the judging rubric do not mean the organizers compute a harmonic mean.

## Attacker, explanations and originality

**The in-loop attacker is inert (both reviews).** A frozen 2B model asked to guess a deleted name from one chunk recovers \~0% from step one, so the −0.2 penalty never fires and only adds latency. Review 2 adds the converse failure: a correct age or occupation guess may come from priors, not leakage. Real re-identification is linkage across surviving quasi-identifiers (age band + place + occupation + rare condition), which a per-chunk attacker cannot attempt.

Replace it with two things:

- **In the loop:** a rarity penalty on tokens surviving near redacted spans, scored by corpus IDF (Review 1). Cheap, dense, and it targets the "rare words" attack surface the brief names.
- **Offline, as the bonus deliverable:** three separate attacks with a larger model and encounter-level context (Review 2). Residual extraction asks whether identifiers are still present. Attribute inference asks whether attributes beat a prior-only baseline. Record linkage asks whether anonymized records match candidates from combined clues. Give the attacker everything the evaluator sees, including preserved ICD metadata, and keep it outside training.

**Explanations.** Reward presence of a reason and the model learns plausible wording, not faithful reasons. Generate explanations from the rule or transformation that actually fired where possible, and sample-check correctness. Drop the "AGE → decade band (HIPAA Safe Harbor)" wording: Safe Harbor has specific age and date provisions (ages over 89 aggregate to 90+), and decade banding is not a compliance certification. Describe the transformation, not a guarantee.

**Originality (25% of the grade).** A deterministic perturbation layer for non-ICD-relevant numerics, as in the organizers' HbA1c 8.3 → 8.1 example, is cheap and visible (Review 1). Encounter-level linking and the linkage attack are the other two originality points already in this report.

## Revised execution order

Each step has a gate; do not start the next until the gate is met. Steps 1–3 run outside the GPU clock.

1. **Fix the reward and the labeling spec.** Gate: four-cell sanity tests pass, including selective over-redaction and whole-chunk deletion scoring low; beta and penalty-weight direction tests pass; external-cause rule in the spec.
2. **Prove deployment within 4 GB.** Gate: quantized model and a dummy adapter load and produce valid output on the longest, densest inputs under the memory cap.
3. **Audit labels and freeze evaluation.** Gate: manual audit of the targeted sample done; encounter-level holdout reserved; absolute ICD F1 and per-category recall reported for the current baseline.
4. **Rules, entity linking, SFT.** Gate: measured gain over the current baseline on the frozen evaluation, same rendering and precision.
5. **Mid-term organizer evaluation** on the SFT output. Gate: feedback received; local metrics compared with organizer metrics.
6. **Bounded PPO pilot (\~30 real steps, PII-oversampled, critic pre-warmed).** Gate: arm 3 beats arm 2 on the exported quantized model without utility or runtime loss. If not, ship SFT.
7. **Freeze the deliverable.** Final JSONL from the exported model, README with pinned runtime and settings, measured memory and runtime, claims that match what was run.

**Six questions to settle in the team review**

- [ ] Does the plan text state the beta direction as Review 2 describes? If so, who fixes the reward code and its tests?
- [ ] Can the exact deployment load a trained adapter within 4 GB today?
- [ ] PPO is settled as the pilot algorithm. Who owns critic warm-up and the explained-variance abort check?
- [ ] Who checks the annotation categories most likely to differ from the organizers' interpretation (eponyms, activities, places)?
- [ ] How will external-cause examples enter the silver set, given train has none?
- [ ] What is the fallback submission if training, export or integration fails?
