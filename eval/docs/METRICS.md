# Metrics reference

Every number the harness reports, how it is computed, and from what. One row per metric. Names are the report column
and, in `code`, the key in `data/eval/scores/<model>/<suite>.json` (or the file named in the section). The drift
metrics have their own table in [CORPUS-DRIFT.md](CORPUS-DRIFT.md); the gold levels are defined in [GOLD.md](GOLD.md).

All metrics compare the original record with the model's output and never look inside the model. Strings are matched
with one rule everywhere (`common.pattern`): case-insensitive, whole words, and a number never matches inside a larger
number (`59` is not found in `120/59`, `5.9` or `1,590`). A gold string is counted by its occurrences in its own field
(`dialogue` or `note`), so models may rewrite freely and no offsets are needed.

## Which metric answers which organizer criterion

| Task criterion (task PDF, p. 3) | Headline metric | Section |
|---|---|---|
| (i) proportion of removed privacy-related information | privacy (consensus); without gold: removed (Phi) | [Privacy](#privacy), [Phi check](#independent-privacy-check-phi) |
| (ii) F1 of ICD-10 code prediction, anonymized vs original | F1 on the anonymized text and Δ F1 (anonymized minus original) of the LLM judge (`llm_judge`, code level) and the decision tree (`tree`, category level), the LLM judge's agreement; Kaggle: agreement only | [ICD-10 utility](#icd-10-utility) |
| (iii) recall of medical concepts | concepts kept, ICD concepts kept | [Text utility](#text-utility) |
| Scalability: runtime, peak RAM, peak VRAM (≤ 4 GB) | ms/enc, peak RAM MB, peak VRAM ≤ 4096 MB | [Efficiency](#efficiency) |
| Bonus (a) adversarial attacks, proportion of recovered information | recovered by the LLM agent adversary (A6), max over A1-A4 and A6 | [Attacks](#adversarial-attacks-bonus-a) |
| Bonus: explainability share | explained | [Explainability](#explainability-bonus) |
| Bonus (d): no ICD-10 codes at inference | ablation: every headline metric with and without codes, Δ | [No-codes](#no-codes-bonus) |
| Bonus (e) robustness, "complexity of additional data" | robust-suite privacy, complexity rank | [Robustness](#robustness-bonus-e) |

## Validity

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| valid | `validity.valid_rate` | Share of input records whose output record exists, has exactly the input keys (plus an optional `explanations` list), the same `dataset`, string `dialogue`/`note`, and `icd10_codes` byte-identical to the input (reordering counts as a change). Record count, ids and order must match too; violations are listed in `validity.errors` | 0-1, must be 1 |
| records changed | `validity.records_changed` | Share of records where `dialogue` or `note` differs from the input | 0-1 |

## Privacy

Gold: the suite's `pii` items. On `aci_eval` that is the merged gold of three annotation sets (harness, VM, ketai),
each item with `votes` (how many golds list it) and `direct` (a direct identifier). On other suites, single-source gold.

**Occurrence leak rule** (`metrics.leaked`). For a gold item with `n0` occurrences in the original field, the leaked
count `n1` is the larger of: the whole string's occurrences still in the output, and the item's occurrences in which a
*distinctive token* survives at its own position (the token alignment of `metrics.align`; a survivor of the same word
elsewhere in the field, such as `Horner's syndrome` next to `HORNER, Dennis`, is neither credited nor blamed). Distinctive tokens are
digit runs of 3+ digits, and words of 4+ characters that are not cue words (`patient`, `hospital`, `street`, ...) and,
outside name categories, not ordinary English words (Debian `wamerican-large`). So `[PATIENT] Campbell` leaks
`Andrew Campbell`; `[PATIENT] Li` does not leak `Wei Li` (tokens under 4 characters never count).

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| privacy (consensus) | `privacy.removed` | Σ(n0 − n1) / Σ n0 over the gold items of the **consensus** level: direct identifiers or items with ≥ 2 votes. On suites without votes, over all items. The headline (organizers' criterion i) | 0-1, higher better |
| high-risk | `privacy.removed_high_risk` | The same over items whose risk is high: PATIENT_NAME, PERSON_NAME, DOB, DATE, AGE, ID, CONTACT | 0-1 |
| entity-level | `privacy.entity_removed` | Share of consensus items fully removed: n1 = 0, no occurrence and no distinctive token left (the paper's strict entity-level matching) | 0-1 |
| direct | `privacy.levels.direct.removed` | Occurrence removal over direct identifiers only (names, DOB, dates, IDs, contacts): a gate | 0-1 |
| union | `privacy.levels.union.removed` | Occurrence removal over every merged item, including single-vote quasi-identifiers: a pessimistic bound | 0-1 |
| level entity | `privacy.levels.<level>.entity_removed` | Entity-level removal per level | 0-1 |
| privacy (harness gold), VM, ketai | `privacy.by_source.<gold>.removed` | Occurrence removal on one annotation set's own item list (`pii_by_source`), untouched by the merge. `harness` reproduces the A+D team's own number (0.8388 for `ad_night`) | 0-1 |
| votes:1/2/3 | `privacy.breakdown["votes:<n>"]` | Occurrence removal over union items with exactly n votes | 0-1 |
| src:\<gold\> | `privacy.breakdown["src:<gold>"]` | Occurrence removal over union items that gold supports | 0-1 |
| by category, field, language | `privacy.breakdown["cat:…"/"field:…"/"lang:…"]` | Occurrence removal over consensus items of one category, one field, or one record language | 0-1 |
| counts | `privacy.counts` | [removed, total] occurrences behind every share above | – |
| leaks | `privacy.leaks` | Every item with n1 > 0: text, category, risk, votes, left/of. Quotes PII, never in the report | – |

## Text utility

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| keep traps | `keep.kept` | Σ min(n0, occurrences in output) / Σ n0 over the gold `keep` strings: things that must survive (eponyms such as `Parkinson's`, `NOT_A_NAME` traps, clinical look-alike numbers such as `3/6` murmur, ICD code evidence) | 0-1, higher better |
| concepts kept | `concepts.kept` | The same over gold medical-concept strings (organizers' criterion iii). `aci_eval`: the VM and ketai concept annotations merged (6,607 items) | 0-1 |
| ICD concepts kept | `concepts.breakdown.icd_relevant` | The same over concepts flagged relevant to the encounter's gold codes | 0-1 |
| concepts by type | `concepts.breakdown["type:…"]` | The same per concept type (DIAGNOSIS, SYMPTOM, FINDING, MEDICATION, PROCEDURE, TEST, ANATOMY) | 0-1 |
| non-PII kept | `edits.non_pii_tokens_kept` | Word tokens of each field aligned to the output with difflib (no autojunk); share of tokens outside gold PII occurrences that the alignment keeps. Exhaustive-gold suites only (there unlabelled = not private) | 0-1, higher better |
| PII tokens changed | `edits.pii_tokens_changed` | Share of tokens inside gold PII occurrences that the alignment marks as changed | 0-1 |
| edit precision | `edits.edit_precision` | Changed tokens that were gold PII / all changed tokens. Low = the model removes text the gold calls harmless | 0-1 |
| terms | `terms.kept` | Occurrences of ICD-10-CM 2023 description words (4+ letters, generic words removed) outside gold PII, kept in the output / in the original: an independent medical vocabulary for suites without concept gold | 0-1 |
| placeholders / 1k words | `placeholders_per_1k_words` | `[UPPER_CASE]` tags per 1,000 output words: an over-redaction signal readable without gold | ≥ 0 |
| Kaggle medical mean | report only | Mean of a metric over the medical Kaggle suites, shown only when the model has a value on every one of them | 0-1 |

## ICD-10 utility

Coders (`eval/coders.toml`) predict codes from the original and from the anonymized text of the same records; only
records with gold codes are scored. `llm_judge` is the primary coder, the report's **LLM judge**: Claude Sonnet 5.5
agents code each note from a prompt (`eval judge`, below the table; note only, every version of a note in a different
batch; with repeat passes, the codes most passes give per note). `tree` is the
**decision tree** of `../decision-models` (ICD-10 lexicon candidates scored by a depth-2 tree fit on the 140 ACI train
encounters); its own metric is the 3-character category F1, so the report reads it at the category level. Codes are normalized to one canonical ICD-10-CM spelling,
gold codes too, and matched exactly. A *ranked* coder (`knn`, `lexical`, command) is cut at k = the record's number of
gold codes; an *unranked* one (the LLM coders) is scored on its whole set. Per coder in `icd.<coder>`:

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| precision, recall, F1 (micro) | `icd.<c>.<arm>.code.{precision,recall,f1}` | Pooled over records: tp / (tp + fp), tp / (tp + fn), 2tp / (2tp + fp + fn). An empty denominator scores 0. `arm` = `original` (coder quality, the ceiling) or `anonymized` (the organizer-style number) | 0-1 |
| ICD F1 anon | `icd.<c>.anonymized.code.f1` | The anonymized arm's micro F1 (headline with the primary coder) | 0-1 |
| mean per-note F1 | `icd.<c>.<arm>.code.mean_f1` | Mean over records of each record's F1 | 0-1 |
| CI95 | `icd.<c>.<arm>.code.ci95` | 2.5 and 97.5 percentiles (linear interpolation) of the micro F1 over 1,000 bootstrap resamples of the records, seed 0 | – |
| category level | `icd.<c>.<arm>.category.*` | The same with every code cut to its 3-character category (`E11.9` → `E11`) | 0-1 |
| ICD agree | `icd.<c>.agreement.code.f1` | Anonymized-text predictions scored against original-text predictions as if those were gold: the change anonymization alone causes, whatever the coder's own errors. 1 = identical predictions | 0-1, higher better |
| category Δ F1 [CI95] | `icd.<c>.f1_delta_category` | Anonymized minus original category-level micro F1 with the paired bootstrap CI: the decision tree's Δ F1 | −1-1, 0 = no loss |
| agreement only | `icd.<c>.agreement` without `original`/`anonymized` | On a suite without gold codes (Kaggle), an unranked coder is scored on the agreement arm alone, over every record | 0-1 |
| F1 Δ [CI95] | `icd.<c>.f1_delta` | Anonymized minus original micro F1, with a paired bootstrap CI: both arms scored on the same 1,000 resamples | −1-1, 0 = no loss |
| errors | `icd.<c>.errors.{original,anonymized}` | Predictions that failed (bad JSON, cut generation, timeout) after retries; scored as empty sets, never cached | count |
| invalid codes | `icd.<c>.invalid_codes` | Predicted strings that are not valid ICD-10-CM syntax, dropped before scoring | count |
| valid for comparison | `icd.<c>.valid_for_comparison` | False when any prediction failed; the report marks such cells with `!` | bool |
| coder error | `icd.<c>.error` | The whole coder could not run (Ollama unreachable, three consecutive failures); the rest of the score is still written | text |

## Efficiency

From `data/eval/runs/<model>/<suite>/run.json`, one fresh process per (model, suite), Linux only.

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| wall s | `run.wall_s` | From process start to its exit (model loading included) | s |
| ms/enc | `run.ms_per_encounter` | 1000 × wall s / records | ms |
| peak RAM MB | `run.peak_ram_mb` | Peak of the summed RSS of the process and all its children, sampled every 0.2 s, at least the kernel's `ru_maxrss`. Overcounts shared pages | MB |
| peak VRAM MB | `run.peak_vram_mb` | Peak of the summed `nvidia-smi` memory of the tree's processes, sampled every 0.5 s, CUDA context included | MB |
| VRAM ≤ 4096 MB | `run.within_vram_budget` | peak VRAM ≤ 4,096 MB, the challenge budget | yes/no |
| timed out | `run.timed_out` | The model exceeded its `timeout` (default 3,600 s) and its process tree was killed (exit code 124) | bool |

## Explainability (bonus)

From `bonus/explain.py`, stored as `explain` in every score file. A model logs decisions in
`runs/<model>/<suite>/output.decisions.jsonl` (or an inline `explanations` list), each with offsets into the original.

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| explained | `explain.explained` | Original word tokens that the token alignment marks as changed, covered by a decision (other than `keep`) with a non-empty reason / all changed tokens. Null when nothing changed, 0 when tokens changed but no decisions were logged | 0-1 |
| decisions with a reason | `explain.with_reason` / `explain.decisions` | The literal "decisions having an explanation" ratio, shown beside it | 0-1 |
| unaligned | `explain.unaligned` | Decision rows that could not be placed in the original text | count |

## Adversarial attacks (bonus a)

From `bonus/attack.py`, in `data/eval/attacks/<model>/<suite>.json`. Each attack runs on the model's output
(`anonymized`) and on the original text (`control`, the positive control: what the attack gets when nothing was
removed). Gold on `aci_eval`: the consensus level. `recovered` = hits / total.

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| A1 NER residual scan | `attacks.A1.<arm>.recovered` | Independent detectors (deid_roberta i2b2, GLiNER multi-PII, the VM regex layer) scan each field; a hit = a non-low-risk gold item covered by a detected span (the span contains the item, or is whole words of it, 3+ characters, not only titles or cue words). Spans with a placeholder do not count | 0-1, lower better |
| A2 roster dictionary | `attacks.A2.<arm>.recovered` | The attacker knows every person name in the suite's gold plus a public first-name list and greps for them; a hit = a name token of a gold name still present (whole word) in its own encounter field | 0-1 |
| A3 rare words | `attacks.A3.<arm>.recovered` | Identifier-like tokens (4+ letters, proper noun or known first name, not clinical, not ordinary English, not a month or weekday) that occur in only one encounter; a hit = still present in that encounter. Needs no gold | 0-1 |
| A4 LLM inference | `attacks.A4.<arm>.recovered` | A local LLM (Ollama qwen3.5:2b by default, `attacks.A4.llm`) reads the record and guesses the patient's name, age, place and employer or job; hits against the gold values (any name token, age within ±1 year, a gold place or job inside the guess). A refusal or failed call counts in `attacks.A4.errors` and as no guess | 0-1 |
| A6 LLM agent adversary | `attacks.A6.<arm>.recovered` | The same guesses made by LLM agents from a prompt, one agent per blinded model output (`eval adversary`); `control` = the agent run on the original text, `llm` = the agent's model, `missing`/`malformed` = records without a usable answer (no guess). The report's "Proportion of recovered information (LLM adversary)" and the x axis of the adversarial Pareto front | 0-1 |
| A5 record linkage | `attacks.A5.<arm>.recovered` | TF-IDF nearest neighbour from each output record to the original records (placeholders removed); a hit = it links to its own original. Near 1 by design (clinical content is kept): bounds the threat model, excluded from the max | 0-1 |
| by category | `attacks.<A>.<arm>.by_category` | The same share per gold category | 0-1 |
| recovered (max) | `max_recovered` | Largest anonymized `recovered` over the attacks that ran, A5 excluded | 0-1, lower better |

## Independent privacy check (Phi)

From `bonus/phi_privacy.py`, in `data/eval/phi/<model>/<suite>.json`. The fine-tuned `ab-ai/PII-Model-Phi3-Mini`
(Q4_K_M GGUF, Ollama) scans each field in chunks of at most 2,000 characters with the model card's prompt; its labels
map to the gold categories (sex, gender, height, amounts, currencies and title prefixes dropped; placeholder values
ignored). It needs no gold, so it also sees PII the golds missed, and it counts what it misses itself as kept.

| Metric | `code` | How it is computed | Range |
|---|---|---|---|
| PII before (Phi) | `total.before` | Distinct (category, value) mentions the detector lists per chunk of the original text, summed over chunks | count |
| PII after (Phi) | `total.after` | The same on the model's output | count |
| removed (Phi) | `total.removed` | 1 − after / before, clipped at 0; null without mentions before. A generalized value (`fifty-something`) still counts after when the detector reads it as an age. The detector's false positives in untouched text (`Chief Complaint` read as a job) count before and after, so this is a lower bound, best compared between models | 0-1, higher better |
| by category | `by_category.<cat>` | The same per gold category | – |
| errors | `errors` | Chunks whose answer could not be parsed: counted as no mention | count |

## No-codes (bonus)

From `bonus/nocodes.py`. A `blind_codes = true` model (`icd10_codes` hidden at inference, restored before scoring) is
paired with its base model (`base` key, else the name without `_nocodes`).

| Metric | How it is computed |
|---|---|
| base, no-codes, Δ | For privacy (consensus), high-risk, keep traps, concepts, ICD concepts and each coder's anonymized ICD F1: the base model's value, the no-codes model's value, and no-codes minus base (`results.csv` rows `delta_vs_base:*`) |
| ablation table | The report: every pair side by side over the main-task headline metrics (privacy, direct, concept recall and its ICD-relevant part, F1 of the LLM judge and the tree, the LLM adversary's recovery), with Δ = without minus with codes |

## Robustness (bonus e)

From `bonus/robustness.py`, over the three robust suites (`robust_discharge`, `robust_medconv`, `robust_synthetic`).

| Metric | How it is computed |
|---|---|
| privacy, high-risk, terms per robust suite | The privacy, high-risk and terms metrics above, on each robust suite |
| complexity rank | The suite's rank by unigram JSD from the ACI train corpus (1 = closest), from `data/eval/drift.json`: the task's "complexity of additional data". All drift metrics: [CORPUS-DRIFT.md](CORPUS-DRIFT.md) |

## Tradeoff

| Metric | How it is computed |
|---|---|
| Pareto | A model is Pareto-optimal when no other non-reference model is ≥ on both consensus privacy and ICD F1 on the anonymized text (primary coder) and > on one. `identity` is a reference and never competes. While no model has a primary-coder result, ICD concepts kept stands in (the axis title says so). No F1 ratio: anonymized ÷ original F1 of a weak coder exaggerates noise and can exceed 1 |
| `tradeoff.svg` | x = consensus privacy, y = ICD F1 on the anonymized text (primary coder); Pareto-optimal models filled, references as squares |
| `tradeoff_adversarial.svg` | x = 1 − the LLM agent adversary's proportion of recovered information (A6), y = ICD F1 on the anonymized text (LLM judge); the same Pareto rule |

No combined score: the organizers weigh the criteria separately.

## Reference row

| Row | What it is | What it shows |
|---|---|---|
| `identity` | The input copied unchanged | Privacy 0, every utility metric at its ceiling; the ICD coder's own quality |
