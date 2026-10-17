# Evaluation pipeline

One harness that runs every team model on the same data and scores it with the same code. Models are black boxes:
a model reads a challenge-format JSONL and writes an anonymized one. The scorer only compares input and output, so a
rule system, an NER stack and an LLM rewriter are measured the same way.

Run everything from the repo root (`neuralwave/`):

```bash
python3 -I eval all                                   # every enabled model x every suite, then the report
python3 -I eval all --models ad_night fable --suites aci_eval
python3 -I eval score --icd llm_note knn "cmd:python3 coder.py --input {input} --output {output}"
python3 -I eval report                                # data/eval/report.md, report.json, results.csv, tradeoff.svg, ...
python3 -I eval --help                                # every command; `python3 -I eval <command> --help` for its options
```

`python3 -m eval ...` works too, and so does a stage module run as a file (`python3 -I eval/run.py ...`).
(`python3 -I -m eval` does not: isolated mode leaves the current directory off `sys.path`, so Python cannot find the
package; `python3 -I eval` runs `eval/__main__.py` directly.)

Needs Python 3.11+ (standard library only) and `data/converted/` (see [`DATA_PREP.md`](../../DATA_PREP.md)). Model
runtimes are separate: see [Models](#models). The LLM ICD coders need a local Ollama server with the coder model
(`ollama serve`, `ollama pull qwen3.5:2b-q4_K_M`); without it they report an error and everything else is scored. See
[ICD coders](#icd-coders).

Ollama setup that matters for speed and memory:

- **GPU backend.** A CPU-only Ollama build (Arch's plain `ollama` package) codes a note in about 60 s instead of 1-4 s.
  Install a GPU backend (`ollama-vulkan` or `ollama-cuda` on Arch) and check that `ollama serve` logs
  `inference compute ... library=Vulkan` (or CUDA). GPU and CPU generations differ slightly, so keep one device for
  all runs you compare: the LLM caches do not mix them.
- **Prompt cache off.** Start the server as `LLAMA_ARG_CACHE_RAM=0 OLLAMA_NUM_PARALLEL=1 ollama serve`. With the
  default prompt cache, `qwen3.5` keeps about 75 MB of host RAM per coded note (8 GB after about 100 notes), which
  froze a 32 GB machine without swap. The harness does not need that cache: generations are deterministic and
  predictions are cached on disk.
- **Power profile.** On a laptop, `power-saver` cut prompt processing about tenfold; use `performance`.

`daniel-eval/`, `ketai-eval/` and `vm-backup/` (siblings of this repo) are kept as read-only sources: the coder, the
third ACI gold and the VM gold and tools were vendored or merged from them, and nothing here writes to them.

## Commands

| Command | Module | Reads | Writes | Does |
|---|---|---|---|---|
| `suites` | `suites.py` | `data/converted/*`, VM annotations, `data/eval/gold/` | `data/eval/suites/<suite>/` | Fixed, seeded subsets: `encounters.jsonl` (model input), `labels.jsonl` (gold), `suite.json`. Rebuilds every selected suite; `--if-stale` only the missing or stale ones (what `run` and `all` do) |
| `run` | `run.py` | `models.toml`, suites | `data/eval/runs/<model>/<suite>/` | Runs each model in its own process: output, wall time, peak RAM and VRAM of the process tree. Cached. Builds missing or stale suites first |
| `score` | `score.py` | runs, labels | `data/eval/scores/<model>/<suite>.json` | Every metric, plus leak and loss lists (these quote PII) |
| `report` | `report.py` | scores, `runs/*/*/run.json`, attacks, `drift.json`, `suite.json`, merged gold stats, `models.toml`, `coders.toml` | `data/eval/report.md`, `report.json`, `results.csv`, `results_headline.csv`, `tradeoff.svg`, `report_discord.md` | Comparison tables, tradeoff and CSV exports, aggregate numbers only. See [Report and exports](#report-and-exports) |
| `all` | `__main__.py` | | | The steps in order: `suites`, `run`, `score`, `report`, plus the optional `gold`, `attack` and `drift` once they exist |
| `gold` | `gold/merge_aci_gold.py` | the harness, VM and ketai ACI golds | `data/eval/gold/aci_eval/` | Merges the three ACI-eval golds into one with votes (`merged.jsonl`, `merged_stats.json`, `merged_review.md`); `--audit` also writes `audit.md`/`audit.json` (`gold/audit_gold.py`, also runnable alone). Deterministic, so `all` runs it first every time. See [ACI gold and privacy levels](#aci-gold-and-privacy-levels) |
| `attack` | `bonus/attack.py` | runs, suites | `data/eval/attacks/<model>/<suite>.json` | Re-identification attacks A1–A5 on each output, with the original text as positive control. See [Adversarial attacks](#adversarial-attacks-bonus) |
| `robustness` | `bonus/robustness.py` | `data/converted/robust_*`, scores, `drift.json` | the robust sets and suites | Builds the three robust suites when missing and prints the per-model robustness table. See [Robustness](#robustness-bonus) |
| `drift` | `bonus/drift.py` | suites, `data/converted/aci_bench` | `data/eval/drift.json` | Lexical, style and structural drift of every suite from the 140 ACI train encounters, and the `complexity_rank` of the suites. See [Corpus drift](#corpus-drift-bonus) |

`run` and `all` default to every enabled model and every suite. `all` hands `--models`, `--suites`, `--icd` and
`--force` to the steps that take them (and `--if-stale` to `suites`), keeps going when a step fails or crashes, and
exits 1 if any did. A new step is a module with `main(argv: list[str] | None = None) -> int`, registered in
`COMMANDS` and `STEPS` in `__main__.py`.

`data/eval/` is gitignored: it holds anonymized copies of datasets whose licenses forbid re-sharing.

### Caching

- **Suites**: `suite.json` records `input_sha1` and `labels_sha1` (of the suite's two files), `source_sha1`, the sha1
  of everything the suite is built from (the converted source's `encounters.jsonl`, `labels.jsonl` and
  `manifest.json`, for the ACI suites the VM annotation files of its records, and the merged gold
  `data/eval/gold/<suite>/merged.jsonl` the suite uses, also recorded as `merged_sha1`), and `code_sha1`, the sha1 of
  the build code (`suites.py`, `common.py`, `gold/taxonomy.py`). `run` and `all` rebuild a suite when `suite.json` is
  missing or from an older build, its definition in `suites.py` changed, the source or the build code changed, or a
  suite file no longer matches its sha1 (an interrupted build, an edit). They print why. Without the converted source the existing suite is
  kept (with a warning). Building is deterministic: an unchanged source gives byte-identical files. A suite built with
  `eval suites --vm-annotations DIR` records DIR (`vm_annotations`) and every rebuild keeps it until another
  `--vm-annotations` is given; `eval suites --no-merged` likewise records `merged_gold: false` until `--merged`.
- **Runs**: reused when `run.json` has the same `input_sha1`, command, `code_sha1` and exit code 0, and the output
  exists (`--force` reruns). `code_sha1` hashes the resolved command and the files of the model's `fingerprint`, so a
  changed model reruns even when its command is the same, and a gold change reruns `oracle`. While a model runs, its
  `run.json` has `exit_code: null`, so an interrupted run is scored as failed and rerun next time.
- **ICD predictions**: `data/eval/cache/icd/<coder>/<sha1>.json`, keyed by the coder's `config` (for the LLM coders:
  model, digest, prompt, schema and options), the encounter id and the text fields the coder reads. The original text
  is coded once for every model, and a text a model left unchanged reuses that prediction. Failed predictions are
  never cached. Delete a coder's folder to recode.

### Report and exports

`report` rewrites every file below from whatever exists under `data/eval/`, in under a second, and `all` runs it last.
Missing data never stops it: a missing value is `–`, a failed run `failed`, a failed coder or attack `error`. All
files hold aggregate numbers only (no leak lists), so they are safe to paste into a chat or a PR.

| File | Contents |
|---|---|
| `report.md` | Headline on `aci_eval` (privacy at the consensus, direct and entity level, privacy on the harness gold alone, concepts and ICD concepts kept, the primary coder's ICD F1 anon / retained / agreement, ms/enc, peak RAM, VRAM within 4 GB, explained share, max attack recovery; `identity` and `oracle` last, as bounds), then: privacy vs utility with a Pareto flag, privacy levels, votes and sources, ICD coders (one row per model and coder of `coders.toml` or the scores; `!` = not valid for comparison), the main suites, no-codes pairs, attacks, robustness with the complexity rank, suite drift by JSD unigram, the Kaggle tables, `aci_eval` privacy by category and field, efficiency, and how to read it |
| `report.json` | The same numbers (`headline`, `tradeoff`, `icd`, `nocodes`, `attacks`, `robustness`, `efficiency`, ...) and `exports`, the paths of the four files below |
| `results.csv` | Long format, one number per row: `model, suite, category, metric, value, n, ci_low, ci_high, coder, level, notes`. `category` is one of `privacy`, `privacy_by_category`, `privacy_by_field`, `privacy_by_language`, `keep`, `concepts`, `edits`, `terms`, `icd`, `attack`, `robustness`, `drift`, `efficiency`, `explain`, `validity`. `level` is the privacy level (`consensus` = the headline `privacy.removed`, `union`, `direct`, `gold:<source>` for one gold alone, `union` also for `votes:N`/`src:X`) or an attack's `cat:<category>`. ICD metrics are `<arm>.<code or category>.<stat>` (arms `original`, `anonymized`, `agreement`), `f1_delta` and `f1_retention`. Every number of `report.md` is a row (also the breakdowns it leaves out), at full stored precision; suite-level rows (drift, gold counts) have an empty model. UTF-8 with BOM and commas, so Excel and LibreOffice open it without prompts |
| `results_headline.csv` | One row per model, one column per headline metric (keys of `report.HEADLINE`), plus `reference` and `pareto` (1/0) |
| `tradeoff.svg` | Consensus privacy (x) vs ICD F1 retained of the primary coder (y; ICD concepts kept, said so in the axis title, while no real model has a primary-coder result). Filled = Pareto-optimal, hollow = dominated, squares = `identity`/`oracle`. Explicit background, light and dark colors, no external fonts |
| `report_discord.md` | The headline in one short table plus a few bullets, for a chat |

A model is Pareto-optimal when no other real model (references excluded) is at least as good on both consensus privacy
and ICD F1 retained and better on one; there is no combined score, because the organizers weigh the criteria
separately.

## Suites

13 suites (10, plus the 3 robust ones):

| Suite | Records | Gold | Use |
|---|---:|---|---|
| `aci_eval` | 67 | exhaustive, merged gold of three ACI golds (786 PII items: 538 consensus, 261 direct), 6,607 medical concepts, gold ICD-10 | **main score** |
| `aci_dev` | 40 | exhaustive harness gold | second ACI set (A+D tuned on it) |
| `devp_discharge` | 50 | exhaustive, 5 gold ICD-10 codes | other note format, ICD check |
| `imerit_en` / `imerit_multi` | 42 / 60 | partial: patient, physician, DOB | scribe format, DE/FR/ES/NL robustness |
| `sadiqguru` | 4 | partial, hand-labelled | long dialogues |
| `nbme_notes` | 100 | partial: age; graded findings as must-keep gold (`keep`, no concept gold) | findings kept |
| `pmc_patients` | 100 | partial: age | case reports |
| `persuade_pii` / `pii_gpt` | 100 / 100 | names / IDs, e-mails, phones, addresses | generic PII layer, not medical |
| `robust_discharge` / `robust_medconv` / `robust_synthetic` | 166 / 80 / 24 | exhaustive / partial (injected PII) / exhaustive | robustness bonus (role `robust`), see [Robustness](#robustness-bonus) |

The ACI gold is the reference and the Kaggle suites complement it. Each model team saw "its" ACI gold during
development, so `aci_eval` is scored on a merged gold of all three (see [ACI gold and privacy
levels](#aci-gold-and-privacy-levels)) and each gold's own list stays scorable on its own. Nobody tuned on the Kaggle
suites. On `partial` gold only leak recall is meaningful.

### ACI gold and privacy levels

`python3 -I eval gold` (`gold/`) merges the three ACI-eval golds: the harness gold (`data/converted/aci_bench`, made by
the A+D team's annotators), the VM teacher gold (`vm-backup/neuralwave/annotations/eval`, made for the fable model)
and ketai's gold (`ketai-eval/gold/eval_gold.jsonl`). Labels map to the harness categories plus `CONTACT`; per
encounter and field the longest span is canonical and a shorter item inside it adds a nested vote; each item carries
its `votes` (1–3), `sources` and `direct` (names, DOB, dates, IDs, contacts). `aci_eval`'s `labels.jsonl` then holds
every merged item in `pii`, each gold's own list in `pii_by_source`, the merged `keep` (harness keep + ketai's
non-PII terms) and `concepts` (VM + ketai). `eval suites --no-merged` builds the old labels (harness `pii`, VM
`pii_alt`). [`GOLD.md`](GOLD.md) has the rules, the numbers and the caveats.

The privacy block of a score file has three levels:

| Level | Items | Use |
|---|---|---|
| `consensus` | direct, or listed by at least two golds | **headline**: `privacy.removed`, `removed_high_risk`, `breakdown` and `counts` |
| `direct` | names, DOB, dates, IDs, contacts | gate: a real model should come close to 1.0 |
| `union` | every merged item | pessimistic bound (single-vote quasi-identifiers included) |

`privacy.levels.<level>` gives `removed`, `removed_high_risk`, `entity_removed` (share of items fully removed: no
occurrence and no distinctive token left, the paper's strict entity-level matching), `items` and `occurrences`;
`entity_removed` at the top is the consensus one. `privacy.by_source.<harness|vm|ketai>` scores each gold's own list,
so the old numbers stay reproducible: `by_source.harness.removed` is the old `aci_eval` privacy (A+D's cross-check,
`ad_night` 0.8388) and `by_source.vm.removed` the old VM-gold privacy (0.5398). The breakdown also has `level:*`,
`entity:*`, and `votes:*` and `src:*` over the union items. On the other suites the items carry no votes, so the
three levels coincide and `by_source` is empty. `python3 -I eval gold --audit` checks the golds (verbatim spans,
contradictions, regex candidates no item covers, single votes) and that the oracle removes 1.0 at every level.

## Metrics

| Metric | Definition |
|---|---|
| valid | Same records, ids, order and keys, and `icd10_codes` byte-identical (hard constraint). One extra key is allowed: `explanations`, which must be a list |
| privacy | Share of gold PII occurrences gone from the output (case-insensitive whole words; a number does not match inside a larger number, so `59` is not found in `120/59`). A multi-word item also leaks when one of its distinctive tokens survives (`[PATIENT] Campbell` leaks `Andrew Campbell`); tokens under 4 characters are never distinctive, so `[PATIENT] Li` does not leak `Wei Li`. Also reported by risk, category, field and language. On `aci_eval`: the consensus level, with the direct and union levels, entity-level removal and each source gold beside it ([levels](#aci-gold-and-privacy-levels)) |
| high-risk | The same for names, DOB, dates, ages, IDs and contacts |
| keep traps | Share of must-survive gold strings still present: eponyms (`Parkinson's`), `NOT_A_NAME` traps, clinical look-alike numbers (`3/6` murmur), code evidence |
| non-PII kept, edit precision | Token alignment (difflib) of input vs output on exhaustive gold: share of non-PII tokens untouched, and share of changed tokens that were gold PII |
| concepts, ICD concepts | Share of gold medical-concept occurrences kept (all, and those flagged relevant to the encounter's codes) |
| terms | Share of ICD-10-CM description words kept (independent vocabulary, for suites without concept gold) |
| ICD F1, agreement | Local stand-in coders (`coders.toml`): micro F1 vs the gold codes on the original and the anonymized text, at the code and the 3-character category level, the paired-bootstrap F1 change and retention, and anonymized-vs-original agreement (the damage done by anonymization alone). See [ICD coders](#icd-coders) |
| ms/enc, RAM, VRAM | Wall time per record (process start included), peak RSS and nvidia-smi memory of the whole process tree. Budget: 4 GB VRAM |
| explained | Share of changed original tokens (the same token alignment as edit precision, on every suite) covered by a logged decision with a non-empty reason. See [Explainability](#explainability-and-no-codes-bonus) |

### ICD coders

The organizers code the anonymized notes with their own hidden coder. `icd.py` stands in with the coders of
`coders.toml` (code in `coders/`): each one codes the original and the anonymized text of every record that has gold
codes, and both are scored against the gold. A coder never sees the gold codes.

| Coder | What | Reads | Output | Suites |
|---|---|---|---|---|
| `llm_note` (**primary**) | Ollama `qwen3.5:2b-q4_K_M`, daniel-eval's prompt and protocol: JSON schema as `format` (ICD-10-CM syntax only, at most 32 codes), temperature 0, seed 0, thinking off, `num_ctx` 16384 | note | set | `aci_eval` (add `aci_dev`, `devp_discharge` in `coders.toml` on a faster machine) |
| `llm_both` | The same model and instruction over ketai-eval's `DIALOGUE:` / `PHYSICIAN NOTE:` framing, `num_ctx` 24576 | dialogue + note | set | `aci_eval` |
| `knn` | TF-IDF k-nearest neighbours (k = 7) over the 140 ACI train encounters, clinical words only. Supervised: only the ACI train codes | dialogue + note | ranked | `aci_eval`, `aci_dev` |
| `lexical` | Zero-shot: the suite + ACI label space ranked by idf-weighted description words found in the text | dialogue + note | ranked | every suite with codes |
| `paper` | Disabled slot (`kind = "command"`) for the full paper coder | | | |

`score --icd` takes these names (default: every enabled coder with `default = true`: `llm_note knn lexical`; `llm_both` is opt-in, `--icd llm_both`)
or an ad-hoc `"cmd:<command>"`. A command coder gets `{input}` (challenge schema with `icd10_codes: []`) and writes
`{output}`, one `{"encounter_id", "codes": [...]}` per line, ranked unless its entry says `ranked = false`. To add a
coder, add a table to `coders.toml` (keys documented in its header).

How a coder is scored (`icd.<coder>` in the score file):

- Codes are normalized (`coders/codes.py`, from daniel-eval): upper case, a canonical dot after position 3, characters
  other than letters, digits and dots dropped (`i110` and `I11.0` are `I11.0`; `I10` stays `I10`). No hierarchy:
  `I11.0` and `I11.9` differ, and a syntactically valid hallucination is a false positive. An invalid code (`I.10`,
  prose) is dropped and counted in `invalid_codes`. Gold codes go through the same function.
- A ranked coder is cut at k = the record's number of gold codes (so precision = recall, no threshold needed); an LLM
  set is scored whole.
- Three arms: `original` and `anonymized` (vs the gold), and `agreement` (anonymized predictions vs the original
  predictions taken as gold: the damage done by the anonymization alone). Each at the `code` and the 3-character
  `category` level, with micro `precision`, `recall`, `f1`, the mean per-note F1 (`mean_f1`) and the bootstrap
  `ci95` of the micro F1.
- `f1_delta`: anonymized minus original code-level micro F1 with a paired bootstrap CI (1000 resamples of the
  encounters, seed 0; daniel-eval's procedure, which it reproduces exactly). `f1_retention`: anonymized / original.
- A failed prediction (timeout, unfinished generation, malformed answer, prompt over the context budget) counts as an
  empty set, is counted in `errors` and is never cached; `valid_for_comparison` is then false. When the whole coder
  fails (Ollama unreachable, three consecutive failed predictions) its entry is `{"error": ...}` and scoring goes on.

The LLM coders talk to `OLLAMA_HOST` (default `127.0.0.1:11434`) with proxies and redirects disabled. A server that
is not on this machine is allowed but recorded as `endpoint`. A prompt over the budget (UTF-8 bytes + 512 output
tokens + 1024 reserved ≤ `num_ctx`) is rejected, never truncated. `run` unloads every loaded Ollama model before it
measures the models, so a coder left loaded by `score` does not inflate their RAM or VRAM.

### Explainability and no-codes (bonus)

**Explainability** (`bonus/explain.py`, the score file's `explain` block). The task asks for the share of
anonymization decisions that have an explanation. A model logs its decisions in an optional sidecar
`runs/<model>/<suite>/output.decisions.jsonl` (or inline, as an `explanations` list in each output record), one row
per decision with offsets into the original field:

```json
{"encounter_id": "x", "field": "note", "start": 0, "end": 15, "text": "Andrew Campbell", "action": "remove",
 "category": "PATIENT_NAME", "reason": "name introduced with the patient's age", "source": "rule:intro_age"}
```

`action` is `remove`, `generalize`, `perturb` or `keep`. Counting logged decisions would reward logging little, so
the score counts text: `explained` is the share of original word tokens that the token alignment (`metrics.align`,
the one behind edit precision) marks as changed and that a decision other than `keep` with a non-empty reason
covers. `decisions` and `with_reason` give the literal decision ratio beside it, `unaligned` the rows that could not
be placed in the original or are not valid UTF-8 JSON, and `source` says where the rows came from (`sidecar`, `inline`, `none`). `explained` is
empty when nothing changed (`identity`) and 0 when text changed without any log (`oracle`).

| Producer | Sidecar |
|---|---|
| A+D (`ad_*`) | Written by the model as `<output stem>.decisions.jsonl`: `action` REPLACE/GENERALIZE/KEEP and `explanation`, no `text` (offsets index the original). `explain.py` normalizes it. `ad_night_submitted` gets the sidecar delivered beside the precomputed file, which `run` copies along |
| fable (`fable_rules`, `fable`) | Written by `adapters/fable_vm.py`: one `remove` row per replaced merged span, the reason joins each member's detector reason, layer and confidence, `source` the layers |
| Ketai's generalize log | Rows with `text`, `type`, `removed`/`generalized`/`retained` and a reason but no offsets: each covers every occurrence of its text |

Run `python3 -I eval/bonus/explain.py [--models ...] [--suites ...]` to print the scores of finished runs.

**No ICD-10 codes at inference** (`bonus/nocodes.py`). A `blind_codes = true` entry is paired with its base model:
the entry named in its `base` key, else its name without `_nocodes` (`ad_night_nocodes` with `ad_night`).
`nocodes.rows(scores, registry, suite)` gives per pair the base and no-codes values and the delta of privacy,
high-risk, keep traps, concepts, ICD concepts and each coder's ICD F1 on the anonymized text; the report renders them.
`python3 -I eval/bonus/nocodes.py [--suite aci_eval]` prints them.

### Corpus drift (bonus)

`python3 -I eval drift [--suites ...]` (`bonus/drift.py`) measures how far each suite's text (dialogue and note) is
from the 140 ACI train encounters. It answers the robustness bonus's "complexity of additional data". It computes
the lexical, style and structural drift metrics of Cornelius & Rinaldi 2026:

- JSD of the unigram and bigram distributions, in bits (0 means the same distribution, 1 a disjoint one)
- vocabulary Jaccard
- |suite − reference| of entropy, MTLD, Yule's K, Distinct-1/2, 4-gram self-repetition, sentence length (mean and
  std), punctuation, digit and capitalized token ratios, Flesch-Kincaid grade, line breaks per 100 tokens, and the
  key-value, bullet, all-caps and special-character ratios

`data/eval/drift.json` holds the metrics per suite, the reference's own values, and `complexity_rank`: 1 is the suite
closest to ACI train by JSD unigram. The rank covers every suite in `data/eval/suites/`. With `--suites`, an unnamed
suite keeps its entry only while its `encounters.jsonl` (`input_sha1`) and `drift.py` are unchanged; otherwise it is
recomputed too. The neural metrics (GPT-2 perplexity; the embedding metrics MMD, Fréchet, embedding-mean distance,
cluster JSD) are TODO. [`CORPUS-DRIFT.md`](CORPUS-DRIFT.md) has the definitions, the paper's correlations with ΔF1,
and how to read the numbers.

### Adversarial attacks (bonus)

`python3 -I eval attack [--models ...] [--suites aci_eval ...] [--attacks A1 A2 A3 A4 A5 | all] [--limit N]`
(`bonus/attack.py`, ported from the A+D package) tries to recover what a model removed. Every attack also runs on the
suite's original text, the positive control: an attack that recovers little from the original says nothing about
the anonymization. `recovered` is hits / total, overall and per category; `max_recovered` is the largest anonymized
share over the attacks that ran, A5 left out.

| Attack | Attacker | Recovered = |
|---|---|---|
| A1 NER residual scan | Independent detectors: deid_roberta (i2b2), GLiNER multi-PII and the VM regex layer, run by `bonus/_ner_worker.py` with the fable venv | gold items (not low risk) covered by a detected span; spans holding a placeholder don't count |
| A2 roster dictionary | Knows every person name of the suite's gold plus the first names of `nam_dict.txt` (Kaggle `krist0phersmith/pii-real-names-dataset`) | name tokens of gold names still present in their field, more often than the token occurs outside the name |
| A3 rare words | Encounter-unique tokens that are not clinical (ICD-10-CM words, RxNorm), not ordinary English, not calendar words, and are a proper noun of the word list or a known first name | such tokens still present in their encounter (gold-free: `by_category` is the gold category holding the token, or `UNLABELLED`) |
| A4 LLM inference | Ollama `qwen3.5:2b-q4_K_M` (`coders/ollama.py`, schema-constrained JSON) guesses the patient's name, age, place, employer or job | guesses matching the record's gold (any name token, age ±1 year, gold place or job inside the guess) |
| A5 record linkage | Holds the original records; TF-IDF nearest neighbour (scikit-learn's `sublinear_tf` formula, standard library) | records linked to their own original |

Gold is the suite's `pii` at the consensus level (direct identifiers or 2+ votes) when the items carry votes (the
merged ACI gold), else all items; the file records `gold_level`. A5 is an upper bound by design: clinical content is
kept, so whoever holds the originals re-links almost every record. It bounds the threat model (anonymization hides
who the patient is, not which record it was) and is excluded from `max_recovered`. A1's detectors are the ones the
fable models use, so for them A1 is not independent: what their detectors miss, A1 misses too (on `aci_eval` A1
recovers 1.8% from `fable` and 13.6% from `ad_night`, whose own detectors differ).

Cost (measured on `aci_eval`, 67 encounters, RTX 4050):

| Attack | Runs by default | Cost |
|---|---|---|
| A2, A3, A5 | yes (what `all` runs) | standard library, CPU only: about 2.5 s per model (20 s for the 8 models with a run) |
| A1 | only when named | loads two transformer models in the fable venv and scans about one text (a dialogue or a note) per second: ~10 min for 4 models (`--device auto` uses the GPU when 2.5 GB are free, else the CPU with 4 threads, an order of magnitude slower) |
| A4 | only when named | one LLM call per record and version: 67 calls for the control (shared by every model; `identity` reuses it) plus 67 per model. 1.4 s per call on an idle GPU (1.5 min per model), ~6 s while the ICD coder shares the GPU (6 to 7 min per model) |

A1 and A4 cache each result by text in `data/eval/cache/attacks/` (`ner/`, `llm/`; written through a temporary file,
so a killed run loses nothing and the next run resumes where it stopped). The control and unchanged outputs are
computed once, and a later run without A1 or A4 keeps their earlier results while the model output, the suite and
`--limit` are unchanged. A model without a run, a failed run, an unbuilt suite or one without PII gold is skipped with
a one-line reason.

### Robustness (bonus)

Three sets of unseen formats, the A+D team's robustness sets ported to converters (`scripts/convert/robust_*.py`, the
records are identical to theirs) and registered as suites with role `robust`:

| Suite | Records | Content | Gold |
|---|---:|---|---|
| `robust_discharge` | 166 | R1: the devp discharge summaries as lowercase dictation (MRN spoken digit by digit), with injected admission and discharge dates, age, physician and phone. `devp_discharge` already renders the same records as a form and a letter, so R1 keeps the third layout only (`--layouts form letter dictation` rebuilds A+D's 500) | exhaustive, 5 gold codes |
| `robust_medconv` | 80 | R2: real web medical Q&A (cased, punctuated) as dialogue, with an injected patient (name, age, DOB, city, phone, e-mail, employer, job, hospital, visit date) and a signing doctor | partial: the injected values (the posts' own names are not labelled), no codes |
| `robust_synthetic` | 24 | R3: synthetic hospital documents written by the A+D team (ER triage, discharge letter, referral e-mail, nursing handoff, radiology/pathology report, phone triage) | exhaustive, keep traps, hand-assigned codes |

`python3 -I eval robustness [--models ...] [--force] [--run] [--icd ...]` converts the sets when missing (`--force`
reconverts and rebuilds), builds the suites, optionally runs and scores the models on them (`--run`), and prints one
row per model: privacy, high-risk privacy and terms kept on each robust suite, under each suite's complexity rank from
`drift.json` (the bonus's "complexity of additional data"). The robust suites are ordinary suites, so `all` runs and
scores them too once their converted data exists.

## Models

`models.toml` lists every configuration. Each entry has a command template (`{input}`, `{output}`, `{labels}`,
`{suite}`, `{python}`, `{here}` = `eval/adapters`, `{repo}`, `{container}`) or a `precomputed` output path.

| Key | Meaning |
|---|---|
| `cmd`, `cwd`, `env` | Command, working directory and extra environment, with the placeholders above |
| `timeout` | Seconds before the run is killed with its whole process tree (default 3600). Recorded as exit code 124 and `timed_out: true`. A descendant that daemonized itself (double fork, `setsid -f`) has left the tree and survives |
| `fingerprint` | Path, directory or glob, or a list of them (same placeholders; relative ones are taken from `cwd`), hashed into `code_sha1`. Files are hashed with their paths relative to the pattern's fixed prefix |
| `precomputed` | An already anonymized JSONL instead of `cmd`; a new delivery (another sha1) invalidates the cached run. A `<stem>.decisions.jsonl` beside it is copied as the run's explainability sidecar (and part of the cache key) |
| `blind_codes` | `true` hides `icd10_codes` from the model (bonus task) and restores them before scoring |
| `suites`, `enabled` | Only these suites; `false` lists the model as pending and never runs it |

A model runs in its own session, so the harness kills its tree itself on Ctrl-C, SIGTERM and SIGHUP (a SIGKILLed
harness cannot). A command that cannot start (missing executable or `cwd`) is recorded as exit code 127, and the
other models still run.

| Entry | What runs | Setup |
|---|---|---|
| `ad_rules`, `ad_night`, `ad_night_nocodes` | A+D night model (`neural_wave` 93b5eea, Davide's latest zip) in `../models/ad-night/repo`: rules only, and their submission config (rules + deid_roberta + GLiNER), with and without codes. Fingerprint: its `src/` code and lexicons | `uv sync --all-extras` there, then download `obi/deid_roberta_i2b2` and `knowledgator/gliner-pii-base-v1.0` (runs offline) |
| `fable_rules`, `fable` | Fable model (`../vm-backup/neuralwave`) through `adapters/fable_vm.py`: its regex and name layers (+ deid_roberta, GLiNER), every span replaced by `[TYPE]` and logged with its reason in `output.decisions.jsonl`. The VM code has no replacement step, so the adapter adds only that. Fingerprint: the adapter, the VM's `detect/` and `common.py`, and the word list | Own venv in `../models/fable-deps/.venv` with the VM's versions (`transformers==5.16.1 gliner==0.2.29 regex protobuf sentencepiece spacy torch`; spaCy gives the name layer its stop words). HF: `obi/deid_roberta_i2b2`, `urchade/gliner_multi_pii-v1` and the tokenizer files of `microsoft/mdeberta-v3-base`. The name layer's word list is Debian's `wamerican-large` (`../models/fable-deps/american-english-large`) |
| `ketai` | Disabled until it lands | Set `cmd` or `precomputed`, then `enabled = true` |
| `identity`, `oracle` | References: input unchanged (privacy 0, utility 1), and gold PII replaced by `[CATEGORY]` (reads the gold, so its fingerprint includes `{labels}`) | none |

To add a model, add a table to `models.toml` (with a `fingerprint` for its code) and run
`python3 -I eval all --models <name>`.

## Layout

```
eval/
  __main__.py  CLI            common.py  paths, JSONL, matching    models.toml  model registry
  suites.py  run.py  score.py  report.py  metrics.py  icd.py
  adapters/    baselines.py (identity, oracle), fable_vm.py
  coders.toml  ICD coder registry   coders/  base, codes, bootstrap, ollama (LLM coders), knn, lexical, command
  bonus/       bonus tasks
  gold/        taxonomy.py (labels, levels), merge_aci_gold.py (`eval gold`), audit_gold.py
  tests/       pytest suite (hand-made fixtures; test_gold.py's two cross-checks read data/eval and skip without it)
  docs/        this file, CORPUS-DRIFT.md, GOLD.md
```

## Tests

pytest is the one development dependency (the harness itself is standard library only). With `uv`, from the repo
root, without installing anything into the system Python:

```bash
uv run --no-project --with pytest python3 -I -m pytest eval/tests -q
```

The run tests start and kill small processes and read `/proc`, so they are skipped outside Linux.
