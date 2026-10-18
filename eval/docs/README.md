# Evaluation pipeline

One harness that runs every team model on the same data and scores it with the same code. Models are black boxes:
a model reads a challenge-format JSONL and writes an anonymized one. The scorer only compares input and output, so a
rule system, an NER stack and an LLM rewriter are measured the same way.

Run everything from the repo root (`neuralwave/`):

```bash
python3 -I eval all                                   # every enabled model x every suite, then the report
python3 -I eval all --models ad_night anonymous --suites aci_eval
python3 -I eval score --icd llm_judge knn "cmd:python3 coder.py --input {input} --output {output}"
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
| `report` | `report.py` | scores, `runs/*/*/run.json`, attacks, `phi/`, `drift.json`, `suite.json`, converter manifests, merged gold stats, `models.toml`, `coders.toml` | `data/eval/report.md`, `report.json`, `results.csv`, `results_headline.csv`, `tradeoff.svg`, `tradeoff_adversarial.svg`, `report_discord.md` | Comparison tables, tradeoffs, glossary, sources and CSV exports, aggregate numbers only. See [Report and exports](#report-and-exports) |
| `all` | `__main__.py` | | | The steps in order: `suites`, `run`, `score`, `report`, plus the optional `gold`, `attack` and `drift` once they exist |
| `gold` | `gold/merge_aci_gold.py` | the harness, VM and ketai ACI golds | `data/eval/gold/aci_eval/` | Merges the three ACI-eval golds into one with votes (`merged.jsonl`, `merged_stats.json`, `merged_review.md`); `--audit` also writes `audit.md`/`audit.json` (`gold/audit_gold.py`, also runnable alone). Deterministic, so `all` runs it first every time. See [ACI gold and privacy levels](#aci-gold-and-privacy-levels) |
| `attack` | `bonus/attack.py` | runs, suites | `data/eval/attacks/<model>/<suite>.json` | Re-identification attacks A1–A5 on each output, with the original text as positive control. See [Adversarial attacks](#adversarial-attacks-bonus) |
| `robustness` | `bonus/robustness.py` | `data/converted/robust_*`, scores, `drift.json` | the robust sets and suites | Builds the three robust suites when missing and prints the per-model robustness table. See [Robustness](#robustness-bonus) |
| `drift` | `bonus/drift.py` | suites, `data/converted/aci_bench` | `data/eval/drift.json` | Lexical, style and structural drift of every suite from the 140 ACI train encounters, and the `complexity_rank` of the suites. See [Corpus drift](#corpus-drift-bonus) |
| `judge` | `coders/agent.py` | runs, suites | `<container>/judge/<suite>/` (export), `data/eval/judge/<suite>/key.json`, the `llm_judge` ICD cache (import) | The LLM judge by agents: `export` writes the notes in batches with a coding prompt, `import` caches the agents' codes. See [LLM judge by agents](#llm-judge-by-agents) |
| `adversary` | `bonus/adversary.py` | runs, suites | `<container>/adversary/<suite>/` (export), `data/eval/adversary/<suite>/key.json`, attack A6 (import) | The LLM agent adversary: `export` writes blinded sets and one prompt per set for agents, `import` scores their answers as attack A6. See [Agent adversary](#agent-adversary-a6) |
| `phi` | `bonus/phi_privacy.py` | runs, suites | `data/eval/phi/<model>/<suite>.json` | Privacy before/after by an independent PII detector (Phi-3 PII), per category. Not part of `all`: run it last, alone in memory. See [Independent privacy check](#independent-privacy-check-phi) |
| `probe` | `bonus/probe.py` | `models.toml`, a suite | `data/eval/probe/<model>/<suite>.json` | VRAM with the Ollama server included, and latency, of every runnable model on a few records. Not part of `all`: run it alone on the GPU. See [VRAM and latency probe](#vram-and-latency-probe) |

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
  changed model reruns even when its command is the same. While a model runs, its
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
| `report.md` | Two dataset blocks. **ACI** (`aci_eval`): the headline (privacy at the consensus, direct and entity level and on the harness gold alone, medical concept recall, F1-Score ICD-10 code prediction with the LLM judge and the decision tree, each with Δ F1 (anonymized minus original), the LLM judge's agreement, proportion of recovered information by the LLM adversary and the max over attacks, explained share, runtime, peak RAM, VRAM within 4 GB; `identity` last, as the reference), the gold-privacy and the adversarial-privacy Pareto fronts, privacy levels, votes and sources, privacy by category and field, the Phi before/after check, F1 of every coder (ACI only; `!` = not valid for comparison), the main suites, attacks, and the with/without-codes ablation. **Kaggle**: privacy and ICD-10-CM terms per suite, ICD agreement per coder (no F1), utility with gold, privacy by language, Phi. Then robustness, suite drift and efficiency, a **Glossary** of every column (definition and task criterion) and the **Sources** (datasets with origin and license, coders, adversary, detector). Column names follow the task PDF and come from one registry (`report.COLUMNS`), so a metric has the same header in every table |
| `report.json` | The same numbers (`headline`, `tradeoff`, `tradeoff_adversarial`, `icd`, `nocodes`, `attacks`, `phi`, `robustness`, `efficiency`, `columns`, ...) and `exports`, the paths of the five files below |
| `results.csv` | Long format, one number per row: `model, suite, category, metric, value, n, ci_low, ci_high, coder, level, notes`. `category` is one of `privacy`, `privacy_by_category`, `privacy_by_field`, `privacy_by_language`, `keep`, `concepts`, `edits`, `terms`, `icd`, `attack`, `robustness`, `drift`, `efficiency`, `explain`, `validity`. `level` is the privacy level (`consensus` = the headline `privacy.removed`, `union`, `direct`, `gold:<source>` for one gold alone, `union` also for `votes:N`/`src:X`) or an attack's `cat:<category>`. ICD metrics are `<arm>.<code or category>.<stat>` (arms `original`, `anonymized`, `agreement`), `f1_delta` and `category.f1_delta`. Every number of `report.md` is a row (also the breakdowns it leaves out), at full stored precision; suite-level rows (drift, gold counts) have an empty model. UTF-8 with BOM and commas, so Excel and LibreOffice open it without prompts |
| `results_headline.csv` | One row per model, one column per headline metric (keys of `report.HEADLINE`), plus `reference` and `pareto` (1/0) |
| `tradeoff.svg` | Consensus privacy (x) vs the primary coder's ICD F1 on the anonymized text (y; ICD concepts kept, said so in the axis title, while no real model has a primary-coder result). Filled = Pareto-optimal, hollow = dominated, square = `identity`. Explicit background, light and dark colors, no external fonts |
| `tradeoff_adversarial.svg` | The same front with x = 1 − the LLM adversary's proportion of recovered information (attack A4): privacy measured by what an attacker gets back instead of by the gold |
| `report_discord.md` | The headline in one short table plus a few bullets, for a chat |

A model is Pareto-optimal when no other real model (references excluded) is at least as good on both consensus privacy
and ICD F1 on the anonymized text and better on one; there is no combined score, because the organizers weigh the criteria
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
the A+D team's annotators), the VM teacher gold (`vm-backup/neuralwave/annotations/eval`, made for the anonymous model)
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
contradictions, regex candidates no item covers, single votes).

## Metrics

Summary below; **[METRICS.md](METRICS.md) has one row per reported number** (privacy levels, ICD sub-metrics, attacks,
efficiency, bonus metrics, Pareto), how each is computed, its key in the score files, and which organizer criterion
it answers. The drift metrics are in [CORPUS-DRIFT.md](CORPUS-DRIFT.md).

| Metric | Definition |
|---|---|
| valid | Same records, ids, order and keys, and `icd10_codes` byte-identical (hard constraint). One extra key is allowed: `explanations`, which must be a list |
| privacy | Share of gold PII occurrences gone from the output (case-insensitive whole words; a number does not match inside a larger number, so `59` is not found in `120/59`). A multi-word item also leaks when one of its distinctive tokens survives (`[PATIENT] Campbell` leaks `Andrew Campbell`); tokens under 4 characters are never distinctive, so `[PATIENT] Li` does not leak `Wei Li`. Also reported by risk, category, field and language. On `aci_eval`: the consensus level, with the direct and union levels, entity-level removal and each source gold beside it ([levels](#aci-gold-and-privacy-levels)) |
| high-risk | The same for names, DOB, dates, ages, IDs and contacts |
| keep traps | Share of must-survive gold strings still present: eponyms (`Parkinson's`), `NOT_A_NAME` traps, clinical look-alike numbers (`3/6` murmur), code evidence |
| non-PII kept, edit precision | Token alignment (difflib) of input vs output on exhaustive gold: share of non-PII tokens untouched, and share of changed tokens that were gold PII |
| concepts, ICD concepts | Share of gold medical-concept occurrences kept (all, and those flagged relevant to the encounter's codes) |
| terms | Share of ICD-10-CM description words kept (independent vocabulary, for suites without concept gold) |
| ICD F1, agreement | Local stand-in coders (`coders.toml`): micro F1 vs the gold codes on the original and the anonymized text, at the code and the 3-character category level, the paired-bootstrap F1 change Δ F1 (no ratio: a ratio of two small F1s exaggerates noise and can exceed 1), and anonymized-vs-original agreement (the damage done by anonymization alone). See [ICD coders](#icd-coders) |
| ms/enc, RAM, VRAM | Wall time per record (process start included), peak RSS and nvidia-smi memory of the whole process tree. Budget: 4 GB VRAM |
| explained | Share of changed original tokens (the same token alignment as edit precision, on every suite) covered by a logged decision with a non-empty reason. See [Explainability](#explainability-and-no-codes-bonus) |

### ICD coders

The organizers code the anonymized notes with their own hidden coder. `icd.py` stands in with the coders of
`coders.toml` (code in `coders/`): each one codes the original and the anonymized text of every record that has gold
codes, and both are scored against the gold. A coder never sees the gold codes.

| Coder | What | Reads | Output | Suites |
|---|---|---|---|---|
| `llm_judge` (**primary**, the LLM judge) | Claude Sonnet 5.5 agents, from a prompt: `kind = "agent"`, its predictions come only from `eval judge import` (see [LLM judge by agents](#llm-judge-by-agents)) | note | set | `aci_eval` |
| `tree` | The decision-tree coder of `../decision-models` (`icd_rules.py`, scorer `tree`: ICD-10 lexicon candidates, a depth-2 tree fit on the 140 ACI train encounters), through `adapters/tree_coder.py` (`kind = "command"`, numpy, about 1 s per suite). Its own headline is the 3-character category F1 (0.689 on the original `aci_eval`, the package's number), so the report reads its category level | dialogue + note | set | `aci_eval` (F1); the Kaggle medical suites (agreement only) |
| `knn` | TF-IDF k-nearest neighbours (k = 7) over the 140 ACI train encounters, clinical words only. Supervised: only the ACI train codes | dialogue + note | ranked | `aci_eval`, `aci_dev` |
| `lexical` | Zero-shot: the suite + ACI label space ranked by idf-weighted description words found in the text | dialogue + note | ranked | every suite with codes |
| `paper` | Disabled slot (`kind = "command"`) for the full paper coder | | | |

`score --icd` takes these names (default: every enabled coder with `default = true`: `llm_judge tree knn lexical`)
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
- On a suite without any gold code an unranked coder (`tree`, the LLM coders) is scored on the `agreement` arm alone
  (`icd.agreement_only`), over every record: that is the Kaggle ICD table. A ranked coder has no k there and is skipped.
- `f1_delta`: anonymized minus original code-level micro F1 with a paired bootstrap CI (1000 resamples of the
  encounters, seed 0; daniel-eval's procedure, which it reproduces exactly). `f1_delta_category`: the same at the
  category level.
- A failed prediction (timeout, unfinished generation, malformed answer, prompt over the context budget) counts as an
  empty set, is counted in `errors` and is never cached; `valid_for_comparison` is then false. When the whole coder
  fails (Ollama unreachable, three consecutive failed predictions) its entry is `{"error": ...}` and scoring goes on.

The LLM coders talk to `OLLAMA_HOST` (default `127.0.0.1:11434`) with proxies and redirects disabled. A server that
is not on this machine is allowed but recorded as `endpoint`. A prompt over the budget (UTF-8 bytes + 512 output
tokens + 1024 reserved ≤ `num_ctx`) is rejected, never truncated. `run` unloads every loaded Ollama model before it
measures the models, so a coder left loaded by `score` does not inflate their RAM or VRAM.

### LLM judge by agents

The LLM judge (`llm_judge`, `kind = "agent"`) is Claude Sonnet 5.5 run as agents from a prompt, not through an API.
`coders/agent.py`:

1. `python3 -I eval judge export` collects every note still to code on `aci_eval` (the originals and every finished
   run's output, each distinct text once) and writes batches of about 40 to `<container>/judge/aci_eval/`, outside the
   repo: `batch_XX.jsonl` with only `record_id` and `note`, shuffled, never two versions of one encounter in a batch
   (an agent cannot carry codes over from an original to its anonymized version). `PROMPT.md` is the prompt,
   `prompts/batch_XX.md` the one for each batch, `agents/batch_XX.md` what each agent reads (`AGENT.md`, how to read
   the notes one at a time up to an `=== END` marker, then the prompt) and `ORDER.md` the message that starts it.
   The key stays in `data/eval/judge/aci_eval/key.json`.
2. One Sonnet agent per batch, started with `ORDER.md`, writes `answers/batch_XX.jsonl`: one
   `{"record_id", "codes": [...]}` per line, and returns the last words of each note so the runner can check that it
   read every note to the end. `python3 -I eval judge status` counts the answers.
3. `python3 -I eval judge import` writes each answer into the coder's ICD cache (`data/eval/cache/icd/llm_judge/`),
   keyed like every prediction by the coder config, the encounter and the note; then `python3 -I eval score` scores it
   like any coder (F1 original and anonymized, Δ F1, agreement).

**Repeat passes** cut the agents' own variance: two agents coding the same note disagree on part of its codes.
`python3 -I eval judge export --pass 2 --batch-size 10` writes every note again, cached or not, in new batches (seed =
pass − 1) to `<container>/judge/aci_eval_pass2/` (key `key_pass2.json`); pass 3 likewise. `import` reads every pass
and caches, per note, the codes that more than half of the passes that answered it gave (2 of 3; with two passes,
both), and writes `data/eval/judge/aci_eval/passes.json`: the agreement of every two passes (micro F1 of one pass's
codes against the other's, code and category level), a direct measure of the agent variance.

A note without an answer is a failed prediction (`no_agent_answer`, counted in `errors`, never cached); while nothing
is imported the coder's entry is an error. The next `export` skips notes already cached, so a new model only adds its
own new notes. To give those notes repeat passes too, export them as new passes with `--new-only` (only the notes not
cached yet), all of them before importing: `judge export --pass 4 --new-only`, then `--pass 5` and `--pass 6` likewise,
one agent per batch, then `import`. Each new note gets the majority of its 3 passes; the notes passes 1-3 coded keep
their vote, and `passes.json` compares only passes that coded the same notes. A new export refuses to run over existing answers unless `--force`. Keep every pass's folder and key:
`import` recomputes the vote from all of them.

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
empty when nothing changed (`identity`) and 0 when text changed without any log.

| Producer | Sidecar |
|---|---|
| A+D (`ad_*`) | Written by the model as `<output stem>.decisions.jsonl`: `action` REPLACE/GENERALIZE/KEEP and `explanation`, no `text` (offsets index the original). `explain.py` normalizes it. `ad_night_submitted` gets the sidecar delivered beside the precomputed file, which `run` copies along |
| anonymous (`anonymous_rules`, `anonymous`) | Written by `adapters/anonymous_vm.py`: one `remove` row per replaced merged span, the reason joins each member's detector reason, layer and confidence, `source` the layers |
| Ketai's generalize log | Rows with `text`, `type`, `removed`/`generalized`/`retained` and a reason but no offsets: each covers every occurrence of its text |

Run `python3 -I eval/bonus/explain.py [--models ...] [--suites ...]` to print the scores of finished runs.

**No ICD-10 codes at inference** (`bonus/nocodes.py`). A `blind_codes = true` entry is paired with its base model:
the entry named in its `base` key, else its name without `_nocodes` (`ad_night_nocodes` with `ad_night`).
`nocodes.rows(scores, registry, suite)` gives per pair the base and no-codes values and the delta of privacy,
high-risk, keep traps, concepts, ICD concepts and each coder's ICD F1 on the anonymized text (also rows of
`results.csv`). The report's ablation table puts every pair side by side over the main-task metrics of the headline
(privacy, direct, concept recall, F1 of the LLM judge and the tree, the LLM adversary's recovery) and
names the models without a twin. Pairs now: `ad_night`, `term_generalizer` (its delivered `_noicd_ablation` file).
`zeroshot` and `anonymous` never read the codes, so a twin would be identical.
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
| A1 NER residual scan | Independent detectors: deid_roberta (i2b2), GLiNER multi-PII and the VM regex layer, run by `bonus/_ner_worker.py` with the anonymous venv | gold items (not low risk) covered by a detected span; spans holding a placeholder don't count |
| A2 roster dictionary | Knows every person name of the suite's gold plus the first names of `nam_dict.txt` (Kaggle `krist0phersmith/pii-real-names-dataset`) | name tokens of gold names still present in their field, more often than the token occurs outside the name |
| A3 rare words | Encounter-unique tokens that are not clinical (ICD-10-CM words, RxNorm), not ordinary English, not calendar words, and are a proper noun of the word list or a known first name | such tokens still present in their encounter (gold-free: `by_category` is the gold category holding the token, or `UNLABELLED`) |
| A4 LLM inference | Ollama `qwen3.5:2b-q4_K_M` (the default `--llm`, schema-constrained JSON) guesses the patient's name, age, place, employer or job. `--llm claude-sonnet-5-5` runs it through `bonus/_claude_worker.py` instead (official SDK via `uv run --with anthropic`, structured output, needs `ANTHROPIC_API_KEY`, sends the record text to the API) | guesses matching the record's gold (any name token, age ±1 year, gold place or job inside the guess); a refusal counts in `errors` and as no guess |
| A6 LLM agent adversary | LLM agents, one per blinded model output, prompted as red-team auditors (`eval adversary`, below). The report's headline adversary | the same rule as A4 |
| A5 record linkage | Holds the original records; TF-IDF nearest neighbour (scikit-learn's `sublinear_tf` formula, standard library) | records linked to their own original |

Gold is the suite's `pii` at the consensus level (direct identifiers or 2+ votes) when the items carry votes (the
merged ACI gold), else all items; the file records `gold_level`. A5 is an upper bound by design: clinical content is
kept, so whoever holds the originals re-links almost every record. It bounds the threat model (anonymization hides
who the patient is, not which record it was) and is excluded from `max_recovered`. A1's detectors are the ones the
anonymous models use, so for them A1 is not independent: what their detectors miss, A1 misses too (on `aci_eval` A1
recovers 1.8% from `anonymous` and 13.6% from `ad_night`, whose own detectors differ).

Cost (measured on `aci_eval`, 67 encounters, RTX 4050):

| Attack | Runs by default | Cost |
|---|---|---|
| A2, A3, A5 | yes (what `all` runs) | standard library, CPU only: about 2.5 s per model (20 s for the 8 models with a run) |
| A1 | only when named | loads two transformer models in the anonymous venv and scans about one text (a dialogue or a note) per second: ~10 min for 4 models (`--device auto` uses the GPU when 2.5 GB are free, else the CPU with 4 threads, an order of magnitude slower) |
| A4 | only when named | one LLM call per record and version: 67 calls for the control (shared by every model; `identity` reuses it) plus 67 per model, sent as one batch per model. With Sonnet about 3,500 input tokens per call (roughly $0.6 per model at $2 / $10 per million tokens). Locally (Ollama) 1.4 s per call on an idle GPU |

A1 and A4 cache each result by text in `data/eval/cache/attacks/` (`ner/`, `llm/`; written through a temporary file,
so a killed run loses nothing and the next run resumes where it stopped). The control and unchanged outputs are
computed once, and a later run without A1 or A4 keeps their earlier results while the model output, the suite and
`--limit` are unchanged. A model without a run, a failed run, an unbuilt suite or one without PII gold is skipped with
a one-line reason.

### Agent adversary (A6)

The headline adversary runs as a prompt, not as an API call: LLM agents attack the outputs and the harness scores
their guesses. `bonus/adversary.py`:

1. `python3 -I eval adversary export [--models ...]` writes the package to `<container>/adversary/<suite>/`, outside
   the repo, so an agent working there never sits next to the gold: one `set_XX.jsonl` per distinct model output
   plus the original text (the positive control), each holding only `record_id`, `dialogue` and `note`, records
   shuffled and renamed per set and the sets numbered at random (identical outputs share a set: `ad_night_submitted`
   with `ad_night`, `identity` with the control). `PROMPT.md` is the prompt, `prompts/set_XX.md` the one to give each
   agent. The key (which set is which model, which record is which encounter) stays in
   `data/eval/adversary/<suite>/key.json`.
2. Run one agent per set with its working directory set to the package and access to that folder only (an agent that
   reads two sets, the key or the gold breaks the blinding). Each writes `answers/set_XX.jsonl`, one line per record:
   `record_id`, `first_name`, `last_name`, `age`, `city_or_state`, `employer_or_job`, `evidence`.
   `python3 -I eval adversary status` counts the answers per set.
3. `python3 -I eval adversary import --attacker <agent model>` scores every answered set like A4 against the gold
   (consensus level on `aci_eval`), with the control set's score as `control`, and writes attack A6 into each
   model's attack file. A missing or malformed answer counts as no guess (`missing`, `malformed`). A model whose
   output changed since the export is skipped. `eval attack` never runs or overwrites A6. A new export renumbers the
   sets, so it refuses to run over existing answers unless `--force` (which deletes them).

### Independent privacy check (Phi)

`python3 -I eval phi [--models ...] [--suites aci_eval ...] [--model OLLAMA_MODEL] [--limit N]`
(`bonus/phi_privacy.py`) measures privacy without the gold: the fine-tuned `ab-ai/PII-Model-Phi3-Mini` (Phi-3-mini
PII extractor, MIT; Q4_K_M GGUF of `mradermacher/PII-Model-Phi3-Mini-GGUF`, 2.4 GB:
`ollama pull hf.co/mradermacher/PII-Model-Phi3-Mini-GGUF:Q4_K_M`) reads the original text (**before**) and the
model's output (**after**) in chunks of 2,000 characters with the model card's prompt. Its labels map to the gold
categories (`LABELS`; sex, gender, height, amounts and title prefixes are dropped, placeholders ignored), and
`removed` = 1 − after / before, in total and per category. It is evaluation only, never a model row. It unloads every
other Ollama model first and the detector at the end, so it runs alone in memory: run it **last**. Answers are cached
by chunk text (`data/eval/cache/phi/`), so the originals are scanned once. On some chunks the detector loops (the same labels repeated up to its output limit, so the answer never closes): its complete `label: [values]` pairs are kept and counted in `salvaged`; a chunk that still cannot be read is an `error`, never asked twice in a run, so an unchanged chunk counts the same before and after. About 400 calls for the `aci_eval`
originals and up to 400 per model.

### VRAM and latency probe

`python3 -I eval probe [--models ...] [--suite aci_eval] [--records 10]` (`bonus/probe.py`) answers what `run` cannot:
`run` measures a model's own process tree over a whole suite, so a model whose LLM runs in the shared Ollama server
(`zeroshot`, `unified_harness`) shows 0 MB of VRAM, and its wall time includes loading. The probe runs every model
with a command twice, on the suite's first record and on its first N records (blinded for a `blind_codes` model), each
time after every Ollama model is unloaded, and samples every 0.2 s the RSS and nvidia-smi memory of the model's tree
and, apart, of the Ollama server's tree. `latency_ms_per_encounter` = (wall(N) − wall(1)) / (N − 1), the time a record
adds once the model is loaded; `startup_s` = wall(1) − latency; `total_peak_vram_mb` = the peak of the two VRAMs summed
per sample, against the 4 GB budget. It takes a few minutes for every model. Run it alone on the GPU, with only the
Ollama server the models use (the probe counts every `ollama serve`): another process's load would slow the models
down. The report shows it in the headline (latency, peak VRAM incl. Ollama) and in its own table.

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
| `blind_codes` | `true` hides `icd10_codes` from the model (bonus task) and restores them before scoring; `base` names the model it is compared with (default: the name without `_nocodes`) |
| `reads_codes` | `false`: the model never reads `icd10_codes` (`anonymous_rules`, `anonymous`, `zeroshot`), so the ablation says a twin would be identical instead of listing it as missing |
| `suites`, `enabled` | Only these suites; `false` lists the model as pending and never runs it |

A model runs in its own session, so the harness kills its tree itself on Ctrl-C, SIGTERM and SIGHUP (a SIGKILLed
harness cannot). A command that cannot start (missing executable or `cwd`) is recorded as exit code 127, and the
other models still run.

| Entry | What runs | Setup |
|---|---|---|
| `ad_rules`, `ad_night`, `ad_night_nocodes` | A+D night model (`neural_wave` 93b5eea, Davide's latest zip) in `../models/ad-night/repo`: rules only, and their submission config (rules + deid_roberta + GLiNER), with and without codes. Fingerprint: its `src/` code and lexicons | `uv sync --all-extras` there, then download `obi/deid_roberta_i2b2` and `knowledgator/gliner-pii-base-v1.0` (runs offline) |
| `anonymous_rules`, `anonymous` | Anonymous model (`../vm-backup/neuralwave`) through `adapters/anonymous_vm.py`: its regex and name layers (+ deid_roberta, GLiNER), every span replaced by `[TYPE]` and logged with its reason in `output.decisions.jsonl`. The VM code has no replacement step, so the adapter adds only that. Fingerprint: the adapter, the VM's `detect/` and `common.py`, and the word list | Own venv in `../models/anonymous-deps/.venv` with the VM's versions (`transformers==5.16.1 gliner==0.2.29 regex protobuf sentencepiece spacy torch`; spaCy gives the name layer its stop words). HF: `obi/deid_roberta_i2b2`, `urchade/gliner_multi_pii-v1` and the tokenizer files of `microsoft/mdeberta-v3-base`. The name layer's word list is Debian's `wamerican-large` (`../models/anonymous-deps/american-english-large`) |
| `unified_harness_rules`, `unified_harness`, `unified_harness_nocodes` | unified-harness, this repo's anonymizer (`src/unified_harness`), through `adapters/unified_harness.py`: rules only, and its default config (`llm` mode: rules + `qwen3.5:2b-q4_K_M`, `privacy` preset with F0 and F2), with and without codes. The adapter imports the package in-process (its anonymization path is standard library only) with `UNIFIED_HARNESS_HOME` in a fresh folder next to the output, so the LLM reply cache starts cold on every run and the repo's `outputs/` are never touched. It copies the decisions (`output.decisions.jsonl`) and the summary (`output.summary.json`, which holds the Ollama VRAM the harness cannot see in the process tree). The model publishes nothing when one record fails its own checks, so such a suite is a failed run. Fingerprint: the adapter and `src/unified_harness/` (code, lexicons, `models.toml`) | Python 3.12+. The LLM rows: Ollama with `qwen3.5:2b-q4_K_M` at the digest pinned in `src/unified_harness/models.toml` (the run checks it) |
| `unified_harness_submitted` | The delivered `aci_eval` file (`../anonymized-2026-10-18-unified-harness.jsonl`): `precomputed`, checks the `unified_harness` run against it | none |
| `term_generalizer`, `term_generalizer_nocodes` | The term-generalizer's delivered `aci_eval` outputs (`../vllm4gb-pilot-ppo_generalize.jsonl` and its `_noicd_ablation` twin, made without the codes): `precomputed`, so the harness scores them without running the model | none |
| `zeroshot` | Zero-shot baseline: `../zeroshot/anonymize.py --mode extract` through `adapters/zeroshot_llm.py`. Ollama `qwen3.5:2b-q4_K_M` returns the PII spans of each 2,000-character chunk as JSON (task definition and label set, no examples), each span becomes `[TYPE]`. It never reads the codes. The adapter keeps the work folder next to the output, so an interrupted run resumes, and fails rather than deliver missing records | Ollama with the model (see the ICD coders) |
| `ketai` | Disabled until it lands | Set `cmd` or `precomputed`, then `enabled = true` |
| `identity` | Reference: input unchanged (privacy 0, utility 1) | none |

To add a model, add a table to `models.toml` (with a `fingerprint` for its code) and run
`python3 -I eval all --models <name>`. To score a delivered JSONL without running anything, give it a `precomputed`
entry (`{suite}` in the path for one file per suite, `suites = ["aci_eval"]` for one file) and run
`python3 -I eval run --models <name> --suites aci_eval`, then `score` and `report`.

## Layout

```
eval/
  __main__.py  CLI            common.py  paths, JSONL, matching    models.toml  model registry
  suites.py  run.py  score.py  report.py  metrics.py  icd.py
  adapters/    baselines.py (identity), anonymous_vm.py, zeroshot_llm.py, unified_harness.py, tree_coder.py (ICD coder)
  coders.toml  ICD coder registry   coders/  base, codes, bootstrap, ollama (LLM coders), knn, lexical, command
  bonus/       bonus tasks
  gold/        taxonomy.py (labels, levels), merge_aci_gold.py (`eval gold`), audit_gold.py
  tests/       pytest suite (hand-made fixtures; test_gold.py's two cross-checks read data/eval and skip without it)
  docs/        this file, METRICS.md, CORPUS-DRIFT.md, GOLD.md
```

## Tests

pytest is the one development dependency (the harness itself is standard library only). With `uv`, from the repo
root, without installing anything into the system Python:

```bash
uv run --no-project --with pytest python3 -I -m pytest eval/tests -q                  # all 311, about 25 s
uv run --no-project --with pytest python3 -I -m pytest eval/tests/test_gold.py -q     # one file
uv run --no-project --with pytest python3 -I -m pytest eval/tests -q -k privacy       # by name
```

The tests use hand-made fixtures with answers worked out by hand, so they need neither `data/` nor `models/` nor a
running Ollama. `conftest.py` holds the shared 3-record suite and replaces the English word list and the ICD-10-CM
vocabulary with fixed ones. A few tests use real data and skip when it is missing: the two A+D cross-checks in
`test_gold.py`. `test_run.py` starts and kills small processes and reads `/proc`, so it is skipped outside Linux.

| File | Tests | What it checks |
|---|---:|---|
| `test_validity.py` | 12 | The hard output contract: same records, ids, order and keys; `icd10_codes` byte-identical (reordering counts as a change); text fields are strings; `explanations` (a list) is the only extra key allowed |
| `test_metrics.py` | 24 | Privacy on the 3-record suite: full removal, identity, partial leaks, a surviving distinctive token leaking a name (`[PATIENT] Campbell`), tokens under 4 characters not distinctive (`[PATIENT] Li`), the number boundary (`59` not inside `120/59`), tokens that also occur outside the item. Keep and concept survival, edits (exact, over- and under-redaction, exhaustive gold only), clinical terms, placeholder rate |
| `test_gold.py` | 18 | The merged ACI gold: label mapping to common categories, DATE → DOB/YEAR rules, majority vote and tie priority, longest span canonical with nested votes, nested items surviving only with standalone occurrences, keep/PII conflicts, concept union, union/consensus/direct filters, privacy levels and per-source scores, deterministic output, the audit, the suite built from the merged gold. With data: `ad_night` harness gold = 0.8388 |
| `test_codes.py` | 30 | ICD-10-CM normalization (ported from daniel-eval): one canonical spelling per code, invalid codes rejected, exact (not hierarchical) matching, first-seen order, 3-character categories, gold codes undotted or as dicts |
| `test_bootstrap.py` | 8 | Set scoring and the paired bootstrap (ported from daniel-eval): empty denominators, micro vs mean-per-note F1 with hallucinated codes, failed predictions scored as empty, identical arms giving Δ = 0 with CI [0, 0], numbers pinned to daniel-eval's own output on the same fixture |
| `test_coders.py` | 47 | The coders against a mocked HTTP layer: request body identical to daniel-eval's, context guard, malformed and unexpected answers, transport failures, loopback guard, no proxy or redirect, retries, three consecutive failures failing the coder, unreachable server, model not installed, unload; the kNN, lexical and command coders (contract, missing records, timeout, cannot start) |
| `test_icd.py` | 14 | `icd.score`: the exact JSON shape, ranked coders cut at k = number of gold codes, unranked sets scored whole, category level, identical arms, failed predictions counted and never cached, invalid codes counted, coder-level failure as an `error` entry, cache compatible with the old plain lists and robust to a corrupt file, the `coders.toml` registry and defaults, an end-to-end LLM coder with a fake server, agreement alone on a suite without gold codes |
| `test_attack.py` | 21 | The attacks on the 3-record fixture: identity output recovers exactly what the original-text control recovers, a fully cleaned output leaves A2 nothing, hand-computed A3 (rare tokens) and A5 (record linkage), A1 with a fake NER worker, A4 with a fake LLM (age within ±1) and with a fake Claude batch (one batch, cached guesses, refusals counted, no key), the worker's closed schema, the consensus gold level, A5 left out of the max, skips, the per-record cache surviving a cut file |
| `test_robustness.py` | 6 | The robust-set converters on tiny inputs (discharge layouts and gold, seeding per row, PII injected into medconv with labels, synthetic documents with nested items folded) and the robustness table with drift ranks |
| `test_drift.py` | 32 | Corpus drift: zero on identical corpora, saturated on disjoint vocabularies, known small-case values for JSD, MTLD, Yule's K, entropy, Distinct-n, repetition, the syllable heuristic and Flesch-Kincaid, sentence splitting, structural ratios (bullets, key-value lines, line breaks), the `drift.json` contract |
| `test_explain.py` | 18 | Explainability: token alignment, full and partial coverage of changed tokens, decisions with an empty reason not counted, keep decisions, no sidecar vs inline `explanations`, A+D sidecar normalization and scores, offsets checked against the original text, a sidecar with a bad encoding, the anonymous adapter's decisions, the sidecar copied for precomputed models |
| `test_nocodes.py` | 3 | No-codes pairing (by `_nocodes` suffix or `base` key) and the paired values and deltas |
| `test_report.py` | 9 | Report on a synthetic `data/eval` tree: every headline cell is also a row in `results.csv` (the tree at its category level), Pareto flags of both fronts, missing data and errors shown as `–`/markers without crashing, the ablation table, CSV BOM and header, well-formed SVG, the Discord summary, the glossary (every column) and the sources |
| `test_judge.py` | 6 | The LLM judge by agents: the export (each distinct note once, never two versions of an encounter in a batch, failed runs ignored, the key kept outside the package, each agent's file = reading protocol + prompt), nothing imported = a coder error, the import into the cache scored by `icd.score`, a missing answer = a failed prediction, cached notes not exported again; repeat passes (every note re-exported in new batches, majority of 3, both of 2, pass agreement in `passes.json`); `--new-only` passes for notes that came later (only those exported, the earlier votes kept) |
| `test_adversary.py` | 4 | The agent adversary: the blinded export (only record ids and text, shared sets, the key kept outside the package, filled prompts), the refusal to overwrite answers, the import scored like A4 with the original-text set as control (missing and malformed answers), A6 kept by automated attack runs, a changed output skipped |
| `test_probe.py` | 3 | The VRAM and latency probe: latency and startup from the one-record and N-record runs (a failed run or noise leaves them empty), the budget flag, the Ollama server's processes counted apart from the model's, a probe end to end with a fake model (blinded input, the per-model file) |
| `test_phi.py` | 9 | The Phi check: chunking at line breaks, the detector's dict answers (JSON or Python literal), the label map, before/after counts with a fake detector, the cache, unparseable answers counted as errors, answers cut by a repetition loop salvaged, a failed chunk never asked twice in a run (identity stays at removed = 0) |
| `test_adapters.py` | 3 | The zero-shot runner (copies a complete output, refuses a partial one), the decision-tree coder's output contract (skipped without `../decision-models` or numpy) and the unified-harness runner in rules mode (records, sidecar and summary copied, every change explained by the sidecar, no work folder left, nothing published for an input the model refuses) |
| `test_run.py` | 15 | The run cache's code fingerprint (globs, renames, location independence, relative paths) and the process runner: timeout and SIGTERM/SIGHUP to the harness kill the model's whole process tree, a command that cannot start is a failed run, an exit at the deadline is not a timeout, wall time ends at the model's exit, an interrupted run is never cached |
| `test_suites.py` | 9 | Suite caching: built once and reused, rebuilt when the source, the definition, the build code or a suite file changes, byte-identical rebuilds, a missing source keeps the old suite, `--if-stale`, the VM annotation dir kept across rebuilds |
| `test_main.py` | 8 | The command line: every command listed in `--help`, a missing module exits 2, every `all` step is a command, a crashing step does not stop `all`, stage modules also run as files |
| `test_common.py` | 15 | Package paths and `OLLAMA_HOST` parsing (host, host:port, URLs, IPv6, empty host, bad port) |

Not covered by the tests, checked by hand instead and recorded in `EVAL_MERGE_LOG.md`: real model runs (byte-identical
to the old harness), the live Ollama coder (identical to daniel-eval's CLI on the `identity` pair), and the real NER
worker of attack A1.
