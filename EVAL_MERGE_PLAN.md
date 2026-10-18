# Plan: merge the eval harnesses into `neuralwave/eval/`

## Dispatch sheet for the executing subagents

Run from `/home/criscola/IdeaProjects/neuralwave-container/neuralwave/`. Read `AGENTS.md` first (no AI attribution anywhere, never commit `data/eval/` or `data/kaggle/`). Every WP: stdlib-only harness code, `python3 -I`, tests under `eval/tests/`, update `docs/README.md` for what you changed, and append a short "Done / Deviations / Reference numbers reproduced" note to `EVAL_MERGE_LOG.md` at the repo root.

| WP | Title | Depends on | Can run in parallel with |
|---|---|---|---|
| WP0 | Migrate `scripts/eval/` into `eval/` package | – | – (must finish first) |
| WP1 | ICD coders (`eval/coders/`, `eval/icd.py`) | WP0 | WP2, WP3, WP4, WP6 |
| WP2 | Merged ACI gold (`eval/gold/`) | WP0 | WP1, WP3, WP4, WP6 |
| WP3 | Attacks and robustness (`eval/bonus/attack.py`, `robustness.py`) | WP0 (A4 uses WP1's Ollama helper; stub until merged) | WP1, WP2, WP4, WP6 |
| WP4 | Corpus drift (`eval/bonus/drift.py`, `docs/CORPUS-DRIFT.md`) | WP0 | WP1, WP2, WP3, WP6 |
| WP6 | Explainability sidecar and no-codes sub-table | WP0 | WP1–WP4 |
| WP5 | Report, tradeoff, CSV exports | WP1, WP2, WP3, WP4, WP6 | – |
| WP7 | Full verification and docs | WP5 | – |
| WP8 | Animated pipeline visualization | WP7 | – |

Integration contract between parallel WPs (so they can merge cleanly): score JSON keys are `validity, privacy, keep, concepts, edits, terms, icd{<coder>}, explain`; attack output goes to `data/eval/attacks/<model>/<suite>.json`; drift to `data/eval/drift.json`; merged gold to `data/eval/gold/aci_eval/merged.jsonl`; the report reads only these files plus `runs/*/*/run.json`.

## Context

Four harnesses (plus a fifth inside the A+D model repo) evaluate the anonymizers for the Neural Wave
"Anonymization of patient data" challenge. Each measures a slice, with incompatible golds, ICD coders,
F1 definitions and failure semantics:

| Harness | Path (under `neuralwave-container/`) | Unique value to absorb |
|---|---|---|
| Unified pipeline | `neuralwave/scripts/eval/` (uncommitted in git) | Model registry, 10 suites, privacy/keep/edit/term metrics, cost measurement, report |
| ICD utility | `daniel-eval/` | Ollama coder with strict JSON schema, canonical code normalization, paired bootstrap CI, 22 pytest cases |
| Ketai baseline | `ketai-eval/` | Third ACI gold (484 PII, 4,334 concepts, 173 ambiguous terms), category-level (3-char) F1, F1 retention, `audit_gold.py`, age/date generalization logic |
| VM dev tools | `vm-backup/neuralwave/evaluation/` | VM gold (815 PHI with tiers, 5,211 concepts), `check_annotations.py` verbatim check, `annotate_headless.py` (resumable LLM annotator) |
| A+D eval package | `models/ad-night/repo/src/neural_wave/eval/` | `attack.py` (5 adversarial attacks A1–A5), `robustness.py` (3 unseen-format sets R1–R3), harness gold source |

The organizers score (task PDF, p. 3): (i) proportion of removed privacy-related information, (ii) ICD-10
F1 on anonymized vs original, (iii) medical concept recall; plus originality/scalability and bonus tasks
(adversarial attacks, explainability share, runtime/peak RAM/peak VRAM, no-codes mode, robustness with
"complexity of additional data"). Hidden gold, on-prem, 4 GB VRAM. The paper Cornelius & Rinaldi 2026
(`s44342-026-00072-9.pdf`) contributes: a harmonized 4-type PHI schema (PERSON, DATE, CONTACT, ADDRESS),
strict entity-level matching as the primary P/R/F1, and lexical/structural/semantic corpus-drift metrics
that predict out-of-distribution degradation (JSD unigram/bigram r=0.54, MTLD, perplexity, Flesch-Kincaid,
vocab Jaccard, MMD, Fréchet, cluster JSD).

Goal: one package `neuralwave/eval/` that is the only harness, scores every team model the same way,
reproduces the old harnesses' reference numbers, covers every organizer metric and every bonus metric,
has tests, and is documented. Execution is dispatched to Opus 5.5 subagents by work package (WP).

## User decisions (already made)

- New top-level package `neuralwave/eval/` (not `scripts/eval/`). Migrate the unified pipeline into it.
- `daniel-eval/`, `ketai-eval/`, `vm-backup/` stay untouched. Read-only sources.
- Drift metrics: lexical ones implemented; `CORPUS-DRIFT.md` explains each metric in a concise table and lists the embedding metrics (MMD, Fréchet, cluster JSD, perplexity) as TODO rows.
- Absorb A+D `attack.py` and `robustness.py` as generic bonus modules runnable on every model.
- Final step: an animated visualization of the eval pipeline.
- Repo rules (`neuralwave/AGENTS.md`): no AI attribution anywhere; never commit `data/eval/` or `data/kaggle/`; stdlib-only stays the norm for the harness (anonymizers bring their own venvs).

## Target layout

```
neuralwave/
  eval/                         # package, `python3 -I -m eval ...` from repo root (add neuralwave/eval/__main__.py)
    __init__.py  __main__.py    # CLI: suites | run | score | report | attack | robustness | drift | gold | all
    common.py                   # paths (REPO, CONTAINER, DATA, OUT=data/eval), ENCOUNTER_KEYS, CATEGORY_RISK, pattern/count/words, OLLAMA_URL
    suites.py  run.py  score.py  report.py  metrics.py   # migrated from scripts/eval, changed as below
    icd.py                      # orchestration only: cached_predict(), score(coder, orig, anon)
    coders/  base.py codes.py knn.py lexical.py command.py ollama.py bootstrap.py __init__.py
    coders.toml  models.toml
    gold/   taxonomy.py merge_aci_gold.py audit_gold.py
    bonus/  attack.py robustness.py explain.py drift.py
    adapters/ baselines.py anonymous_vm.py
    tests/  test_codes.py test_bootstrap.py test_coders.py test_metrics.py test_gold.py test_drift.py test_validity.py
    docs/   README.md CORPUS-DRIFT.md GOLD.md pipeline.html
  data/eval/                    # gitignored: suites/ runs/ scores/ cache/ gold/ attacks/ report.md report.json
```

`scripts/eval/` is removed in WP0 after migration (it is uncommitted, so no history to keep). `AGENTS.md`
and `DATA_PREP.md` rows that point at `scripts/eval/` are updated.

## Work packages

Dependencies: WP0 → {WP1, WP2, WP3, WP4, WP6} in parallel → WP5 (report) → WP7 (verification) → WP8 (viz).
Each WP ends with its tests passing under `python3 -I -m pytest neuralwave/eval/tests` (pytest is the one dev
dependency; the harness itself stays stdlib).

### WP0. Migrate the unified pipeline into the package (prerequisite)

1. `git mv`-equivalent of `scripts/eval/*` into `neuralwave/eval/`; absolute imports (`from eval import common`); `__main__.py` with subcommands mapping to the old scripts; `{here}` placeholder in `models.toml` now points to `neuralwave/eval/adapters`.
2. Fix defaults: `run` and `evaluate` both default to all suites (today `run.py:178` defaults to `aci_eval` only).
3. `run.py`: add `timeout` per model (`models.toml` key, default 3600 s), kill the process tree on timeout, record `exit_code=124`. Add model code fingerprint to the run cache key: sha1 of the resolved `cmd` plus `spec.fingerprint` (optional toml key: a path or glob whose contents are hashed, e.g. the model's `src/`), so a changed model with the same command reruns.
4. `suites.py`: `ensure()` rebuilds when `input_sha1` of the converted source or the gold file changed (today it never rebuilds, `suites.py:142-148`). Correct the README: 10 suites.
5. `metrics.validity`: unchanged strictness, but allow one optional extra key `explanations` (list) so explainability can ride on the output file (see WP6) and strip it before comparison.
6. Regression gate: re-score all existing runs under `data/eval/runs/` and assert equality with the current `data/eval/scores/*/*.json` (copy them to `data/eval/scores_baseline/` first). Headline numbers to reproduce: `ad_night` aci_eval privacy 0.839, `anonymous` 0.944 1.000/0.634 (VM gold), `identity` 0.
7. Tests: `test_validity.py`, `test_metrics.py` (privacy, leaked/distinctive tokens, survival, edits, terms on hand-made 3-record fixtures with known answers; include the `120/59` number-boundary case and the `[PATIENT] Li` short-token case as documented behaviour).

### WP1. ICD coders (`eval/coders/`, `eval/icd.py`)

Design: one `Coder` protocol, universal set-based scoring, ranked coders cut at k=|gold| first.

- `coders/base.py`: `Coder(name, config, ranked: bool, inputs: "note"|"both", predict(records) -> dict[id, Prediction])`; `Prediction(codes, error, attempts, seconds, raw_invalid)`.
- `coders/codes.py`: vendor daniel's `normalize_codes` (`daniel-eval/src/icd_evaluation/data.py:59-78`) but return `None` on invalid instead of raising; strip non-alphanumerics first (ketai leniency), validate `[A-Z][0-9][A-Z0-9](\.?[A-Z0-9]{1,4})?`, canonical dot after position 3; `normalize_set` keeps first-seen order and accepts `{code,...}` dicts; `category(code)=code[:3]`. Gold codes go through the same function (11 of 153 gold codes are undotted).
- `coders/ollama.py`: vendor daniel's `model.py` (strict JSON-schema `format`, temperature 0, seed, `think:false`, `done_reason=="stop"` check, context guard, digest via `/api/tags`). Changes: URL from `common.OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")` (loopback guard kept, non-loopback allowed but recorded as `endpoint`); `PROMPT_NOTE` (daniel's verbatim) and `PROMPT_BOTH` (ketai's `DIALOGUE:\n…\n\nPHYSICIAN NOTE:\n…` framing with daniel's instruction and schema); `device` default `auto`; `unload(model)` (`keep_alive: 0`). Drop ketai's loose-JSON salvage: the schema-constrained format makes it unnecessary.
- `coders/bootstrap.py`: vendor daniel's `metrics.py` `interval`, `rates`, paired bootstrap (1000 resamples, seed 0, encounter unit).
- `coders/knn.py`, `lexical.py`, `command.py`: move from `icd.py`; `ranked=True`; `command` gets `timeout`, `check=False`, per-record `missing_output` errors; input/output file contract unchanged (`{input}` challenge-schema JSONL with `icd10_codes: []`, `{output}` JSONL `{"encounter_id","codes":[…]}`).
- `coders.toml` registry: `llm_note` (qwen3.5:2b-q4_K_M, note only, PRIMARY), `llm_both`, `knn` (suites aci_eval, aci_dev), `lexical`, `paper` (cmd, disabled, slot for Daniel's full paper coder). `--icd` takes registry names; `cmd:<template>` ad-hoc form kept.
- `icd.py: score`: per record normalize gold; ranked coders → `pred[:len(gold)]`; rows (tp,fp,fn) at code and category level for original, anonymized, agreement (anonymized vs original as pseudo-gold); micro P/R/F1, mean-per-note F1, CI95 per arm, paired `f1_delta` with CI, `f1_retention`, `errors`, `invalid_codes`, `valid_for_comparison`. Failed predictions count as empty. Cache `data/eval/cache/icd/<coder>/<sha1(config,id,text-fields-read)>.json` storing `{codes, attempts, seconds, usage}`; errors never cached; old plain-list cache files read as `{"codes": list}`. Coder-level failure (Ollama unreachable, 3 consecutive failures) → `icd[<name>] = {"error": …}` and scoring continues.
- `run.main` calls `coders.ollama.unload()` first when the server is reachable, so a `score → run` sequence does not leave ~2.5 GB resident while an anonymizer's VRAM is measured.
- A+D bge-m3 retriever: not ported (same proxy class as `lexical`, needs numpy + a second Ollama model); reachable later as a `cmd` coder if the A+D repo exposes a CLI.
- Tests: port daniel's normalization, hallucination micro-vs-mean, paired-bootstrap-identical-arms, malformed-response and context-guard cases against a mocked HTTP layer. Regression: new `knn`/`lexical` `anonymized.code.f1` equals old `icd.knn.f1_anonymized` on existing scores. Cross-check: run `daniel-eval/scripts/evaluate_icd.py` once on `identity` and `ad_night` aci_eval outputs and assert our `llm_note` micro F1 per arm matches to 1e-4.

### WP2. One ACI-eval gold (`eval/gold/`)

Inputs: harness gold `data/converted/aci_bench/labels.jsonl` (532 items + 194 keep), VM gold `vm-backup/neuralwave/annotations/eval/*.json` (815 PHI, 5,211 concepts), ketai gold `ketai-eval/gold/eval_gold.jsonl` (484 PII, 4,334 concepts, 173 ambiguous). Measured overlap: all three 289, ≥2 exact votes 490, union 1,052; nesting is the dominant disagreement (173 harness-only items are whole-word substrings of a ketai item, e.g. `martha`/`collins` vs `Martha Collins`).

- `taxonomy.py`: keep the harness's 11 categories, add `CONTACT` (high). Maps: VM NAME_PATIENT→PATIENT_NAME; other NAME_*/ketai CLINICIAN_NAME/OTHER_PERSON_NAME→PERSON_NAME (source type kept in `label`); VM/ketai DATE→DOB if another gold says DOB or preceded within 25 chars by `date of birth|dob|born`, YEAR if `^\d{4}$`, else DATE; ACTIVITY/LIFE_EVENT/TEMPORAL/PERSONAL_DETAIL/OTHER_QUASI→OTHER_QI. Category conflicts: majority, tie priority harness > ketai > VM; all labels kept in `source_labels`. `direct = category in {PATIENT_NAME, PERSON_NAME, DOB, DATE, ID, CONTACT}` (paper schema + task example "names, date of birth"); `risk` from `CATEGORY_RISK`.
- `merge_aci_gold.py` → `data/eval/gold/aci_eval/merged.jsonl` + `merged_stats.json` + `merged_review.md`. Item schema: `{field, text, category, risk, direct, sources, votes, nested_votes, source_labels, label, contains|within, standalone_occurrences}`. Nesting rule per (encounter, field): longest span is canonical; a shorter item from another gold that is a whole-word substring adds a nested vote and is listed in `contains`; it survives as its own item only when it has standalone occurrences (`count(s) > Σcount(containers)`), inheriting the container's votes. Keep = harness keep ∪ ketai non-PII ambiguous terms (EPONYM when also a concept, else NOT_A_NAME); `pii ∩ keep` → pii wins if votes ≥ 2 else keep wins (logged). Concepts = VM ∪ ketai, types canonicalized (`lab_or_test→TEST`, add ANATOMY), `icd_relevant = any`, `icd_relevant_votes`; a concept equal to a direct PII item is dropped and logged. Deterministic ordering.
- Levels: `union` (all), `consensus` (`direct or votes ≥ 2`), `direct`. **Headline privacy = consensus**; `direct` shown beside it as a gate (saturates ≈ 1.0 for all real models), `union` as pessimistic bound, `src:harness|vm|ketai` breakdowns so the A+D cross-check (0.839) survives. Entity-level removal (paper's strict matching: item fully removed) reported per level.
- `audit_gold.py` (merge of ketai `audit_gold.py` ideas and VM `check_annotations.py`): verbatim whole-word check per source; contradictions (`pii∩keep`, `pii∩concepts`, PII made only of ICD-description words); regex candidates not covered by any item (ages, dates, titles+name, emails, phones, digit runs ≥ 6); single-vote table; vote histogram per category; diff vs previous merged file.
- `metrics.privacy`: level keys `level:union|consensus|direct`, `votes:1|2|3`, `src:*`, entity-level per level; `removed` = consensus. `score.py` drops `privacy_alt`. `suites.py`: aci_eval labels from `merged.jsonl` when present (`--no-merged` fallback), `suite.json` gets `gold_sources`, `pii_levels`, `merged_sha1`. `aci_dev` stays harness-only (A+D tuned on it; no second gold).
- `docs/GOLD.md`: provenance, three levels, caveat that all three golds are LLM-assisted so votes are less independent than they look.
- Tests: taxonomy rules, nesting rule on a synthetic encounter, level filters, `src:harness` for `ad_night` ≈ 0.839 after re-score.

### WP3. Bonus: adversarial attacks and robustness (`eval/bonus/attack.py`, `robustness.py`)

Port from `models/ad-night/repo/src/neural_wave/eval/attack.py` (287 lines) and `robustness.py` (237 lines), replacing imports of the A+D package (`lexicons`, `io`, `text.Doc`, `config`, `models`, `rules`) with harness equivalents or vendored resources.

- `attack.py`: generic over any `runs/<model>/<suite>/output.jsonl`; always runs on the original text as positive control. A1 independent NER residual scan (deid_roberta + GLiNER via the anonymous venv `models/anonymous-deps/.venv`, reuse `adapters/anonymous_vm.py` loaders); A2 roster dictionary (corpus patient names + the Kaggle names list `krist0phersmith__pii-real-names-dataset` instead of US Census download); A3 rare-word re-identification; A4 LLM inference attack via `coders/ollama.py` request helper (guess name/age/place/employer, score vs merged gold, age ±1); A5 TF-IDF record linkage anonymized→original. Output `data/eval/attacks/<model>/<suite>.json` with `recovered` proportion per attack and per category, plus the positive control. Report section "Adversarial attacks" with recovered-share columns per model.
- `robustness.py`: R1 discharge (devp OCR ground truth, three layouts), R2 medconv with injected PII, R3 the 24 synthetic hospital documents (`outputs/robustness/synthetic.jsonl` + `outputs/gold/robust_synthetic/` copied into `data/converted/robust_synthetic/` via a converter in `scripts/convert/`). Register as suites `robust_discharge`, `robust_medconv`, `robust_synthetic` with role `robust`; dedupe against existing `devp_discharge` (keep devp_discharge as is, R1 adds the two other layouts only). "Complexity of additional data" column comes from WP4 drift metrics.
- Tests: each attack on a 3-record fixture where the answer is known (identity output → recovered = control; a fully cleaned output → 0 for A2).

### WP4. Corpus drift metrics (`eval/bonus/drift.py`, `docs/CORPUS-DRIFT.md`)

Per suite vs the ACI train corpus (140 encounters, dialogue+note), stdlib only: JSD unigram, JSD bigram (ε-smoothed over the vocabulary union), vocab Jaccard, vocab entropy diff, MTLD, Yule's K, Distinct-1/2, 4-gram self-repetition rate, sentence-length mean/std diff, punctuation/digit/capitalized token ratios, Flesch-Kincaid grade diff (syllable heuristic), line-break density, key-value sentence ratio, bullet ratio, all-caps ratio, special-char ratio. Output `data/eval/drift.json` and a report section "Suite complexity (drift from ACI)" ordered by JSD unigram; the robust suites and Kaggle suites get a `complexity` rank used in the robustness table. `CORPUS-DRIFT.md`: one table row per metric (name, family, formula in words, what a high value means, paper correlation with ΔF1 where given), plus TODO rows for perplexity (GPT-2), MMD, Fréchet distance, embedding-mean distance and cluster-coverage JSD, each with the sentence encoder it would need. Tests: identical corpora → all divergences 0; a shuffled-vocabulary corpus → Jaccard 0.

### WP5. Report (`eval/report.py`) and headline

- Headline per model: `privacy (consensus)`, `direct`, `entity-level`, `concepts kept`, `ICD concepts kept`, `ICD F1 anon (llm_note)`, `ICD F1 retained`, `ICD agree`, `ms/enc`, `peak RAM`, `peak VRAM ≤4096`, `explained %`, `attack recovered (max)`. The identity row stays as the reference.
- New sections: ICD coders table (model × coder: f1_orig, f1_anon, Δ[CI95], agree, cat-F1, errors, `!` when `valid_for_comparison` is false); privacy levels table; adversarial attacks; robustness (with complexity rank); suite drift table; efficiency (existing).
- Tradeoff view: a privacy-vs-utility table sorted by privacy with a Pareto flag column (model is Pareto-optimal if no other model is ≥ on both consensus privacy and ICD F1 retained). Also emit `data/eval/tradeoff.svg` (stdlib SVG scatter, identity marked) so the README and the viz can embed it. No combined score (organizers weigh criteria equally and separately).
- `report_discord.md` generator: a compact headline-only variant (today it is hand-made).
- Dynamic coder columns; no hard-coded `knn`/`lex`.
- **CSV deliverable** (written by `report` at the end of every `eval all` run, stdlib `csv`, UTF-8 with BOM so Excel opens it directly, `data/eval/results.csv`): one long-format table with columns `model, suite, category, metric, value, n, ci_low, ci_high, coder, level, notes`. `category` is the result family (`privacy`, `privacy_by_category`, `privacy_by_field`, `privacy_by_language`, `keep`, `concepts`, `edits`, `terms`, `icd`, `attack`, `robustness`, `drift`, `efficiency`, `explain`, `validity`); every number that appears in `report.md` is also a row here, plus the per-category/per-field breakdowns the markdown omits. A second wide file `data/eval/results_headline.csv` has one row per model and one column per headline metric, for quick pivoting. Both are listed in `report.json` under `exports`. Test: every headline cell in `report.md` is found as a row in `results.csv` with the same value.

### WP6. Explainability and no-codes coverage

- Explainability bonus = "percentage of anonymization decisions having an explanation". Define an optional sidecar `runs/<model>/<suite>/output.decisions.jsonl` (A+D already writes one) with rows `{encounter_id, field, start, end, text, action: remove|generalize|perturb|keep, category, reason, source}`. `bonus/explain.py` scores: share of changed tokens (from `metrics.edits` alignment) covered by a decision with a non-empty `reason`; report column `explained %`. The anonymous adapter emits decisions from `Span.source/confidence/reason`; ketai's `generalize.py` explanations map directly when his model lands.
- No-codes bonus: already covered by `blind_codes` (`ad_night_nocodes`); add a report sub-table "without ICD-10 codes at inference" that pairs each `*_nocodes` model with its base model.

### WP7. Verification and docs

1. Full run: `python3 -I -m eval all` from `neuralwave/` with Ollama serving `qwen3.5:2b-q4_K_M`. Expect: every existing model row present; `identity` privacy 0 and utility 1 at every level; `ad_night` `src:harness` 0.839; `knn`/`lexical` numbers unchanged from `scores_baseline/`; `llm_note` arms match `daniel-eval` to 1e-4; no coder error cells; attacks' positive control > anonymized recovered for every model.
2. `audit_gold.py` report has zero verbatim failures and an empty regex-gap list at the union level.
3. `pytest` green; `python3 -I -m eval --help` lists every subcommand.
3b. `data/eval/results.csv` and `results_headline.csv` exist after `eval all`, open in LibreOffice/Excel without encoding or delimiter prompts, and contain a row for every headline cell.
4. `docs/README.md` rewritten: prerequisites (`ollama serve`, `ollama pull qwen3.5:2b-q4_K_M`, model venvs), one-command run, suites table (10 + 3 robust), metrics table with definitions, coders table, gold levels, bonus modules, how to add a model (`models.toml`) and a coder (`coders.toml`), what is cached where, Linux-only cost measurement caveats. Update `AGENTS.md` layout table and `DATA_PREP.md` pointers. Mention that `daniel-eval/`, `ketai-eval/` and `vm-backup/` are retained as read-only sources.
5. Commit on a branch in `neuralwave/` with the attribution rule respected; `data/eval/` and `data/kaggle/` stay ignored.

### WP8. Animated visualization of the eval pipeline (final step)

- `neuralwave/eval/docs/pipeline.html`: a single self-contained page (inline SVG + CSS/JS animation, no CDN needed for offline on-prem viewing) showing the data flow left to right: converted datasets and the three ACI golds → `gold merge` → suites (10 + 3 robust) → model runs (one lane per `models.toml` entry, with the cost probe) → scorers (privacy levels, keep/edits/terms, concepts, ICD coders with Ollama, attacks, drift) → report/tradeoff. Animated tokens travel along the edges; clicking a stage shows its inputs, outputs and the file paths under `data/eval/`; a play/pause control and a "step" mode. Optionally reads `data/eval/report.json` if served next to it to populate live numbers in the stage tooltips, otherwise shows static placeholders.
- Load `artifact-design` (and `artifact-diagramming`) before writing it; also publish it as a private Artifact so the team can open it without cloning. Link it from `docs/README.md`.

## Critical files

- Base to migrate: `neuralwave/scripts/eval/{common,suites,run,score,metrics,icd,report,evaluate}.py`, `models.toml`, `adapters/`.
- To vendor: `daniel-eval/src/icd_evaluation/{model,metrics,data}.py`; `ketai-eval/gold/{audit_gold,build_gold}.py`; `vm-backup/neuralwave/evaluation/check_annotations.py`; `models/ad-night/repo/src/neural_wave/eval/{attack,robustness}.py` and `outputs/robustness/`, `outputs/gold/robust_synthetic/`.
- Golds: `neuralwave/data/converted/aci_bench/labels.jsonl`, `vm-backup/neuralwave/annotations/eval/`, `ketai-eval/gold/eval_gold.jsonl`.
- Reference docs: task PDF `neuralwave/task/…public.pdf`; paper `s44342-026-00072-9.pdf` (drift metric definitions pp. 9–10, Table 3 label harmonization p. 8).

## Risks to state in the report

- The three ACI golds are all LLM-assisted; "votes" overstate independence.
- Rebuilding `aci_eval` labels changes every published ACI privacy number; `src:harness` keeps the A+D cross-check reproducible.
- `llm_note` first pass costs 67+67 Ollama calls per model on aci_eval (plus devp if enabled); roughly 20–40 min for 8 models on the RTX 4050, then cached by text, so unchanged notes share predictions.
- Cost measurement is Linux-only and sums RSS over the process tree (overcounts shared pages); documented, not fixed.
- The organizers' hidden gold is unknown; the three privacy levels bracket it.
