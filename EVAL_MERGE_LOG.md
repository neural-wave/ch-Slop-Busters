# Eval merge log

## WP0

### Done

- Moved `scripts/eval/*` into the package `eval/`, with absolute imports (`from eval import ...`, `from eval.common import ...`):
  `__init__.py`, `__main__.py`, `common.py`, `suites.py`, `run.py`, `score.py`, `report.py`, `metrics.py`, `icd.py`,
  `models.toml`, `adapters/{__init__,baselines,anonymous_vm}.py`, `docs/README.md`, `tests/`, and empty packages `bonus/`,
  `gold/`, `coders/`. `evaluate.py` became the `all` command. `scripts/eval/` is deleted. The only `sys.path`
  changes left are the bootstrap in `__main__.py` (repo root replaces `eval/`), the one in `adapters/baselines.py`
  (it runs as a standalone script), and `anonymous_vm.py`'s VM import.
- `common.py`: `HERE` = `eval/`, `REPO` = `neuralwave/`, `CONTAINER` = its parent, `ADAPTERS`, and
  `OLLAMA_URL` from `OLLAMA_HOST`, parsed like ollama's `envconfig.Host`: unset → `http://127.0.0.1:11434`;
  `host`, `host:port`, `:port` (empty host → `127.0.0.1`), a bare or bracketed IPv6 address (→ `[addr]`), or a URL.
  A missing or invalid port becomes 11434, or 80/443 in an http/https URL.
- CLI `python3 -I eval <command>`: `suites | run | score | report | attack | robustness | drift | gold | all`. The
  dispatch table (`COMMANDS`) imports modules lazily. A command whose module does not exist yet prints
  `<cmd>: not implemented yet` and exits 2. `all` runs `STEPS` in order: `[gold] suites run score [attack] [drift] report`.
  Optional steps are skipped while their module is missing. `all` forwards `--models`, `--suites`, `--icd` and
  `--force` to the steps that take them, and `--if-stale` to `suites`. `attack` gets `--models --suites`, `drift`
  gets `--suites`, and `gold` gets nothing. It keeps going after a failed step, or one that raises (the traceback
  is printed), and exits 1. Ctrl-C and SIGTERM/SIGHUP still stop it.
- The stage modules `run.py`, `score.py`, `suites.py` and `report.py` also run as files (`python3 -I eval/run.py`):
  like `adapters/baselines.py`, they put the repo root on `sys.path` when they have no package.
- `run` and `all` default to every enabled model and every suite.
- `run.py`: each model has a `timeout` (default 3600 s). The child starts in a new session with stdin `/dev/null`.
  On timeout, `killpg` kills its process group and SIGKILL goes to every pid of the `/proc` tree walk, including
  children that left the group with `setsid`. `run.json` then gets `exit_code: 124` and `timed_out: true`.
  - The harness waits with `waitid(WNOWAIT)`, marks the exit under the timer's lock and only then reaps, so the timer
    never signals a reaped (reusable) pid. A run counts as timed out only when the timer fired and the child died of
    SIGKILL, so a run that exits on its own right at the deadline keeps its exit code.
  - Ctrl-C, SIGTERM and SIGHUP to the harness kill the model's tree too: while a model runs, `measured()` turns
    SIGTERM and SIGHUP into `SystemExit(128 + signum)` (main thread only; an ignored signal stays ignored, so `nohup`
    keeps working), and the handlers are restored afterwards.
  - A command that cannot start (missing executable or `cwd`) is recorded with exit code 127 (126 for a permission
    error) and the error in `run.log`, and the other models still run.
  - Wall time ends when the model exits, not after the RAM/VRAM sampler stopped. Before, it included an in-flight
    `nvidia-smi` call (about 2 s on an idle GPU), e.g. `identity` on `aci_eval` was 2.18 s, now 0.25 s.
  - The run cache key now includes `code_sha1`: the sha1 of the resolved command (JSON) plus the files of the
    optional `fingerprint` key. Each file is hashed with its path relative to the pattern's fixed prefix, in sorted
    order. A run is reused only when `input_sha1`, `cmd`, `code_sha1` and `exit_code == 0` all match and
    `output.jsonl` exists. While a model runs, `run.json` holds `exit_code: null` (the old output is deleted first),
    so an interrupted run is scored as failed and rerun next time.
- `models.toml`: `{here}` is now `eval/adapters`. The header documents `timeout` and `fingerprint`. The
  fingerprints are:
  - `identity`: `baselines.py` and `common.py`
  - `anonymous`, `anonymous_rules`: `anonymous_vm.py`, the VM's `detect/**/*.py` and `common.py` (`Span`, `TIER`, `NEVER_PHI`),
    and the `--words` list `models/anonymous/american-english-large`
  - `ad_*`: the A+D repo's `src/**/*.py` and `src/**/*.gz`

  `anonymous_vm.py`'s `DEFAULT_ROOT` now uses `parents[3]`.
- `suites.py`: `suite.json` records `source_sha1`. This is the sha1 over the converted source's `encounters.jsonl`,
  `labels.jsonl` and `manifest.json`, plus, for `aci_bench` suites, the VM annotation files of the suite's records
  in sorted order. It also records `labels_sha1` (beside `input_sha1`) and `code_sha1`, the sha1 of the build code
  (`suites.py` and `common.py`). `ensure()` rebuilds a suite in these cases, and prints why:
  - `suite.json` is missing.
  - `suite.json` has no `source_sha1`.
  - `source_sha1` changed.
  - The definition changed.
  - The build code changed.
  - `encounters.jsonl` or `labels.jsonl` no longer matches its sha1 (an interrupted build, an edit).

  `eval suites` rebuilds every selected suite; `eval suites --if-stale` only the stale ones, and that is the `suites`
  step of `all`. A suite built with `--vm-annotations DIR` records DIR as `vm_annotations` in `suite.json` (only when
  it is not the default), and `ensure()` and every rebuild keep it until another `--vm-annotations` is given. A
  missing VM dir prints a warning.

  When the converted source is missing, `ensure()` keeps the existing suite and warns, or leaves the suite out if it
  was never built (`run` skips it). The README now states 10 suites.
- `metrics.validity`: the strictness is unchanged, but one optional `explanations` key is allowed. It must be a list
  (anything else gives `explanations not a list`), and it is stripped before the key comparison.
- Tests (`eval/tests/`, 83 passing with `uv run --no-project --with pytest python3 -I -m pytest eval/tests -q`):
  - `test_validity.py` (12)
  - `test_metrics.py` (24), covering:
    - privacy: full removal, partial leak, the `[PATIENT] Campbell` distinctive-token leak, the `120/59` number
      boundary, and `[PATIENT] Li` documented as not a leak
    - survival, edits and terms. `common_words` and `clinical_vocabulary` are monkeypatched.
  - `test_run.py` (15): `code_sha1`, the timeout killing a grouped child and a `setsid` child, a normal exit,
    SIGTERM and SIGHUP to the harness killing the model (and an ignored SIGHUP staying ignored), a command that
    cannot start, an exit at the deadline that is not a timeout, wall time without a slow `nvidia-smi`, and an
    interrupted run that is not cached
  - `test_suites.py` (9): rebuild rules, byte-identical rebuild, a missing source, damaged suite files, a build code
    change, `--if-stale`, and the recorded VM annotation dir
  - `test_main.py` (8): help, a missing module, a crashing step that does not stop `all`, the stage modules as files
  - `test_common.py` (15): paths, and `OLLAMA_HOST` forms including `:11434`, `::1` and an invalid port
- Docs: `eval/docs/README.md` covers the commands table, caching, the 10 suites, the `models.toml` keys, the layout
  and tests. `AGENTS.md` layout row, `DATA_PREP.md` pointer and the `.gitignore` comment now point to `eval/`.
  `.gitignore` also ignores `.pytest_cache/`. Outside `data/` and `EVAL_MERGE_PLAN.md`, no `scripts/eval` reference
  is left.

### Deviations

- The CLI is `python3 -I eval ...`, not `python3 -I -m eval ...`. Isolated mode implies `-P`, which drops the cwd
  from `sys.path`, so `-I -m eval` cannot find the package. Running the directory executes `eval/__main__.py` with
  `eval/` as `sys.path[0]`, and the bootstrap replaces that entry with the repo root. `python3 -m eval` (without
  `-I`) also works.
- pytest is not installed for the system Python. Tests run through `uv run --no-project --with pytest ...` and the
  harness stays stdlib-only.
- `all` lives in `__main__.py` (`STEPS`), not in a separate module. The optional steps are registered already, so
  later WPs only add their modules. They must accept the options listed above, or edit their `STEPS` entry.
- Fingerprints go slightly beyond the brief:
  - `identity` also hash `eval/common.py`, because `load_jsonl`/`write_jsonl`/`pattern` shape their
    output.
  - `ad_*` also hash `src/**/*.gz`, because the A+D lexicons live in `src/neural_wave/data/`.
  - Relative fingerprint paths resolve from the model's `cwd`.
- `ensure()` also rebuilds when the suite definition (`source`, `where`, `where_not`, `limit`, `role`, `about`,
  seed) differs from `suite.json`, when the build code changed, and when a suite file no longer matches its sha1.
  `pii_gold` is not compared, because `suite.json` stores the levels found. `suites` accepts `--suites` as an alias of
  `--only`, and `--if-stale`. `stale()` returns the reason (empty when up to date) instead of a bool.
- `suite.json` gains `labels_sha1`, `code_sha1`, and `vm_annotations` for a non-default VM dir. The `nbme_notes`
  description now says its graded findings are must-keep gold (they are `keep` items, not concepts).
- `run.json` gains `code_sha1` and `timed_out`, and holds `exit_code: null` while the model runs. `wall_s` now ends
  at the model's exit, so short runs are no longer inflated by an `nvidia-smi` sample.
- The run timeout does not reach a descendant that daemonized itself (double fork, `setsid -f`): it has left the
  `/proc` tree. Making the harness a child subreaper would find it but could not tell which run it came from, so
  this is documented instead. A SIGKILLed harness cannot kill the model either.
- `report.main` now takes `argv` (with `--help`). Every command module has `main(argv=None) -> int`.

### Reference numbers reproduced

- Regression gate, run before any model rerun: all runs were re-scored with `python3 -I eval score` after deleting
  `data/eval/scores/`. **71 of 71** score files are equal (parsed JSON) to `data/eval/scores_baseline/`. The
  regenerated `report.md` is identical to the old one apart from the timestamp, and `report.json` is equal apart
  from `generated`.
- `aci_eval` privacy / VM-gold privacy:

  | Model | privacy | VM-gold privacy |
  |---|---:|---:|
  | `ad_night` | 0.8388 | 0.5398 |
  | `ad_night_submitted` | 0.8388 | 0.5398 |
  | `ad_night_nocodes` | 0.8428 | 0.5452 |
  | `ad_rules` | 0.8279 | 0.5316 |
  | `anonymous` | 0.9444 | 0.8391 |
  | `anonymous_rules` | 0.6355 | 0.4458 |
  | `identity` | 0.0 | 0.0 |
- Suite rebuild: in all 10 suites, `encounters.jsonl` and `labels.jsonl` are byte-identical to the pre-change
  copies. `suite.json` differs only by the new `source_sha1`, `labels_sha1` and `code_sha1` (and the `nbme_notes`
  description). A second `eval suites --if-stale` rebuilds nothing.
- Reruns through `python3 -I eval run`: the `aci_eval` `output.jsonl` of `identity`, `ad_rules` and
  `anonymous_rules` is byte-identical to `data/eval/runs_baseline/`. `identity` on `sadiqguru` through `eval all` is too.
  A second `run` reuses the cache. After re-scoring, all 71 files still match the baseline when the `run` block is
  ignored. The report differs only in the timings and RAM of the rerun models.
- Timeout: a throwaway spec with a 2 s timeout spawned `sleep` and a `setsid` `sleep`. The run ended after 2.09 s
  with exit 124, `timed_out: true` and no surviving process. The throwaway run dir was removed.
- After the review fixes, verified again: all suites rebuilt once (build code changed) with byte-identical files;
  `identity`, `ad_rules` and `anonymous_rules` rerun with `--force` on `aci_eval`, outputs byte-identical to
  `runs_baseline/`; `data/eval/scores/` deleted and every run re-scored: **71 of 71** files equal to
  `scores_baseline/`, 66 including the `run` block and 5 (the re-executed runs) apart from its timings, the new
  `code_sha1`/`timed_out` keys and the `scripts/eval/` → `eval/` adapter path. The headline numbers above are
  unchanged. `python3 -I eval all --models identity --suites sadiqguru` exits 0 and its output is byte-identical too.
  SIGINT, SIGHUP and SIGTERM sent to a harness running a model through `timeout -s <sig>` leave no surviving model.

### Left for later

- Every other cached run will rerun once on the next `eval run`/`eval all`, because its command path changed and
  `code_sha1` is new: `ad_night*` and `anonymous` on all suites, and the four models above on the suites other than
  `aci_eval` (`identity` also because `common.py` changed, `anonymous_rules` and `anonymous` because their
  fingerprint grew). `ad_night`, `ad_night_nocodes` and `anonymous` need the GPU.
- WP2 must add the merged gold file to `source_sha1` (or as `merged_sha1`) so that `aci_eval` rebuilds when it changes.

## WP6: Explainability and no-codes coverage

### Done

- `eval/bonus/explain.py`: `score(run_dir, original, anonymized)` gives the score file's `explain` block:
  `{explained, changed_tokens, covered_tokens, decisions, with_reason, unaligned, source}`. `explained` is the share of
  original word tokens that `metrics.align` marks as changed and that a decision with a non-empty reason and an action
  other than `keep` covers. It is `null` when nothing changed and 0.0 when text changed without any log.
  - Decisions come from `runs/<model>/<suite>/output.decisions.jsonl`, else from the inline `explanations` lists of
    the output records (`source`: `sidecar`, `inline`, `none`).
  - `normalize()` maps every producer to `{encounter_id, field, start, end, text, action, category, reason, source}`:
    - A+D's REPLACE/GENERALIZE/KEEP become remove/generalize/keep. A REPLACE by a non-placeholder becomes perturb.
    - The reason is `explanation`, else `detector_reason` + `policy_reason`.
    - Ketai's `type` and `removed`/`generalized`/`retained` are mapped too.
  - `place()` checks each row against the original:
    - offsets whose slice equals `text` (or rows without `text`) are used as they are;
    - a mismatched row moves to the nearest whole-word occurrence of its text;
    - a row without offsets covers every occurrence of its text.
    Malformed rows and rows that cannot be placed are counted in `unaligned`. The sidecar is decoded line by line
    (`utf-8-sig`), so a line that is not valid UTF-8 is one unaligned row and does not stop `eval score`.
  - Also runs as a file: `python3 -I eval/bonus/explain.py [--models] [--suites]`.
- `eval/metrics.py`: `align(orig, anon) -> [(token, start, end, kept)]` extracted from `edits()`. `edits()` now uses it
  and its results are unchanged.
- `eval/score.py`: `out["explain"] = explain.score(run_dir, original, anonymized)` on every suite.
- `eval/run.py`:
  - A precomputed model also copies `<precomputed stem>.decisions.jsonl` to `output.decisions.jsonl` when it exists.
    Its path and sha1 join the cached `cmd`, so a new sidecar reruns the copy.
  - Every (re)run first deletes the old `output.decisions.jsonl`, so a model that stops writing one leaves none.
- `eval/adapters/anonymous_vm.py`:
  - `merge()` also returns the members of each merged span.
  - `decision()` writes one row per merged span to `<output stem>.decisions.jsonl`: offsets and text in the original
    field, `remove` (or `keep` for a tier left out by `--tiers`), the span type, a reason joining each member's detector
    reason, layer and confidence, and `source` = the layers (`names+ner`).
- `eval/bonus/nocodes.py`:
  - `pairs(registry)`: each `blind_codes` entry with its `base` key or its name without `_nocodes`. An entry whose
    base is missing is left out.
  - `rows(scores, registry, suite="aci_eval")`: one row per pair, `{"base", "nocodes", "suite", "metrics": {metric:
    {"base", "nocodes", "delta"}}}` with delta = no-codes minus base. The metrics are `privacy` (`privacy.removed`,
    the consensus level after WP2), `privacy_high`, `keep`, `concepts`, `icd_concepts` and `icd_f1:<coder>`
    (`anonymized.code.f1`, or the old `f1_anonymized`; coders with an `error` are skipped).
  - `python3 -I eval/bonus/nocodes.py [--suite]` prints them.
- Tests: `eval/tests/test_explain.py` (18) and `test_nocodes.py` (3), all passing.
  - `test_explain.py` covers: `align`, full and partial coverage, no sidecar, identity, an empty reason, a keep
    decision, inline explanations, sidecar over inline, A+D normalization and scoring, Ketai rows without offsets,
    placement checks, unaligned rows, a sidecar with a BOM and non-UTF-8 lines, the anonymous `merge`/`decision` rows, and the precomputed sidecar copy (cache
    invalidation, no stale sidecar).
- `eval/docs/README.md`:
  - an `explained` row in the metrics table;
  - a section "Explainability and no-codes (bonus)" with the sidecar schema, the producers and both tools;
  - the sidecar notes in the `precomputed` row and the anonymous row.

### Deviations

- `explain` has one key beyond the contract: `unaligned`, the decision rows that could not be placed in the original.
- The unit is changed tokens, not logged decisions. Counting logged decisions would reward logging little.
  `decisions` and `with_reason` still give the literal ratio.
- `explain.py` and `nocodes.py` are not `eval` commands. Both run as files: `__main__.py` is not WP6's to edit.
- `nocodes.rows()` reads both ICD shapes, the WP1 arms and the old `f1_anonymized`, so it works before and after WP1.
- `run.py` change beyond the brief: it deletes a stale sidecar before each run.

### Reference numbers reproduced

- `anonymous_rules` and `anonymous` reran on `aci_eval` with `python3 -I eval run --models anonymous_rules anonymous --suites aci_eval`.
  `anonymous` used the GPU, 2716 MB VRAM. Both `output.jsonl` files are byte-identical (`cmp`) to
  `data/eval/runs_baseline/<model>/aci_eval/output.jsonl`.
  - The new sidecars have 485 rows (`anonymous_rules`) and 1836 rows (`anonymous`).
  - `text == original[field][start:end]` on 485/485 and 1836/1836 rows.
- A+D sidecars:
  - Both `data/eval/runs/ad_night/aci_eval/output.decisions.jsonl` and the delivered
    `models/ad-night/repo/outputs/anonymized_eval_patient_doctor_encounters.decisions.jsonl` have 775 rows. They differ
    from each other only in a few rows (from line 268), while the two `output.jsonl` are identical.
  - Neither has a `text` key (A+D's `explain_text` is off).
  - Indirect check on both: 775/775 rows in bounds and on word boundaries. 646/646 REPLACE/GENERALIZE slices contain a
    token that the alignment marks as changed. 122/129 KEEP slices are fully kept; the other 7 are partial shields
    whose rest is anonymized separately.
  - Direct check: A+D rules-only with `--explain-text`, run into the scratchpad with `-B` from outside their repo. Its
    output is byte-identical to our `ad_rules` run, and its sidecar equals ours apart from `text`. **635/635 rows that
    carry `text` match the original slice (100%).** Their KEEP rows (128) carry no text.
  - Replaying the decisions does not rebuild the output, but the offsets are not the cause: A+D post-processes the
    text, merging adjacent placeholders and replacing only the number of `50-year-old`.
- `ad_night_submitted` reran (precomputed copy). `output.jsonl` is byte-identical to the baseline and the sidecar was
  copied. A second `run` reuses the cache.
- `aci_eval` explain after `python3 -I eval score --suites aci_eval --icd knn lexical`:

  | Model | explained | changed tokens | decisions (with reason) | source |
  |---|---:|---:|---:|---|
  | `ad_night` | 1.0 | 810 | 775 (775) | sidecar |
  | `ad_night_submitted` | 1.0 | 810 | 775 (775) | sidecar |
  | `ad_night_nocodes` | 1.0 | 817 | 773 (773) | sidecar |
  | `ad_rules` | 1.0 | 788 | 763 (763) | sidecar |
  | `anonymous` | 1.0 | 2910 | 1836 (1836) | sidecar |
  | `anonymous_rules` | 1.0 | 843 | 485 (485) | sidecar |
  | `identity` | null | 0 | 0 | none |


  On the other suites, every `ad_*` run is explained 1.0. `anonymous`/`anonymous_rules` show 0.0 there until their next run:
  their fingerprint changed, so the next `eval run` redoes them and writes the sidecars.
- `edits()` regression:
  - Before WP2 landed, re-scoring all 8 models on `aci_eval` gave `edits` blocks equal to `data/eval/scores_baseline/`,
    and `privacy`, `keep`, `concepts`, `terms` and `validity` equal too. With the merged gold the labels differ, so
    only `validity` (and `identity`'s `edits`) still equal the baseline; `privacy.by_source.harness.removed` equals
    the baseline `privacy.removed` for all 8 models.
  - A copy of the pre-refactor `edits()` agrees with the new one on **83/83** runs (every model and suite with
    current labels; 71/71 at first, before the robust suites).
- No-codes pair on `aci_eval` (`ad_night_nocodes` vs `ad_night`), from `nocodes.rows()` on the current scores
  (privacy is now the consensus level): privacy 0.7738 -> 0.7793 (+0.0055), high-risk 0.9404 -> 0.9404, keep
  0.9933 -> 0.9888 (-0.0045), concepts 0.9989 -> 0.9987 (-0.0002), ICD concepts 0.9995 -> 0.9993 (-0.0002), ICD F1
  kNN 0.268 -> 0.268, lexical 0.1569 -> 0.1634 (+0.0065). On the harness gold alone (`by_source.harness`), privacy
  is 0.8388 -> 0.8428 as before.

## WP4: Corpus drift metrics

### Done

- `eval/bonus/drift.py` (stdlib only), CLI `python3 -I eval drift [--suites ...]`, also `python3 -I eval/bonus/drift.py`.
  - Reference: the 140 ACI train encounters (`data/converted/aci_bench`, labels `meta.split == "train"`), dialogue and
    note, 229,443 word tokens.
  - Each suite present under `data/eval/suites/` is measured on its `encounters.jsonl` dialogue and note.
  - The definitions follow the paper's appendix (pp. 9-10). The metrics are JSD unigram and JSD bigram
    (eps-smoothed over the union vocabulary, in bits, so in [0, 1]) and vocabulary Jaccard. For 17 statistics the
    file gives `|suite - reference|` and the suite's own value:
    - entropy, MTLD, Yule's K, Distinct-1 and Distinct-2, 4-gram self-repetition
    - sentence-length mean and std
    - punctuation, digit and capitalized token ratios
    - Flesch-Kincaid grade (syllable heuristic)
    - line breaks per 100 tokens
    - key-value, bullet, all-caps and special-character ratios
- `data/eval/drift.json` follows the contract (`reference`, `suites`, `complexity_rank`, where rank 1 is the suite
  closest to ACI train by JSD unigram, ties by name). It always covers every suite present in `data/eval/suites/`.
  `--suites` recomputes the named suites. Each other entry is kept only while the reference, `drift.py`
  (`reference.code_sha1`) and that suite's `encounters.jsonl` (`input_sha1`, the same value as in its `suite.json`)
  are unchanged; otherwise it is recomputed too, so a suite rebuilt after the last drift run never keeps a stale
  entry or rank (review fix). `all` passes its `--suites`, so a default `all` recomputes everything (about 8 s). The
  output is deterministic: values are rounded to 6 decimals and the float sums use `math.fsum`. Two runs gave
  byte-identical files.
- `eval/docs/CORPUS-DRIFT.md`: the corpora and tokenization rules, and one row per metric (key, family, definition,
  what a high value means, the paper's r with ΔF1). The r values are taken from the figure labels of Fig. 4 (p. 7,
  main) and Fig. 5 (p. 12, ablation). The doc also has TODO rows for perplexity (GPT-2), MMD, Fréchet,
  embedding-mean distance and cluster-coverage JSD, each with the model it needs, and caveats on reading the
  numbers. `eval/docs/README.md` gets a `drift` row in the commands table, a "Corpus drift (bonus)" section and the
  layout line.
- `eval/tests/test_drift.py` (32 tests):
  - Identical corpora: both JSDs are 0, Jaccard is 1 and all 17 diffs are 0.
  - A corpus with every word type renamed: Jaccard 0, JSD unigram and bigram 1 in bits (ln 2 in nats).
  - JSD 0.5 for half-shared mass.
  - Known MTLD (3.0, 4.48, all-unique), Yule's K, entropy, Distinct-1/2, self-repetition, n-grams not crossing texts.
  - 10 syllable cases and the FK grade (-2.62).
  - Sentence splitting (abbreviations, initials, ordinals, ACI's spaced periods).
  - Every structural and token-class ratio on a hand-counted form, 9 bullet openings, and the literal `\n` escape.
  - The drift.json contract and determinism. An unselected suite is kept while unchanged, and recomputed when its
    `encounters.jsonl` or the drift code changed. A missing reference or missing suites.

### Deviations

- `reference` gets one extra key, `values`, with the reference's own statistics: the paper reports m_A, m_B and
  |m_A - m_B| (p. 9). Each suite entry also carries `records`, `tokens`, `types` and its own `<stat>` beside every
  `<stat>_diff`. The contract keys are unchanged. Two more extra keys, for the staleness check: `reference.code_sha1`
  (sha1 of `drift.py`) and `input_sha1` in each suite entry.
- Bullet openings go beyond the paper's "bullet symbol or an ordinal with a trailing period" (p. 10): an ordinal
  closed by a parenthesis (`2)`, `a)`, `(1)`, `(a)`) and a bullet glyph glued to its word (`•Fever`) count too. On
  the 13 suites and the reference, the bullet ratio is identical under the strict paper rule.
- Choices the paper leaves open, documented in CORPUS-DRIFT.md:
  - The log base is 2.
  - The pseudo-count is ε = 1e-9.
  - MTLD uses `TTR <= 0.72` and a partial last factor, and an all-distinct text counts as one factor.
  - Self-repetition is counted over n-gram occurrences.
  - The special characters are `@#$%^&*_~|\<>{}=+` and the backtick.
  - Punctuation is Unicode category P*.
  - Line breaks include the literal `\n` escape found in `pmc_patients` (293 of them).
- The "Suite complexity (drift from ACI)" report section and the complexity column of the robustness table are not
  written here: `report.py` belongs to WP5, which reads `drift.json`.

### Reference numbers reproduced

- The paper does not publish numbers for these corpora, so there is nothing to reproduce. Sanity anchors:
  - `aci_dev` (40 of the 140 reference encounters) is the in-distribution floor: JSD unigram 0.032, bigram 0.180,
    Jaccard 0.537.
  - `aci_eval` comes next at 0.043.
  - `imerit_multi` (non-English) is last at 0.843.
- `complexity_rank` over the 13 suites present: aci_dev 1, aci_eval 2, imerit_en 3, sadiqguru 4, robust_medconv 5,
  persuade_pii 6, pmc_patients 7, nbme_notes 8, robust_synthetic 9, pii_gpt 10, robust_discharge 11,
  devp_discharge 12, imerit_multi 13.
- After the review fixes, a full recompute on the current suites (the robust suites rebuilt at 17:55, aci_eval at
  18:03) gives the same values and ranks as before. Only the new `input_sha1` and `code_sha1` keys were added.

## WP2: One ACI-eval gold (eval/gold/)

### Done

- `eval/gold/taxonomy.py`: the label maps of the three golds (VM types, ketai categories) onto the harness's 11
  categories plus `CONTACT`; the DATE → DOB (another gold says DOB, or `date of birth|dob|born` within 25 characters
  before an occurrence) → YEAR (`^\d{4}$`) → DATE rule; `majority()` with tie priority harness > ketai > VM; `DIRECT`
  (PATIENT_NAME, PERSON_NAME, DOB, DATE, ID, CONTACT); concept types (`lab_or_test` → TEST, ANATOMY kept);
  `LEVELS` and `in_level()` (an item without `votes` belongs to every level).
- `eval/gold/merge_aci_gold.py` (`python3 -I eval gold [--audit]`, also runnable as a file, no arguments needed):
  writes `data/eval/gold/aci_eval/merged.jsonl`, `merged_stats.json`, `merged_review.md`; a changed `merged.jsonl`
  first moves to `merged.prev.jsonl`. Deterministic: a rerun is byte-identical (checked on all five files, audit
  included). Missing VM or ketai gold: merged without it (warning); missing `data/converted/aci_bench`: warning, exit 0,
  nothing written. Rows `{encounter_id, pii, pii_by_source{harness,vm,ketai}, keep, concepts}`, items
  `{field, text, category, risk, direct, sources, votes, nested_votes, source_labels, label, contains?, within?,
  standalone_occurrences}` as in the contract. `pii_by_source.harness` is built exactly like the old suite `pii`.
- `eval/gold/audit_gold.py` (alone or via `gold --audit`): `audit.md` + `audit.json` next to the merged gold:
  verbatim whole-word check per source and list, contradictions (PII ∩ keep, PII ∩ concepts, PII made of ICD-10
  description words, keep strings inside a longer PII item), regex candidates no union/keep item covers (ages, dates,
  title + name, e-mails, phones, digit runs ≥ 6), single-vote table, votes per category, diff vs
  `merged.prev.jsonl`. Exit 1 when merged.jsonl is missing.
- `metrics.privacy` (shared): classic keys on the consensus level, plus `entity_removed`, `levels.{union,consensus,
  direct}` = `{removed, removed_high_risk, entity_removed, items, occurrences}`, `by_source.<gold>` on each gold's own
  `pii_by_source` list, breakdown/counts keys `level:*`, `entity:*`, `votes:*`, `src:*`. Token-leak semantics
  (`leaked`, `_distinctive`) untouched; `key="pii_alt"` still works (WP0's test). Helpers `_outcomes`, `_summary`.
- `score.py` (shared): `privacy_alt` dropped. `common.CATEGORY_RISK` (shared): `CONTACT: high` (+ comment line).
- `suites.py` (shared): `GOLD_DIR`, `merged_gold()`, labels of a suite with `data/eval/gold/<suite>/merged.jsonl` come
  from it (`pii` = union items with metadata, `pii_by_source`, `keep`, `concepts`; no `pii_alt`); `suite.json` gets
  `gold_sources`, `pii_levels`, `merged_sha1`; the merged file is hashed into `source_sha1`; `--merged/--no-merged`
  (the old harness + VM path, VM `CONTACT → ID` kept there) is recorded as `merged_gold: false` and kept by `ensure()`
  like `vm_annotations`; `stale()` reports "merged gold switched on/off"; `gold/taxonomy.py` joins the build code
  (`BUILD_CODE`); the aci_eval `about` text; the summary line prints `gold harness+vm+ketai` instead of `alt-gold N`.
- `report.py`: not changed. It reads `privacy_alt` through `g()`, so the "privacy (VM gold)" column shows `–`; checked
  by rendering the report into a scratch dir (no crash).
- Tests: `eval/tests/test_gold.py`, 20 tests (taxonomy and DATE rules, majority, nesting and survival on a synthetic
  encounter, keep conflicts and rejected votes, concepts, level filters, privacy levels/by_source/breakdown, levels
  coinciding without metadata, deterministic main + prev file, missing sources, audit,
  the suite built from the merged gold and `--no-merged`/`--merged`, plus two data tests that skip without data:
  `ad_night` `by_source.harness` 0.8388 / `vm` 0.5398 with the real word list).
- Docs: `eval/docs/GOLD.md` (sources, harmonization table, merge rules, item schema, counts, levels, results, audit,
  caveats incl. LLM-assisted golds); README: `gold` command row, suites caching bullet, `aci_eval` row, new section
  "ACI gold and privacy levels", privacy/high-risk metric rows, layout and tests lines.

### Deviations

- The plan's "173 harness-only items" are the harness items not in the ketai gold (231), of which 173 are whole-word
  substrings of a ketai item. Harness items in neither other gold: 129, 107 of them inside a ketai item.
- Nesting applies to every shorter item, also from the same gold (VM lists `andrew campbell` and `andrew`); nested
  votes count only other golds. Standalone occurrences are counted on spans (an occurrence not inside any container
  occurrence) rather than `count(s) > Σ count(containers)`, which double counts nested containers.
- `votes` = golds that list the item, a part of it, or (for a survivor) a container of it. Added rule: a gold never
  votes, through nesting or inheritance, for a string it lists as not PII (its keep items, ketai's non-PII ambiguous
  terms, ketai's built-in `NOT_PII` words from `build_gold.py`: dax, dragon, doctor, patient, nurse, weekdays).
  Without it, ketai's container `inner marrial soccer league` made `soccer` (ketai: not PII; harness: code evidence)
  a 2-vote PII item.
- PII ∩ keep is decided on the item's own votes (golds that list it or a part of it), not inherited ones; the losing
  PII strings are dropped before nesting (second pass). 8 conflicts went to PII, 31 to keep.
- Merged `risk` follows `CATEGORY_RISK` (plan); the harness's 7 per-item risk overrides survive only in
  `pii_by_source.harness`. Items without a whole-word occurrence would be dropped (there are none).
- Breakdown `votes:*` and `src:*` are over the union items (`src:vm` = merged items VM supports); the other classic
  keys are consensus. `levels` and `by_source` carry `items` and `occurrences` beside the contract keys. Leaks list
  every union item, with `votes`/`direct` when present.
- `direct` does not saturate: `ad_night` 0.926, `anonymous` 0.982, because VM and ketai annotate bare month names as DATE.
- 16 keep strings sit inside union PII items (e.g. `hay` in `moving hay`), and 31 VM items lost to keep.
- A DOB cue before any occurrence makes a VM/ketai DATE a DOB, also for a nested survivor whose standalone occurrence
  has no cue (`january` in D2N021); both are direct and high risk.
- `merged.prev.jsonl` is how the audit diffs against the previous merge (deleted once after development, so the
  current audit says "no previous merged file").

### Reference numbers reproduced

- Exact overlap before nesting (merged_stats.json): all three 289, ≥ 2 votes 490, union 1,052; 231 harness items not
  in ketai, 173 of them inside a ketai item. After nesting: 1,052 − 235 absorbed − 31 keep wins = 786 items (union),
  538 consensus (= all items with ≥ 2 votes; every direct item has ≥ 2), 261 direct; votes 1/2/3 = 248/82/456 (456
  three-vote items vs 289 exact); occurrences 1,017 / 725 / 393. Keep 267, concepts 6,607 (none dropped as direct PII).
- `python3 -I eval suites --only aci_eval`: `encounters.jsonl` byte-identical, `input_sha1` a141a148e6b7 unchanged,
  so every model run stays cached; labels_sha1 808563adf82e, merged_sha1 d6f7f8ce4b73. Rebuilt again after the later
  `suites.py` edits: both files byte-identical. `run --models identity --suites aci_eval`: identity rerun (common.py in
  its fingerprint).
- Audit: 0 verbatim failures for PII and keep strings in all three golds (1 VM concept, `SPO2`, is not a whole
  word); 0 regex gaps; 18 PII-and-concept, 16 keep-inside-PII.
- Re-scored every model on aci_eval (`score --suites aci_eval --icd knn lexical`), final run at 18:10:

  | model | consensus | high-risk | entity | direct | union | harness | VM | ketai |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | `ad_night` | 0.7738 | 0.9404 | 0.7472 | 0.9262 | 0.5615 | **0.8388** | **0.5398** | 0.8211 |
  | `ad_night_submitted` | 0.7738 | 0.9404 | 0.7472 | 0.9262 | 0.5615 | **0.8388** | 0.5398 | 0.8211 |
  | `ad_night_nocodes` | 0.7793 | 0.9404 | 0.7509 | 0.9262 | 0.5664 | 0.8428 | 0.5452 | 0.8257 |
  | `ad_rules` | 0.7586 | 0.9384 | 0.7286 | 0.9237 | 0.5506 | 0.8279 | 0.5316 | 0.8058 |
  | `anonymous` | 0.9310 | 0.9821 | 0.9126 | 0.9822 | 0.8368 | 0.9444 | 0.8391 | 0.9465 |
  | `anonymous_rules` | 0.6097 | 0.8310 | 0.5558 | 0.8041 | 0.4562 | 0.6355 | 0.4458 | 0.6651 |
  | `identity` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

  `by_source.harness` and `by_source.vm` equal the WP0 baseline privacy and VM-gold privacy of all six real models to
  the fourth decimal. Utility moved with the new labels too: e.g. `ad_night` keep 1.0 → 0.9933, concepts 1.0 →
  0.9989, edit precision 0.9605 → 0.9852; `anonymous` keep 0.7452 → 0.7623, concepts 0.9509 → 0.9458.
- Tests: `test_gold.py` 20 passed; full suite `uv run --no-project --with pytest python3 -I -m pytest eval/tests -q`:
  274 passed (final run, other packages' tests included). `uvx --offline ruff check --line-length 160 eval`: clean.
  The 12 other suites rebuilt into a scratch dir give byte-identical `encounters.jsonl` and `labels.jsonl`.

### Left for other packages

- WP5: the report should show `privacy.levels`, `entity_removed` and `by_source` (the old "privacy (VM gold)" column
  is `by_source.vm.removed` now and shows `–` until then).
- Every suite is stale once ("build code changed": `suites.py` and `gold/taxonomy.py` are build code); their rebuild
  gives byte-identical `encounters.jsonl` and `labels.jsonl` (checked in a scratch dir), so no run is invalidated.
- `scripts/convert/common.py` has no `CONTACT` category (not WP2's file); only the merged ACI gold uses it.

## WP5: Report, tradeoff, CSV exports

### Done

- `eval/report.py` rewritten. `python3 -I eval report` (about 0.6 s) reads only `scores/`, `runs/*/*/run.json`, `attacks/`,
  `drift.json`, `suites/*/suite.json`, `gold/aci_eval/merged_stats.json`, `eval/models.toml` and `eval/coders.toml`. It
  writes, under `data/eval/`:
  - `report.md`
  - `report.json`
  - `results.csv`
  - `results_headline.csv`
  - `tradeoff.svg`
  - `report_discord.md`, now generated (it used to be written by hand)
- How it works: every value first becomes a row of the long table (`Results`), and every markdown cell is then looked
  up from those rows. So a number can only appear in the markdown if it is also in the CSV.
- Graceful degradation:
  - a missing value shows `–`
  - a failed run shows `failed`
  - a coder or attack with an `error` shows `error`, plus a footnote with the message
  - unreadable or half-written JSON is skipped with a warning on stderr
  - a missing `coders.toml` or `models.toml` gives empty columns
  - a `run.json` with `exit_code: null` (a run in progress) is ignored
- Headline (`aci_eval`), with `identity` last as bounds. Columns:
  - privacy at the consensus, direct and entity level
  - privacy (harness gold) = `by_source.harness.removed`
  - concepts kept and ICD concepts kept
  - ICD F1 anon, retained and agree, from the coder with `primary = true` (`llm_note`)
  - from the `aci_eval` run: ms/enc, peak RAM, and whether peak VRAM is ≤ 4096
  - explained
  - attack recovered (max)
- Sections:
  - data status: which models still lack a primary-coder result, failed runs, missing `drift.json`
  - privacy vs utility, with a Pareto flag and a link to `tradeoff.svg`
  - privacy levels (union, consensus, direct, high-risk, entity level per level) and the gold counts from `merged_stats.json`
  - votes and sources (`votes:1/2/3`, `src:*`, each gold alone)
  - ICD coders on `aci_eval` (model × coder: F1 orig, F1 anon, Δ [CI95], retained, agree, cat-F1, errors, `!` when
    `valid_for_comparison` is false), and the same on the other suites. The coder columns are dynamic: the
    `coders.toml` order with the primary coder first, then any other coder found in the scores.
  - the main-suite tables for `aci_eval` and `aci_dev` (valid, privacy, high-risk, keep, non-PII kept, edit precision,
    concepts, ICD concepts, terms, placeholders, explained)
  - "Without ICD-10 codes at inference" (from `nocodes.rows()`)
  - adversarial attacks on `aci_eval` (anonymized / control per attack, max), and the max per suite for the suites a real
    model was attacked on
  - robustness (privacy, high-risk and terms per robust suite, with the complexity rank)
  - suite drift ordered by JSD unigram
  - the Kaggle tables: privacy, terms, utility with gold, by language
  - `aci_eval` privacy by category and field
  - efficiency
  - an updated "How to read this"
- `results.csv`:
  - long format, written with stdlib `csv`, UTF-8 with BOM, comma delimiter, full stored precision
  - column header exactly as in the plan
  - categories exactly the 15 in the plan
  - the `level` and `metric` conventions are documented in the module docstring and the README
- `results_headline.csv`: one row per model, the `HEADLINE` keys plus `reference` and `pareto`.
- `report.json`:
  - `headline`, `tradeoff` (points, Pareto flags, the y metric and whether it fell back), `icd`, `nocodes`, `attacks`,
    `robustness`, `complexity_rank`, `efficiency`, `suites`
  - `exports`: the paths of the four files
- `tradeoff.svg` (standard library):
  - x = consensus privacy on 0–1; y = primary-coder ICD F1 retained
  - y falls back to ICD concepts kept while no real model has a primary-coder result, and the axis title says so
  - filled markers are Pareto-optimal, hollow ones dominated, squares are `identity`; identical points share one
    label
  - labels are placed greedily (free of markers and other labels, with leader lines)
  - explicit background; light colors as attributes, dark ones in a `prefers-color-scheme` block; system font stack;
    native `<title>` tooltips
  - checked visually through `rsvg-convert`
- `eval/tests/test_report.py`: 8 tests on a synthetic tree in `tmp_path`, with `report.OUT`, `MODELS_TOML` and
  `CODERS_TOML` monkeypatched:
  - every headline cell equals its CSV row after `fmt` rounding (60+ cells)
  - spot formats
  - Pareto flags on a 3-model example (alpha and beta optimal, gamma dominated) and `pareto()` with ties and missing values
  - missing data, a failed run, a coder error, an attack error and a still-running run render without crashing
  - an empty tree and a half-written JSON
  - the y-axis fallback
  - BOM and exact headers of both CSVs, and the 15 categories
  - the CI and `n` columns
  - the `exports` paths
  - the SVG parses with `xml.etree`
  - the discord file
  - a leaked string planted in the score files reaches no output
- `eval/docs/README.md`: the `report` row of the commands table, and a new section "Report and exports" (the files,
  the CSV conventions, the Pareto rule).

### Deviations

- ms/enc, peak RAM and "peak VRAM ≤ 4096" in the headline come from the `aci_eval` run. The efficiency section keeps the
  all-suite aggregate. `ad_night_submitted` (precomputed) therefore has `–` there.
- The rule "a model is Pareto-optimal if no other model is ≥ on both" is applied with "and > on one". Two models with
  equal values (`ad_night` and `ad_night_submitted`) are therefore both optimal. References are never in the comparison.
- The headline shows "explained" as a share (0–1), like every other share column, not as a percentage.
- Kaggle tables cover roles `kaggle` and `generic`. The robust suites moved to their own section. The old "utility with
  gold" columns for devp ICD (lexical) moved to the ICD table for the other suites, which is dynamic per coder.
- `results.csv` conventions not fixed by the plan:
  - `level = consensus` on the classic `privacy.removed` / `removed_high_risk` / `entity_removed` rows of every suite;
    on a suite without vote metadata every level is the same
  - `gold:<source>` for `by_source`
  - `union` for the `votes:*` / `src:*` breakdowns
  - no-codes deltas are rows `delta_vs_base:<metric>` on the no-codes model
  - medical Kaggle means use suite `kaggle_medical_mean`
  - the drift reference uses suite `reference`
  - merged-gold counts are rows with an empty model
- `report.py` was formatted with `ruff format --line-length 160`; other files were not touched.

### Reference numbers reproduced

- `python3 -I eval report` on the real data at 18:54: no crash, 8 models, 4,737 CSV rows.
- Spot checks, `results.csv` against the score files:

  | Check | Value |
  |---|---|
  | `ad_night` `gold:harness` removed | 0.8388 |
  | `identity` consensus | 0.0 |
  | `anonymous` direct | 0.9822 |
  | `ad_night` lexical `f1_delta` | -0.0065 [-0.0213, 0.0] |
  | `ad_night` attack `max_recovered` | 0.1549 |
  | `anonymous` `aci_eval` ms/enc | 2256.1 |
- A heuristic pass over every numeric table cell of the real `report.md` found a matching CSV value or CI for that model
  for all 1,265 numeric tokens.
- Tradeoff without `llm_note` results (the fallback, ICD concepts kept): Pareto-optimal `anonymous`, `ad_night`,
  `ad_night_nocodes`, `ad_night_submitted`; `ad_rules` and `anonymous_rules` dominated.
- Tests: `test_report.py` 8 passed; full suite `uv run --no-project --with pytest python3 -I -m pytest eval/tests -q`
  286 passed; `uvx --offline ruff check --line-length 160 eval` clean.

### Left open

- No score file had `icd.llm_note` / `llm_both` while this ran. The ICD headline columns show `–`, and the tradeoff uses
  the fallback axis. The next `eval report` after the LLM warm-up picks them up with no code change. Models without the
  coder then show `–` and are listed under "not plotted" in the SVG.
- A4 appears as a column once any attack file has it. An `error` without arms renders as `error`.
- The GPU models (`ad_night`, `ad_night_nocodes`, `anonymous`) have no attack or robust-suite rows for most suites until
  their reruns.

## WP3: Adversarial attacks and robustness

### Done

- `eval/bonus/attack.py`: A1 to A5 ported from the A+D package and made generic over every `runs/<model>/<suite>/output.jsonl`;
  every attack also runs on the original text as positive control. Output `data/eval/attacks/<model>/<suite>.json` =
  `{model, suite, n_records, gold_level, attacks: {A1..A5: {name, anonymized, control, notes, error?}}, max_recovered, sources}`,
  each arm `{recovered, hits, total, by_category}`; `max_recovered` leaves A5 out. All 29 attack files checked against this
  contract (no missing keys, `max_recovered` consistent, no `error`). An arm whose total is 0 (A3 on `robust_discharge` and
  `devp_discharge`, A2 on `nbme_notes` and `pmc_patients`) has `recovered: null`.
- A1 (NER residual scan) runs `bonus/_ner_worker.py` with the anonymous venv; A4 (LLM inference) goes through
  `eval.coders.ollama.generate_json` with a strict JSON schema. Both cache per text under `data/eval/cache/attacks/{ner,llm}/`
  (A4 keyed by model, digest, schema and prompt). The cache is now written through a temporary file and an atomic rename, and an
  unreadable entry is recomputed, so a run killed mid-write resumes cleanly (`cache_read`/`cache_write`, test added).
  A4 prints progress every 20 new calls and records `new_calls`.
- A4 run on `aci_eval` for `identity`, `ad_night`, `anonymous` (266 cached LLM calls in total).
- `eval/bonus/robustness.py` + `scripts/convert/robust_{discharge,medconv,synthetic}.py`: suites `robust_discharge` (166),
  `robust_medconv` (80), `robust_synthetic` (24), role `robust`, with WP4's complexity rank. `python3 -I eval robustness`
  prints its table.
- `python3 -I eval attack` with no arguments (what `eval all` runs: A2, A3, A5 on `aci_eval`, every model with a run) takes
  20 s for 8 models; a missing run, failed run, unbuilt suite or suite without PII gold prints `skipped: <reason>`.
- Cost of each attack documented in `eval/docs/README.md` (Adversarial attacks section) and in the `attack.py` docstring.
- Tests: `eval/tests/test_attack.py`, `test_robustness.py`; full suite 287 passed; `ruff check --line-length 160 eval` clean.

### Deviations

- Scope cut for time: A4 ran only for `identity`, `ad_night`, `anonymous`. Deferred for every other model
  (`ad_night_nocodes`, `ad_night_submitted`, `ad_rules`, `anonymous_rules`); with an idle GPU it takes ~1.5 min per model:
  `python3 -I eval attack --models ad_night_nocodes ad_night_submitted ad_rules anonymous_rules --suites aci_eval --attacks A4`
  (the control is already cached; earlier A2/A3/A5 results are kept).
- No new A1 runs: A1 exists for `identity`, `ad_night`, `anonymous` on `aci_eval` only. Deferred:
  `python3 -I eval attack --models ad_night_nocodes ad_night_submitted ad_rules anonymous_rules --suites aci_eval --attacks A1`
  (never together with an A4 or ICD job on the GPU).
- A4 measured 1.4 s per call on an idle GPU and ~6 s per call while WP1's ICD coder shared Ollama (6 to 7 min per model).
- A1 is not independent for the anonymous models: it uses the same detectors (stated in the notes and the README).
- The robust suites have runs only for `ad_rules`, `anonymous_rules`, `identity`; `ad_night` and `anonymous` were not run
  on them (`python3 -I eval robustness --run`).

### Reference numbers reproduced

Attacks on `aci_eval` (67 encounters, consensus gold), anonymized / control recovered share:

| model | A1 | A2 | A3 | A4 | A5 | max |
|---|---:|---:|---:|---:|---:|---:|
| ad_night | 0.136 / 0.942 | 0.017 / 1.000 | 0.155 / 1.000 | 0.105 / 0.803 | 1.000 / 1.000 | 0.155 |
| ad_night_nocodes | – | 0.017 / 1.000 | 0.155 / 1.000 | – | 1.000 / 1.000 | 0.155 |
| ad_night_submitted | – | 0.017 / 1.000 | 0.155 / 1.000 | – | 1.000 / 1.000 | 0.155 |
| ad_rules | – | 0.017 / 1.000 | 0.197 / 1.000 | – | 1.000 / 1.000 | 0.197 |
| anonymous | 0.018 / 0.942 | 0.021 / 1.000 | 0.155 / 1.000 | 0.007 / 0.803 | 1.000 / 1.000 | 0.155 |
| anonymous_rules | – | 0.262 / 1.000 | 0.437 / 1.000 | – | 1.000 / 1.000 | 0.437 |
| identity | 0.942 / 0.942 | 1.000 / 1.000 | 1.000 / 1.000 | 0.803 / 0.803 | 1.000 / 1.000 | 1.000 |

- Checked: control > anonymized for every real model and every attack that ran (A5 excepted: an upper bound by design, 1.0 on
  both arms); `identity` anonymized = control for every attack.
- A4 by attribute (name / age / place / employer or job): control 0.968 / 0.963 / 0.182 / 0.292; `ad_night` 0 / 0.185 / 0 / 0.25;
  `anonymous` 0 / 0.019 / 0 / 0.

Robustness (`python3 -I eval robustness`), privacy / high-risk privacy / terms kept:

| model | robust_discharge (rank 11) | robust_medconv (rank 5) | robust_synthetic (rank 9) |
|---|---|---|---|
| ad_rules | 0.985 / 1.000 / 1.000 | 0.968 / 1.000 / 0.999 | 0.954 / 0.983 / 0.998 |
| anonymous_rules | 0.672 / 0.733 / 1.000 | 0.467 / 0.724 / 1.000 | 0.450 / 0.575 / 0.999 |
| identity | 0 / 0 / 1.000 | 0 / 0 / 1.000 | 0 / 0 / 1.000 |

## WP1: ICD coders

### Done

- `eval/coders/`: `base.py` (`Coder`, `Prediction`, `CoderError`), `codes.py` (daniel-eval's normalization, `None`
  on an invalid code, junk characters stripped first, canonical dot, `category()`), `bootstrap.py` (daniel-eval's
  `rates`/`interval` and the paired encounter bootstrap, 1000 resamples, seed 0), `ollama.py` (daniel-eval's client:
  strict JSON schema, temperature 0, seed, `think: false`, `done_reason == "stop"`, context guard, digest from
  `/api/tags`, no proxy or redirect; `OLLAMA_HOST`, `endpoint` for a non-loopback server, `PROMPT_NOTE` verbatim,
  `PROMPT_BOTH`, `unload()`, `backend()`), `knn.py`, `lexical.py`, `command.py` (timeout, per-record
  `missing_output`/`invalid_output`), `text.py`.
- `eval/icd.py`: registry (`coders.toml`: `llm_note` primary, `llm_both`, `knn`, `lexical`, `paper` disabled), cache
  `data/eval/cache/icd/<coder>/<sha1(config, id, fields read)>.json` (errors never cached, old plain lists read),
  set scoring with ranked coders cut at k = |gold|, code and category levels for original, anonymized and agreement,
  `f1_delta` with paired CI, `f1_retention`, `errors`, `invalid_codes`, `valid_for_comparison`,
  `changed_predictions`. A coder failure gives `{"error": ...}` and scoring goes on. `score.py --icd` takes
  registry names or `cmd:<template>`; `run.main` unloads Ollama models before measuring.
- `device = "auto"` never sends `num_gpu` (only `"cpu"` sends `num_gpu: 0`). Every one of the 309 cached `llm_note`
  predictions records `backend: gpu`.
- This session: the score block now has exactly the agreed shape. The `backend` key was dropped from
  `icd.<coder>`; it stays in each cache file. `test_icd.py` asserts the full key set of the block, its arms and levels.
- Cache warmed for `llm_note` on `aci_eval`, every model with a run: 309 predictions (67 originals, shared by
  `identity`; then 56 `ad_night`, 0 `ad_night_submitted`, 3 `ad_night_nocodes`, 9 `ad_rules`, 64 `anonymous`,
  51 `anonymous_rules`), 0 errors. Wall time about 26 min in four commands
  (`python3 -I eval score --suites aci_eval --icd llm_note knn lexical --models ...`), mean 4.7 s per call: the GPU
  was shared with WP3's Ollama calls (2.3 to 6 s per call).
- `aci_dev` and `devp_discharge` re-scored with `--icd knn lexical` (no LLM), so every score file has the new shape.
- Tests: `test_codes.py`, `test_bootstrap.py`, `test_coders.py`, `test_icd.py`. Full suite 287 passed, ruff clean.

### Deviations

- `backend` is not part of the score block (see above), only of the cache files.
- Daniel cross-check done for the `identity` pair only. The `ad_night` pair was stopped to save time (see below).
- Deferred to a faster or idle machine, with the commands:
  - Daniel cross-check on `ad_night`, from a scratch copy of `daniel-eval/` (it refuses an `--out-dir` outside its
    own `outputs/`), `R=<repo>/data/eval`:
    `python3 -I -B scripts/evaluate_icd.py --original $R/suites/aci_eval/encounters.jsonl --anonymized $R/runs/ad_night/aci_eval/output.jsonl --out-dir outputs/ad_night --device auto --model qwen3.5:2b-q4_K_M --seed 0`
    (add `--resume` to continue a partial run dir). Expected to match our `ad_night` anonymized micro F1 0.0915.
  - `llm_both` on `aci_eval`: `python3 -I eval score --suites aci_eval --icd llm_note llm_both knn lexical`.
    It is about 376 cold calls with dialogue + note prompts, and Ollama reloads the model at `num_ctx` 24576.
  - `llm_note` on `aci_dev` and `devp_discharge`: `python3 -I eval score --suites aci_dev devp_discharge`.
  - Both are in the default `--icd` set, so a plain `eval score` or `eval all` runs them cold. Pass
    `--icd llm_note knn lexical` to skip them.

### Reference numbers reproduced

- Daniel's CLI (`evaluate_icd.py`, GPU, `--device auto`, same model and seed) on `aci_eval` originals vs `identity`:
  original and anonymized micro P/R/F1 0.1014/0.0915/0.0962 (tp 14, fp 124, fn 139), mean per-note F1 0.0817,
  CI95 [0.0550, 0.1379], delta 0.0 [0, 0], 0 errors. Ours (`llm_note`): identical on every number to 1e-4.
  The codes match record by record on all 67 originals. Daniel's second (stopped) run reproduced its own 56
  original predictions byte for byte, so the generation is deterministic.
- Regression: all 37 `knn`/`lexical` blocks in `data/eval/scores_baseline/*/*.json` (`aci_eval`, `aci_dev`,
  `devp_discharge`) have `original.code.f1` and `anonymized.code.f1` equal to the old `f1_original` and
  `f1_anonymized`. No gold code in these three suites changes under normalization: the converters already write
  them canonically. The 11 "undotted" codes are 3-character categories such as `I10`.
- `llm_note` on `aci_eval` (code level; delta = anonymized minus original, paired bootstrap CI95):

  | Model | orig F1 | anon F1 | f1_delta [CI95] | retention | agreement F1 |
  |---|---:|---:|---:|---:|---:|
  | `ad_night`, `ad_night_submitted`, `ad_night_nocodes`, `ad_rules` | 0.0962 | 0.0915 | -0.0047 [-0.0198, +0.0039] | 0.9515 | 0.855 |
  | `anonymous` | 0.0962 | 0.0915 | -0.0047 [-0.0199, +0.0043] | 0.9515 | 0.6617 |
  | `anonymous_rules` | 0.0962 | 0.0903 | -0.0059 [-0.0208, +0.0024] | 0.9382 | 0.8278 |
  | `identity` | 0.0962 | 0.0962 | +0.0000 [0, 0] | 1.0 | 1.0 |
