# Changelog

## 1.0.0 (18 October 2026)

### Changed
- Default run: mode `llm` (rules + `qwen3.5:2b-q4_K_M` via Ollama, no PyTorch), preset `privacy`,
  features F0 and F2. Compose uses the 147 MB image; the PyTorch `full` mode stays available.

### Added (M5–M8)
- F0 code-evidence guard: activities are kept only when the sentence tells an injury mechanism, and then
  in every mention. F2 targeted LLM detector on personal segments, with occupations and a replay cache.
  `privacy` preset. F4 agreement rule (measured, opt-in).
- `llm` mode, `--detector-model` with digests pinned in `models.toml`, per-detector time, LLM calls,
  cache hits and VRAM in the summary, edits by detector family in the scorer.
- Unified scorer (both teams' metrics), `scripts/crosscheck_scorers.py`, `gate`, `icd-judge` hook,
  `scripts/error_analysis.py`, `docs/error_analysis.md`.
- `full` image (CPU torch), `compose.yaml`, `compose.gpu.yaml`, `docs/docker.md`.
- `docs/solution.html`, generated from saved results by `scripts/build_solution_page.py`.

### Removed
- The GitHub Actions CI workflow (not needed for the final run; tests run locally with `uv run pytest`).

### Fixed (M6)
- Ollama calls honour `UNIFIED_HARNESS_OLLAMA_URL` (they were hard-coded to localhost).
- Fail-closed tests: unreachable LLM, malformed input, failure while publishing.

## 0.1.0

### Added (M2)
- Merged silver labels (`eval/labels.py`, `scripts/labels/merge.py`): union of the upstream and Ketai
  eval labels with overlap-based agreement, conflict listing and adjudication.
- Splits tune / validation / test in `configs/splits.json`; scoring uses `labels/merged/<split>/`.
- `docs/evaluation.md` (labels, splits, metrics, gate).
- Release guard `scripts/check_no_patient_text.py` with a reviewed allowlist.

### Fixed (M2)
- Ignore rules anchored to the project root; lexicons, label scripts and data tests are now tracked.
- Hard-case expectations that quote encounter text moved to the git-ignored `labels/` folder; examples
  from the challenge data in comments, docs and tests replaced with invented ones.

### Added (M1)
- Baseline pipeline ported from Alind+Davide's `neural_wave` (shield, rules, quasi-identifiers, formats,
  identity graph, policy, verification, encoders) into `text/`, `shield/`, `detect/`, `link/`, `policy/`,
  `verify/`; output byte-identical to upstream in rules mode.
- `anonymize` with explicit `rules` / `full` modes, fail-closed checks and the output contract;
  `evaluate`, `run`, `sweep`, `attack`, `robustness`, `show` commands.
- Evaluation modules (privacy, utility, ICD proxies, attacks, robustness, efficiency), label scripts,
  labeling guidelines, lexicon builder; 185 upstream tests ported (data tests skip without `data/`).
- Container smoke test that anonymizes the synthetic fixture.

### Fixed (M1)
- Encoders load from the pinned `models/` folder instead of the Hugging Face cache.

### Added (M0)
- Project scaffold: package `unified_harness`, `pyproject.toml` with locked dependencies, Python 3.12.
- Output contract: strict JSONL reading, input and output validation, dated output names
  (`anonymized-<YYYY-MM-DD>-unified-harness.*`), atomic publishing that refuses contract violations and
  never overwrites without `--overwrite`.
- `unified-harness doctor` (rules / full readiness, fails closed) and `unified-harness validate`.
- Pinned model manifest and `scripts/download_models.py`.
- Dockerfile with the `rules` target; CI workflow (lint, tests, offline image smoke test).
