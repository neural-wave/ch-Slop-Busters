# Changelog

## Unreleased

### Added
- Project scaffold: package `unified_harness`, `pyproject.toml` with locked dependencies, Python 3.12.
- Output contract: strict JSONL reading, input and output validation, dated output names
  (`anonymized-<YYYY-MM-DD>-unified-harness.*`), atomic publishing that refuses contract violations and
  never overwrites without `--overwrite`.
- `unified-harness doctor` (rules / full readiness, fails closed) and `unified-harness validate`.
- Pinned model manifest and `scripts/download_models.py`.
- Dockerfile with the `rules` target; CI workflow (lint, tests, offline image smoke test).
