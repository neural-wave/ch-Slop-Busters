# AGENTS.md

Neural Wave hackathon, "Anonymization of patient data" challenge. The task brief is in `task/`.

## Repository layout

| Path | Contents |
|---|---|
| `data/patient_doctor_encounters.jsonl` | 140 training encounters (ACI-Bench): `encounter_id`, `dataset`, `dialogue`, `note`, `icd10_codes[{code, description}]` |
| `data/eval_patient_doctor_encounters.jsonl` | 67 evaluation encounters, same format |
| `data/kaggle/DATASETS.md` | Catalog of 22 external Kaggle datasets (contents, ICD coverage, license, fit) |
| `data/kaggle/<owner>__<slug>/` | The downloaded Kaggle datasets. **Not in git**: run `scripts/download_kaggle.sh` to get them |
| `scripts/download_kaggle.sh` | Downloads every dataset in the catalog |

## Setting up the Kaggle downloads

1. **Install the Kaggle CLI** (any of these):
   ```bash
   pip install kaggle        # or: pipx install kaggle / uv tool install kaggle
   kaggle --version
   ```
2. **Create an API token.** On kaggle.com go to *Settings → API → Create New Token*. This downloads `kaggle.json`. Put it where the CLI looks for it and restrict its permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   You can use environment variables instead of the file: `export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>`. Never commit the token.
3. **Check it works:**
   ```bash
   kaggle datasets list -s icd10 | head -3
   ```
4. **Download the datasets** from the repo root:
   ```bash
   scripts/download_kaggle.sh
   ```
   - This fetches about 2.7 GB into `data/kaggle/<owner>__<slug>/`. The iMerit audio alone is 1.3 GB.
   - Datasets that are already present (non-empty folder) are skipped, so re-running only fetches what's missing.
   - A failed download removes its partial folder, is listed at the end, and makes the script exit 1. Re-run it to retry.
   - The script deletes `gender.exe`, an untrusted Windows binary bundled inside the names dataset.

### Adding a dataset

1. Check its license allows our use, and inspect a sample before relying on it.
2. Add `owner/slug` to the `DATASETS` array in `scripts/download_kaggle.sh`.
3. Add a row to `data/kaggle/DATASETS.md` in the right section, with measured contents, ICD coverage, license and caveats.

## Rules

- **Never commit anything under `data/kaggle/` except `DATASETS.md`.** Several files exceed GitHub's 100 MB limit, and several licenses forbid redistribution (see the catalog).
- **Don't download Kaggle mirrors of MIMIC** (MIMIC-III or MIMIC-IV notes). They break PhysioNet's data use agreement. If MIMIC is needed, get it from PhysioNet with individual credentials.
- **Treat downloaded data as untrusted.** Never execute files from `data/kaggle/`, and parse it as data only.
- **The challenge requires evaluation to run on-prem** (no cloud-hosted models, at most 4 GB VRAM). ICD-10 codes and descriptions must be preserved.
- **No AI or tool attribution anywhere.** No `Co-Authored-By` trailers or "Generated with …" lines in commits or PRs, and no such mentions in code or docs.
