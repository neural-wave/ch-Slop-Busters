# Data preparation

We turn the challenge files and the Kaggle downloads into one format, so the anonymizer and the evaluation run on all of
them unchanged. The format spec is in [`scripts/convert/README.md`](scripts/convert/README.md). Sources and licenses
are in [`data/kaggle/DATASETS.md`](data/kaggle/DATASETS.md).

## The two phases

| Phase | Status | Reads | Writes | Does |
|---|---|---|---|---|
| 1. Convert | done | `data/*.jsonl`, `data/kaggle/` | `data/converted/<name>/` | Reformats each source into the challenge schema and labels only the PII the source already annotates. Never invents PII |
| 2. Inject | planned | `data/converted/<name>/` | `data/injected/<name>/`, same format | Adds synthetic PII with exact spans where the phase-1 gold is missing or partial |

Each dataset folder holds:

- `encounters.jsonl`: exactly the challenge schema (`encounter_id`, `dataset`, `dialogue`, `note`, `icd10_codes`). This
  is the only file the anonymizer reads.
- `labels.jsonl`: the gold, never shown to the anonymizer. It has `pii` spans to remove, `keep` spans that must survive,
  where the codes come from (`icd10_source`: `gold`, `mapped`, `none`) and how complete the PII labels are (`pii_gold`:
  `exhaustive`, `partial`, `none`).
- `manifest.json`: source, license, how the gold was made, counts.
- `gold/`: the same gold in the harness's per-encounter format, written by `export_gold.py`.

```bash
scripts/download_kaggle.sh
python3 -I scripts/convert/convert_all.py --aci-gold-dir <harness>/outputs/gold/eval <harness>/outputs/gold/dev
python3 -I scripts/convert/export_gold.py data/converted/*/
```

## Scripts (`scripts/convert/`)

| Script | Does |
|---|---|
| `common.py` | Shared code: ICD-10-CM 2023 lookup, the `Labels` span builder, the writer that enforces the schema |
| `convert_all.py` | Runs every converter (skips sources not downloaded), then `validate.py` |
| `validate.py` | Checks schema, span offsets, categories and counts. Exits 1 on errors |
| `export_gold.py` | Writes `gold/<encounter_id>.json` (`pii` + `keep` as distinct strings) for the harness |

| Converter | Output | ICD-10 | Privacy gold |
|---|---|---|---|
| `aci_bench` | The 207 challenge encounters (140 train + 67 eval), unchanged | gold | Harness gold for 107 (67 eval + 40 train), exhaustive |
| `devp_discharge` | Synthetic discharge summaries rendered from JSON as a form or a letter. 50 of 500 by default, one per patient and hospital | gold, 5 codes | Exhaustive (every rendered personal field) |
| `imerit_scribe` | 199 transcripts + structured notes in EN/DE/FR/ES/NL. Speaker tags are inferred by alternation | none | Partial: patient name, physician, DOB |
| `sadiqguru_conversations` | 4 long English dialogues | none | Partial: names, years, ages (hand-labelled) |
| `mtsamples` | 4,966 dictated notes, 40 specialties, line breaks restored | none | None (some unlabelled names and dates) |
| `pmc_patients` | 2,000 sampled published case reports (`--limit 0`: 167k) | none | Partial: patient age |
| `nbme_notes` | 1,000 exam notes rebuilt from tokens (some punctuation lost) | none | Partial: patient age. The other graded findings are `keep` spans |
| `medconv_corpus` | 5,609 real patient questions + doctor answers | none | None (real names, unlabelled) |
| `codiesp` | 3,751 Spanish case reports | gold for 1,000 with `--codes` (Zenodo TSVs) | None |
| `symptom2disease` | 1,200 one-turn symptom descriptions | mapped (24 labels) | None |
| `medconv2disease` | 960 short LLM chats | mapped (24 labels) | None |
| `pii_external`, `persuade_pii`, `pii_gemini`, `pii_gpt` | Non-medical essays with injected PII | none | Author's name, IDs, contacts, addresses. Exhaustive for `persuade_pii`, partial otherwise |

## Conclusions: which dataset for which evaluation

Two properties of the official metrics decide this:

- **Privacy** is the proportion of private information removed, which is recall. `partial` gold is enough for it.
- **Utility** (ICD-10 F1, medical concept recall) compares the anonymized text with the original. Any clinical text
  works without gold codes: run the same ICD predictor or concept extractor on both versions. Gold codes are only
  needed to check the predictor itself.

| Evaluation | Use | Notes |
|---|---|---|
| Privacy, main score | `aci_bench` | The reference. Keep its 67 eval encounters as the held-out set |
| Privacy, other formats | `devp_discharge`, `imerit_scribe`, `sadiqguru_conversations` | devp has exact gold but only 10 patients and 5 hospitals. imerit covers names in 5 languages |
| Privacy, single categories | `pmc_patients`, `nbme_notes` (age), essay sets (IDs, e-mails, phones, addresses, names) | Essays are not medical. Use them for the generic PII layer |
| ICD-10 F1 against gold | `aci_bench`, then `devp_discharge` (5 codes) and `codiesp` (with `--codes`, Spanish) | `symptom2disease`/`medconv2disease` codes are mapped from 24 labels: sanity check only |
| ICD-10 F1, original vs anonymized | `aci_bench`, `mtsamples`, `pmc_patients`, `imerit_scribe` (EN) | No gold codes needed |
| Medical concept recall | `nbme_notes` (graded findings as gold), plus original vs anonymized on `aci_bench`, `mtsamples`, `pmc_patients` | |
| Name vs diagnosis traps (Alzheimer, Parkinson) | `aci_bench` only (80 EPONYM + 125 NOT_A_NAME keep spans) | `mtsamples`, `pmc_patients` and `medconv_corpus` have about 1,900 unlabelled eponyms that could be labelled automatically |
| Bonus: without ICD-10 | Every dataset with `icd10_codes: []`, or `aci_bench` with the codes removed | |
| Bonus: robustness | `imerit_scribe` (4 other languages, dialogue + note), `codiesp`, `mtsamples`, `pmc_patients`, `medconv_corpus` | |
| Bonus: adversarial attacks | `pmc_patients` (rare conditions + age + timeline), `medconv_corpus` (real names and places) | |
| Training | Little usable as is | Medical text with complete PII labels exists only in `aci_bench`, which is our evaluation set. The essay sets label only the author's name, so a model trained on them learns to ignore other names. Phase 2 is what makes training data. Split by `meta.split` (ACI), `meta.scenario` (iMerit), `meta.patient` (devp) |

Weak or limited: `pii_gemini` (none of its listed e-mails appear in the text, and names are unlabelled), NBME text (lost
`/` and `-`), `mtsamples` (unlabelled names and dates). `imerit_scribe` and `pmc_patients` are non-commercial
(CC BY-NC-SA).
