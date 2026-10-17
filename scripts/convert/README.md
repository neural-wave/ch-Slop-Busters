# Dataset converters

Stage 1 of the data pipeline. One script per dataset turns our challenge data and the Kaggle downloads
(`data/kaggle/`, see `DATASETS.md`) into one common format. The anonymizer can then run on any of them unchanged,
and a later evaluation step can score them with the same code.

| Stage | Reads | Writes | Does |
|---|---|---|---|
| 1. convert (`scripts/convert/`) | `data/kaggle/`, `data/*.jsonl` | `data/converted/<name>/` | Reformats only. Labels the PII the source already annotates. Never invents PII |
| 2. inject (planned) | `data/converted/<name>/` | `data/injected/<name>/`, same format | Adds synthetic PII with gold spans where `pii_gold` is too weak |

Stage 1 never invents PII. The devp discharge notes are rendered from the record's own JSON fields, and nothing is
added to them. Anything synthetic belongs to stage 2, so the stage-1 output stays a faithful copy of the source.

```bash
python3 -I scripts/convert/convert_all.py            # everything with defaults (~1 min), then validate
python3 -I scripts/convert/convert_all.py --aci-gold-dir <harness>/outputs/gold/eval <harness>/outputs/gold/dev
python3 -I scripts/convert/pmc_patients.py --limit 0 # one dataset, own options (--help on any script)
python3 -I scripts/convert/validate.py               # re-check data/converted/*
python3 -I scripts/convert/export_gold.py data/converted/*/  # harness gold files into data/converted/<name>/gold/
```

Running `convert_all.py` without `--aci-gold-dir` rewrites `aci_bench` without its privacy gold. Use `--only`/`--skip`
to re-run other converters.

Standard library only (Python 3.10+). Output goes to `data/converted/<name>/`, which is gitignored because it is derived
from data whose licenses don't allow re-sharing.

## Format

Each dataset folder holds three files:

| File | Contents |
|---|---|
| `encounters.jsonl` | **Exactly the challenge schema**, in this key order: `encounter_id`, `dataset`, `dialogue`, `note`, `icd10_codes[{code, description}]`. A field the source lacks is `""`. Datasets without codes have `icd10_codes: []`. This is the only file the anonymizer reads. |
| `labels.jsonl` | Evaluation gold, one line per encounter, same order. **Never give it to the anonymizer.** |
| `manifest.json` | Source, license, language, origin, how ICD-10 and PII gold were obtained and how complete they are, and counts. |

`export_gold.py` adds a `gold/` folder with the same gold in the harness's format (see below).

`dialogue` follows the challenge format: one turn per line, `[doctor] …` / `[patient] …`. Source casing and punctuation
are kept (the challenge dialogues are lowercased and tokenized, these are not). Free-text documents go in `note`.

A `labels.jsonl` line:

```json
{
  "encounter_id": "devp-200002",
  "icd10_source": "gold",
  "pii_gold": "exhaustive",
  "pii":  [{"field": "note", "start": 66, "end": 80, "text": "Karan Malhotra", "category": "PATIENT_NAME", "risk": "high", "label": "patient.name"}],
  "keep": [{"field": "note", "start": 189, "end": 213, "text": "type 2 diabetes mellitus", "category": "CODE_EVIDENCE", "why": "E11.9"}],
  "meta": {"source_file": "med_doc_discharge_summary_200002_noisy.jpg", "document_type": "discharge_summary", "layout": "letter", "language": "en"}
}
```

- **`icd10_source`**: `gold` (codes from the source annotation), `mapped` (a fixed label-to-code table, coarse), or
  `none`. Descriptions are the official ICD-10-CM 2023 long descriptions, the same ones the challenge files use.
- **`pii_gold`**: how complete this encounter's `pii` list is.
  - `exhaustive`: every private item is labelled, so removing an unlabelled span counts as over-removal.
  - `partial`: the labelled items are real PII, but there are others, so only recall is meaningful.
  - `none`: no privacy gold, and the empty list means nothing.

  It is per encounter because a dataset can mix levels: ACI-Bench has 107 `exhaustive` and 100 `none`. The manifest's
  `pii_gold` describes what is labelled and why.
- **`pii`** is every occurrence, as character offsets into `field` (`text == record[field][start:end]`), sorted and
  non-overlapping. `category` and `risk` use the taxonomy of the harness's `gold_annotation_guidelines.md`
  (`PATIENT_NAME`, `PERSON_NAME`, `DOB`, `ID`, `DATE`, `AGE`, `YEAR`, `LOCATION`, `ORGANIZATION`, `OCCUPATION`,
  `OTHER_QI`), so all datasets score together with the ACI-Bench gold. `label` keeps the source's own label.
  `PATIENT_NAME` is the data subject's name (the patient, or the author of an essay).
- **`keep`** lists spans that must survive: `CODE_EVIDENCE` (the text behind a gold code), `CLINICAL_FEATURE` (NBME's
  graded findings, concept-recall gold), and the harness's `EPONYM`, `CLINICAL_NUMBER` and `NOT_A_NAME` traps.
- **`meta`** always has `language` (ISO 639-1). The rest is dataset-specific: split, original ids, specialty, source
  label, and so on.

To score with the private harness, `export_gold.py` writes its text-based gold: one `<encounter_id>.json` per encounter
whose `pii_gold` is not `none`, as `{encounter_id, pii: [{field, text, category, risk}], keep: [{field, text, category,
why}]}`, deduplicated on `(field, text.lower())`. That is the format `aci_bench.py --gold-dir` imports: exporting
ACI-Bench and importing it again reproduces its labels exactly. Not yet checked against the harness's own loader.
`CLINICAL_FEATURE` (NBME) is our keep category, not the harness's. The files don't carry `pii_gold`, so on `partial`
datasets only leak recall is meaningful.

## Contract for stage 2 (inject)

- **Input and output.** Read `encounters.jsonl` + `labels.jsonl` and write the same three files to
  `data/injected/<name>/`. Never modify `data/converted/`. Keep `encounter_id`, the `dataset` field and the order of
  the records.
- **Where to inject.** Records with `pii_gold: none` are the targets. Injecting into `partial` records mixes synthetic
  and real PII, so only do it on purpose. Skip `exhaustive` records.
- **Inserting text.** Every insertion shifts the `start`/`end` of every existing `pii` and `keep` span after it in
  the same field. Re-run `validate.py` on the output.
- **Labelling.** Give injected spans `label: "injected:<kind>"`. Set `pii_gold` to `exhaustive` only when the source
  text had no PII of its own (symptom/disease sets), and to `partial` otherwise: mtsamples has dates (812 notes) and
  doctors' names (138), codiesp and pmc_patients ages, dates and places, medconv_corpus real names, nbme relatives'
  ages. nbme's `partial` comes from its labelled patient age only, so it is still a good target. Write
  `"stage": "inject"` and the parent dataset in the manifest.
- **Language.** Use `meta.language` to pick names, date formats and phone formats.

## Datasets

| Script | Records | Fields | ICD-10 | `pii_gold` | Use |
|---|---:|---|---|---|---|
| `aci_bench` | 207 | dialogue + note | gold | exhaustive for 107 (harness gold via `--gold-dir`), none for 100 | main data |
| `devp_discharge` | 50 of 500 (`--all`) | note (rendered from JSON, 2 layouts) | gold (5 codes) | exhaustive (every rendered field) | privacy + utility, other note style. Only 10 patients, 5 hospitals |
| `imerit_scribe` | 199 | dialogue + note | none | partial: annotated patient name, physician, DOB | DE/FR/ES/NL/EN robustness |
| `sadiqguru_conversations` | 4 | dialogue | none | partial: names, years, ages (hand-curated) | long English dialogues |
| `mtsamples` | 4,966 | note | none | none | utility, note styles |
| `pmc_patients` | 2,000 sample of 167k | note | none | partial: patient AGE only | quasi-identifiers, attacks |
| `nbme_notes` | 1,000 | note | none | partial: patient AGE (`keep` = the other graded features) | concept recall, age recall |
| `medconv_corpus` | 5,609 (test split) | dialogue | none | none | informal real text |
| `codiesp` | 3,751 | note (Spanish) | none (gold for 1,000 with `--codes`) | none | Spanish robustness |
| `symptom2disease` | 1,200 | dialogue (1 turn) | mapped (24 labels) | none | utility sanity check |
| `medconv2disease` | 960 | dialogue | mapped (24 labels) | none | utility sanity check |
| `pii_external` | 4,434 | note (essay) | none | partial: injected PII, other names and cities unlabelled | generic PII layer |
| `persuade_pii` | 2,218 | note (essay) | none | exhaustive (inserted PII, mostly names) | names in natural text |
| `pii_gemini` | 3,779 | note (essay) | none | partial: upstream lists are unreliable | weak, recall only |
| `pii_gpt` | 2,000 | note (essay) | none | partial: injected PII | generic PII layer |

The tabular Kaggle sets (`generaterealdata`, `sergionefedov`, `nudratabbas`) have no text to convert. They are seeds
for generating synthetic encounters. The code lists, rare diseases, RxNorm and name lists are lookup resources for
the anonymizer, not datasets.

## Caveats

- `mapped` codes are one code per disease label, picked as the ICD-10-CM index default for the unspecified term.
  They are not encounter-level coding.
- iMerit transcripts have no speaker labels. Turns are tagged alternately, starting with the doctor
  (`meta.speaker_tags`). The same scenarios are translated across languages (`meta.scenario`).
- The text is rebuilt from upstream artifacts in three datasets: MTSamples (commas that replaced line breaks),
  NBME (characters dropped by the tokenizer), and the devp discharge notes (rendered from JSON by our templates).
