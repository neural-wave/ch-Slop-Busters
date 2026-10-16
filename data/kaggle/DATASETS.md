# External Kaggle datasets

Supplementary data for the Neural Wave anonymization challenge. Our own data (`../patient_doctor_encounters.jsonl` + `../eval_patient_doctor_encounters.jsonl`) is the full ACI-Bench set: 207 dialogue + note encounters with ~2 ICD-10 codes each.

Pulled via the Kaggle MCP on 2026-10-16 and extracted into `data/kaggle/<owner>__<slug>/` (22 datasets, ~2.7 GB). Row counts, code counts and text lengths below were measured on the downloaded files; votes / downloads / usability are Kaggle's numbers on that date.

The data itself is **not in git** (files over GitHub's 100 MB limit; several licenses don't allow re-sharing). To get it, run `scripts/download_kaggle.sh` (needs `pip install kaggle` and a Kaggle API token).

**No dataset on Kaggle pairs English free-text encounters with ICD-10 at scale.** The closest matches are a small synthetic ICD-10 set (#1), Spanish clinical cases whose ICD-10 labels live on Zenodo (#2), and ICD-coded MIMIC, whose Kaggle mirrors break PhysioNet's license (see *Not downloaded*).

Legend: **ICD** ✅ = ICD-10 labels in the data, 🔗 = labels available elsewhere, ❌ = none. **Fit** = usefulness for this project, from ★ (low) to ★★★★★ (high). Kaggle stats are written as votes / downloads / usability.

## A. Clinical text with ICD-10 labels

| # | Dataset | Folder | Contents (measured) | ICD | Lang · Origin | License | Kaggle stats | Fit | Use for / caveats |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [devp1866/noisy-medical-document-images-ocr](https://www.kaggle.com/datasets/devp1866/noisy-medical-document-images-ocr) | `devp1866__noisy-medical-document-images-ocr` | 500 discharge summaries + 500 bills (JPG images plus JSON ground truth in `Data/*_ground_truth.csv`). JSON fields: patient name/DOB/MRN, hospital + NPI, diagnosis, `icd_10`, hospital course, CPT procedures, meds, insurance | ✅ 1 per doc, **only 5 distinct codes** (J20.9, E78.5, E11.9, M17.9, I10) | EN · synthetic (Indian hospitals) | CC BY-SA 4.0 | 39 / 816 / 1.00 | ★★★ | PHI comes as labelled fields, so you get leak-check ground truth for free. Narrative is short (median 25 words) and templated. Ignore the images |
| 2 | [serdarcaglar/medprocner-codiesp-spanish-medical-texts](https://www.kaggle.com/datasets/serdarcaglar/medprocner-codiesp-spanish-medical-texts) | `serdarcaglar__medprocner-codiesp-spanish-medical-texts` | 3,751 `.txt` clinical case reports, median 316 words | 🔗 labels **not** in this upload; join the CodiEsp ICD-10-CM/PCS TSVs from Zenodo by filename | ES · real (published cases) | Unknown here (CodiEsp is CC BY 4.0) | 1 / 15 / 0.13 | ★★★ (★★★★ with labels) | Real clinical writing with dates, ages and places. Good for the robustness bonus. Needs the labels added |

## B. ICD-10 codes in tables, no narrative (seeds for synthetic encounters)

| # | Dataset | Folder | Contents (measured) | ICD | Lang · Origin | License | Kaggle stats | Fit | Use for / caveats |
|---|---|---|---|---|---|---|---|---|---|
| 3 | [generaterealdata/synthetic-hospital-records](https://www.kaggle.com/datasets/generaterealdata/synthetic-hospital-records) | `generaterealdata__synthetic-hospital-records` | 30K patients (first/last name, MRN, DOB, city), 70K encounters, 50K prescriptions, 120K vitals, 29K claims, 600 providers; CSV + SQLite | ✅ 40 distinct codes, each with its full official description | EN · synthetic | CC BY 4.0 | 1 / 7 / 1.00 | ★★★ | Ready-made PHI + ICD-10 profiles for LLM-generated encounters. Codes match their diagnoses. Uploaded 2026-10-14, so barely used |
| 4 | [sergionefedov/patient-records-100k-patients-15-conditions](https://www.kaggle.com/datasets/sergionefedov/patient-records-100k-patients-15-conditions) | `sergionefedov__patient-records-100k-patients-15-conditions` | 100K patients, 274,592 visits (primary + secondary ICD-10), 364K meds, 2.8M labs, 11K admissions with outcomes | ✅ 15 distinct codes (E11.9, I10, J44.1, N18.3, F32.9, …) | EN · synthetic | Apache 2.0 | 42 / 2,056 / 1.00 | ★★★ | Realistic comorbidity mixes and labs to condition generation on. Small code space; no names |
| 5 | [nudratabbas/healthcare-fraud-detection-dataset](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset) | `nudratabbas__healthcare-fraud-detection-dataset` | 10,000 claims: age, gender, state, ICD-10, CPT, specialty, amounts | ✅ 10 distinct codes, no descriptions | EN · synthetic | CC0 | 65 / 2,688 / 1.00 | ★★ | Code frequency priors only |

## Reference: code lists & vocabularies

| # | Dataset | Folder | Contents (measured) | ICD | Lang · Origin | License | Kaggle stats | Fit | Use for / caveats |
|---|---|---|---|---|---|---|---|---|---|
| 6 | [mrhell/icd10cm-codeset-2023](https://www.kaggle.com/datasets/mrhell/icd10cm-codeset-2023) | `mrhell__icd10cm-codeset-2023` | 73,639 ICD-10-CM 2023 codes + descriptions (CSV + official txt) | ✅ full code system | EN · official (CMS) | Other | 12 / 1,038 / 0.94 | ★★★★ | Validate codes, protect diagnosis terms from redaction (Parkinson, Alzheimer, Hodgkin, …), map codes to descriptions |
| 7 | [arashnic/icd10-codes-and-descriptions](https://www.kaggle.com/datasets/arashnic/icd10-codes-and-descriptions) | `arashnic__icd10-codes-and-descriptions` | 71,703 ICD-10 codes (headerless CSV: category, sub, code, long/short desc, category name) + `icd9to10dictionary.txt` with 15,086 ICD-9→ICD-10 mappings | ✅ | EN · official | CC0 | 13 / 1,368 / 0.82 | ★★★ | The ICD-9→10 map is what you need to use MIMIC-III or other ICD-9 data |
| 8 | [ahsanneural/rare-diseases-orphadata-2026](https://www.kaggle.com/datasets/ahsanneural/rare-diseases-orphadata-2026) | `ahsanneural__rare-diseases-orphadata-2026` | 11,456 rare diseases (7,541 with ICD-10) + ICD-11/OMIM/UMLS/MeSH cross-refs, prevalence, genes, age of onset | ✅ cross-refs | EN · Orphanet | CC BY 4.0 | 33 / 847 / 0.94 | ★★★ | Rare diagnoses can identify a patient. Use prevalence to flag high-risk codes and terms during anonymization |
| 9 | [nguyenchuhunganh/medical-code](https://www.kaggle.com/datasets/nguyenchuhunganh/medical-code) | `nguyenchuhunganh__medical-code` | `icd10_final.csv`: 15,845 WHO ICD-10 codes with **Vietnamese** descriptions. `rxnorm_final_dictionary.csv`: 82,429 RxNorm concepts with English synonyms | ✅ (VI text) | VI + EN | Unknown | 0 / 7 / 0.29 | ★★ | Only the RxNorm file is useful: a drug-name allowlist so medications aren't redacted as names |

## C. ⚑ Medical text WITHOUT ICD codes

| # | Dataset | Folder | Contents (measured) | ICD | Lang · Origin | License | Kaggle stats | Fit | Use for / caveats |
|---|---|---|---|---|---|---|---|---|---|
| 10 | [imeritinc/multilingual-ambient-scribe-dataset](https://www.kaggle.com/datasets/imeritinc/multilingual-ambient-scribe-dataset) | `imeritinc__multilingual-ambient-scribe-dataset` | 199 doctor–patient encounters: transcript (median 734 words) + MP3 + structured note in `summaries/*.json` (Reason for Visit, HPI, Assessment & Plan, Meds, PMH, exam, …) + **annotated PHI** (Name, Gender, Physician, Age/DOB). German 42, French 48, UK-EN 42, Spanish 35, Dutch 32 | ❌ (diagnoses in Assessment & Plan; can be coded with an LLM) | DE/FR/EN/ES/NL · scripted/acted | CC BY-NC-SA 4.0 | 1 / 509 / 0.71 | ★★★★ | **Closest to our format** (dialogue + note). Best for the robustness bonus and German. Non-commercial license. Audio is 1.3 GB of the folder |
| 11 | [tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) | `tboyle10__medicaltranscriptions` | MTSamples: 4,999 dictated notes (median 385 words; 33 empty), 40 specialties, keywords | ❌ | EN · real-style samples | CC0 | 265 / 29,930 / 0.85 | ★★★★ | Most-used clinical notes corpus; different note styles. A few fake names and dates. (`harshitstark/healthcare-documentation-database` is a copy) |
| 12 | [priyamchoksi/pmc-patients-dataset-for-clinical-decision-support](https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support) | `priyamchoksi__pmc-patients-dataset-for-clinical-decision-support` | 167,034 patient summaries from PubMed Central case reports (median 400 words) + structured age and gender + PMID | ❌ | EN · real (published) | CC BY-NC-SA 4.0 | 6 / 664 / 1.00 | ★★★ | Large pool of details that can identify someone (age, sex, timelines, rare conditions) for attack and robustness tests. Non-commercial |
| 13 | [cnjameshu/nbme-patient-notes](https://www.kaggle.com/datasets/cnjameshu/nbme-patient-notes) | `cnjameshu__nbme-patient-notes` | 1,000 NBME patient notes, **tokenized** (165K tokens, BIO feature labels; rebuild text by `pn_num`) | ❌ (each note belongs to one of 10 cases with a known diagnosis) | EN · USMLE exam notes | Unknown | 1 / 40 / 0.24 | ★★ | Abbreviation-heavy notes ("17yo M c/o palpitations"). The full 42K raw notes are competition data, which needs a Kaggle sign-in |
| 14 | [sadiqguru/medical-record-sample-data](https://www.kaggle.com/datasets/sadiqguru/medical-record-sample-data) | `sadiqguru__medical-record-sample-data` | 4 long doctor–patient conversations (1,250–1,710 words) + MP3: asthma, BP, cardiology referral, diabetes | ❌ | EN · synthetic | MIT | 0 / 7 / 0.81 | ★★ | Tiny, but very close to ACI-Bench style, with names. Good for hand-made test cases |
| 15 | [thedevastator/medical-conversation-corpus-100k](https://www.kaggle.com/datasets/thedevastator/medical-conversation-corpus-100k) | `thedevastator__medical-conversation-corpus-100k` | 106,556 train + 5,609 test single-turn Q&A (`[|Human|]` patient → `[|AI|]` doctor), median ~187 words. ChatDoctor / HealthCareMagic origin | ❌ | EN · real user posts | CC0 (doubtful: scraped) | 21 / 2,541 / 1.00 | ★★ | Real informal text with names, ages, countries for name-detection tests. Its earlier anonymization was crude (emails masked, names kept, "Chat Doctor" artifacts in 63% of rows) |
| 16 | [niyarrbarman/symptom2disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) | `niyarrbarman__symptom2disease` | 1,200 first-person symptom descriptions (median 30 words), 24 disease labels | ❌ (24 labels, easy to map to ICD-10 by hand) | EN · synthetic | CC0 | 112 / 12,701 / 1.00 | ★★ | Quick sanity check for keeping diagnosis-relevant content. Very simple text |
| 17 | [artemminiailo/medicalconversations2disease](https://www.kaggle.com/datasets/artemminiailo/medicalconversations2disease) | `artemminiailo__medicalconversations2disease` | 960 ChatGPT patient–bot chats (median 94 words), same 24 labels as #16 | ❌ | EN · LLM-generated | Apache 2.0 | 11 / 619 / 1.00 | ★ | Conversational variant of #16. Low realism |

## D. Personal-data (PII) detection data (not medical)

| # | Dataset | Folder | Contents (measured) | Lang · Origin | License | Kaggle stats | Fit | Use for / caveats |
|---|---|---|---|---|---|---|---|---|
| 18 | [alejopaullier/pii-external-dataset](https://www.kaggle.com/datasets/alejopaullier/pii-external-dataset) | `alejopaullier__pii-external-dataset` | 4,434 essays (median 310 words), token-level BIO labels: NAME_STUDENT, STREET_ADDRESS, EMAIL, PHONE_NUM, USERNAME, URL_PERSONAL (+ the injected values per row) | EN · LLM-generated | Apache 2.0 | 92 / 3,526 / 1.00 | ★★★ | Best-labelled PII set; ready for token classification |
| 19 | [thedrcat/persuade-pii-dataset](https://www.kaggle.com/datasets/thedrcat/persuade-pii-dataset) | `thedrcat__persuade-pii-dataset` | 2,218 real student essays with synthetic PII inserted, token labels (mostly names: 3,036; few addresses/phones) | EN · real text + synthetic PII | Other | 27 / 182 / 1.00 | ★★ | Names in natural text |
| 20 | [newtonbaba12345/pii-detection-gemini-created-dataset](https://www.kaggle.com/datasets/newtonbaba12345/pii-detection-gemini-created-dataset) | `newtonbaba12345__pii-detection-gemini-created-dataset` | 3,779 essays (median 356 words) with injected PII listed per type (EMAIL, USERNAME, ID_NUM, PHONE_NUM, URL, STREET_ADDRESS). No token labels | EN · LLM-generated | Apache 2.0 | 46 / 719 / 1.00 | ★★ | Needs string matching to build spans |
| 21 | [pjmathematician/pii-detection-dataset-gpt](https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt) | `pjmathematician__pii-detection-dataset-gpt` | 2,000 essays (median 418 words) + dict of injected PII (name, email, username, ID, phone, URL, address). No header, no token labels | EN · LLM-generated | CC BY 4.0 | 50 / 1,070 / 1.00 | ★★ | Same as #20 |
| 22 | [krist0phersmith/pii-real-names-dataset](https://www.kaggle.com/datasets/krist0phersmith/pii-real-names-dataset) | `krist0phersmith__pii-real-names-dataset` | `given_names_data.parquet` + `surnames_data.parquet` (schema not inspected: no parquet library installed) + `0717-182/nam_dict.txt`: 48,890 first names with gender and per-country frequency (incl. DE/CH/AT), with `gender.c` source (GPL) | multi · real names | Apache 2.0 (nam_dict: GPL) | 6 / 187 / 0.63 | ★★★ | Name lists for spotting names and generating fake replacements. **Removed the bundled `gender.exe` Windows binary** (untrusted, not needed) |

## Not downloaded

| Dataset | Reason |
|---|---|
| [chanhainguyen/mimic-iii](https://www.kaggle.com/datasets/chanhainguyen/mimic-iii) | ⚠️ License. Full MIMIC-III v1.4: NOTEEVENTS (4 GB) + DIAGNOSES_ICD (ICD-9; convert via #7). Best ICD-coded clinical text there is, but it's restricted PhysioNet data re-uploaded as "MIT", which breaks PhysioNet's data use agreement. Get it from PhysioNet with your own credentials (MIMIC-IV + MIMIC-IV-Note has ICD-10) |
| [marim0mahmoud/mimic-iii-csv-noteevents-csv](https://www.kaggle.com/datasets/marim0mahmoud/mimic-iii-csv-noteevents-csv) | ⚠️ License. MIMIC-III notes-only mirror |
| [aminexdr/bhc-mimic-iv-summary](https://www.kaggle.com/datasets/aminexdr/bhc-mimic-iv-summary) | ⚠️ License. MIMIC-IV-Note derived (1.4 GB), diagnoses as free text only |
| [isaacritharson/mimic-iv-cleaned-medical-transcripts](https://www.kaggle.com/datasets/isaacritharson/mimic-iv-cleaned-medical-transcripts) | Misnamed: structured MIMIC-IV rows (admissions, meds, DRG), no notes, no ICD; MIMIC-derived |
| [enriquemas/synthetic-ehr-dataset-10-demo-7-languages](https://www.kaggle.com/datasets/enriquemas/synthetic-ehr-dataset-10-demo-7-languages) | ICD-10 codes don't match diagnoses (e.g. CKD → I10, osteoarthritis → I25) |
| [kevinbnisch/st-gemma-medical-center-dataset](https://www.kaggle.com/datasets/kevinbnisch/st-gemma-medical-center-dataset) | Templated notes with nonsensical treatments (atorvastatin for anxiety) |
| [adwaittagalpallewar/medical-document-ocr-text-dataset](https://www.kaggle.com/datasets/adwaittagalpallewar/medical-document-ocr-text-dataset) | Document bodies are random filler words |
| [infobayai/healthcare-discharge-summary](https://www.kaggle.com/datasets/infobayai/healthcare-discharge-summary) | Only 10 PDFs |
| [harshitstark/healthcare-documentation-database](https://www.kaggle.com/datasets/harshitstark/healthcare-documentation-database) | Duplicate of MTSamples (#11) |
| [yousefsaeedian/ai-medical-chatbot](https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot), [punyaslokaprusty/chatdoctor](https://www.kaggle.com/datasets/punyaslokaprusty/chatdoctor) | Same ChatDoctor/HealthCareMagic source as #15 |
| [algerwang/nbme-score-clinical-patient-notes](https://www.kaggle.com/datasets/algerwang/nbme-score-clinical-patient-notes) | Annotations only, no note text (#13 used instead) |

## Suggested use

- **Extra ICD-10 evaluation now:** #1 (personal details + ICD-10, but only 5 codes). Generate ACI-Bench-style encounters from #3/#4 profiles with a local LLM for wider code coverage.
- **Worth the effort:** add the CodiEsp labels to #2 from Zenodo, and LLM-code the iMerit Assessment & Plan sections (#10).
- **Robustness bonus:** #10 (German/French/Dutch/Spanish), #11, #12, #15.
- **Anonymizer building blocks:** #6/#8 to protect diagnosis terms, #9 (RxNorm) to protect drug names, #22 names, #18–#21 for PII detection.
