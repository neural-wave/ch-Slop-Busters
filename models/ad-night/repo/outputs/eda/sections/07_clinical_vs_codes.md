## 7. Clinical content vs ICD-10 codes

Script: `scripts/eda/07_clinical_vs_codes.py`. Overlap = share of content tokens of the code description (after dropping `unspecified`, `with/without`, `initial encounter`, laterality …) found in the text, with a small synonym map (`hypertension` ↔ `high blood pressure`, `dysphagia` ↔ `swallowing`, `calculus` ↔ `stone`). **[regex, approximate]**; examples **[verified]**.

### 7.1 How the codes surface in the text

| overlap band (note) | aci (218 slots) | virtassist (142) | virtscribe (78) | all (438) |
|---|---|---|---|---|
| verbatim / all content words present | 53% | 64% | 50% | **56%** |
| paraphrased (0.5–0.99) | 33% | 26% | 29% | 30% |
| weak (0.25–0.5) | 7% | 5% | 10% | 7% |
| symptom / lab only (<0.25) | 6% | 3% | 10% | 6% |

- Dialogue bands are nearly identical (55 / 31 / 6 / 6%): 378 of 438 code slots have the **same** overlap in both fields, 34 are better supported by the note, 26 by the dialogue. The key phrase of the description appears *verbatim* in only 107 notes / 93 dialogues: most codes are supported by paraphrase (`high blood pressure`, `difficulty swallowing`, `sugar`).
- Best note section for the evidence: HPI 162, CHIEF COMPLAINT 82, ASSESSMENT AND PLAN 44, MEDICAL HISTORY 43, ASSESSMENT 32, RESULTS 18, PHYSICAL EXAM 16. The ASSESSMENT/PLAN sections are the primary evidence for only ~17%; the HPI (where the name, age and social details also live) is the primary evidence for 37%. **Anonymizing the HPI aggressively costs utility; anonymizing SOCIAL HISTORY costs little except for external-cause codes.**
- Codes with weak lexical evidence in *both* fields (27 slots), i.e. inferable only from numbers, findings or clinical reasoning: `I49.3` ventricular premature depolarization (text says `pvcs`, `D2N105`), `J44.9` COPD (`D2N170`), `L21.0` seborrhea capitis (`cradle cap`, `dandruff`, `D2N099`), `E29.1` testicular hypofunction (`hypogonadism`, `testosterone`, `D2N023`), `J02.9` pharyngitis (`sore throat`, `D2N041`), `C56.2` ovarian cancer (`D2N046`: `newly diagnosed ovarian cancer` is in CHIEF COMPLAINT, laterality `left` only in the body), `Q93.88` microdeletions (`smith magenis syndrome`, `D2N076`), `M51.27` disc displacement (`D2N112`), `R87.619` abnormal cervical cytology (`abnormal pap`, `D2N030`), `Z09` follow-up after treatment ×3, `Z79.899` long-term drug therapy. These depend on clinical synonyms, not on PII.
- **External-cause / activity codes (24 slots, all in eval)**: the activity codes are fully lexical (`Y93.53 golf`, `Y93.54 bowling`, `Y93.02 running`, `Y93.66 soccer`, `Y93.23 skiing`), the mechanism codes are paraphrased (`W00.0XXA fall … ice and snow` ← `it was snowing … i fell`, `W21.02XA struck by soccer ball`, `V89.2XXA` ← `car accident`, `W10.1XXA sidewalk curb` ← `struck her right knee on a curb`), and the place codes (`Y92.480 sidewalk`, `Y92.009 private residence`) require `on the sidewalk`, `at home`, `in my yard`. The hobby/event/place vocabulary that phase 3 lists as quasi-identifiers *is the evidence* for ~5% of all code slots and 15% of eval slots.

### 7.2 Sensitive-but-code-relevant content [regex + verified]

| topic | encounters with mention (aci / virtassist / virtscribe) | coded in `icd10_codes` | typical surface forms | examples |
|---|---|---|---|---|
| mental health | 127 (68 / 37 / 22) | 20 (F32.A ×17, G47.00, G31.84, …) | `depression` 141, `anxiety` 49, `psychiatric`, `prozac`, `therapist`, `mood`, `counseling` (boilerplate `Patient Education and Counseling` inflates the count) | `Denies thoughts of self-harm or suicidal ideation` (`D2N172`), `Regarding her depression …` |
| substance use | 68 (42 / 7 / 19) | 2 (`F17.200`, `F10.90`) | `smoking` 73, `alcohol` 69, `drink(ing)` 99, `tobacco`, `vape`, `iv drug use`, `rehab` (mostly *cardiac* rehab) | `significant alcohol use disorder` (`D2N142`), `i used to party a lot and even did use iv drugs` (`D2N038`), `smokes 1 pack per day` |
| HIV / STI / hepatitis | 7 (4 / 0 / 3) | 2 (`B19.20`, `B17.1`) | `hepatitis c` 23, `hep c`, `hcv`, `std`, `hiv` 5, `gonorrhea`, `prep` | `59 year old hiv positive gentleman here for hypogonadism` (`D2N023`, HIV not coded), `any hep c partners` (`D2N038`) |
| sexual / reproductive | 44 (24 / 5 / 15) | 4 (`N62`, `N60.0x`, `D05.1`, `N50.89`) | `period(s)` 47, `testosterone` 20, `vaginal`, `partner`, `prostate`, `menstrual`, `birth control` | `headaches … around my menstrual periods` (`D2N114`, relevant to R51.9 history), `breast reduction` (`D2N024`, `N62`) |
| legal / social | 7 | 0 | `court`, `lawyer`, `abuse`, `disability` | `the lawyer said i needed an mri` after `multiple car crashes` (`D2N127`: litigation context) |
| genetic / family disease | 36 (19 / 1 / 16) | 1 (`Q93.88`) | `family history` 77, `genetic`, `smith-magenis` 7, `autism` 3 | FAMILY HISTORY sections (`Mother: melanoma. Maternal aunt: melanoma.`); `twin sister … autism` (`D2N076`) |
| pregnancy | 0 | 0 | – | no obstetric encounter |

- Policy need: mental-health and substance words are both **sensitive** and **code-bearing** (F32.A is the 3rd most frequent code; F17.200/F10.90 are codes). They cannot be removed without losing codes; they are not identifying on their own (depression in 127 encounters, k≈60). HIV (`D2N023`) and hepatitis C with IV drug history (`D2N113`, `D2N038`) are rare (k ≤ 7) and therefore fingerprints, yet `B19.20`/`B17.1` are the codes. Recommendation (see Decisions): keep the diagnosis term, remove the narrative detail around it (`iv drug use over 15 years ago`, `hep c partners`, `used to party a lot`), and never add or remove a diagnosis word.
- `[patient_guest]` encounters (5, all train) carry third-party health information (sister's medicine, nephew's sinus infection, twin's syndrome) — relatives' diagnoses are sensitive and non-code-bearing: generalize.
