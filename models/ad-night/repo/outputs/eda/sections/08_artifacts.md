## 8. Synthetic-data artifacts

Script: `scripts/eda/08_artifacts.py`. **[regex]** unless marked.

### 8.1 Boilerplate

- 6,048 distinct note sentences; 91 occur in ≥3 notes. The big ones: `Patient Agreements: The patient understands and agrees with the recommended medical treatment plan.` ×95 (virtassist 43, aci 52), `No wheezes, rales, or rhonchi.` ×32, `Auscultation of Lungs: Clear bilaterally.` ×26, `Respiratory: Denies shortness of breath.` ×24, `Lungs are clear to auscultation bilaterally.` ×24, `All questions were answered.` ×13, `After reviewing the patient's examination and radiographic findings today, I have had a lengthy discussion …` ×10 (aci ortho template), `At this point, I discussed the diagnosis and treatment options with the patient.` ×10 (virtassist).
- Bullet labels: `Medical Reasoning:` 241, `Medical Treatment:` 221, `Patient Education and Counseling:` 143, `Patient Agreements:` 97, `Additional Testing:`; ROS/exam labels `Musculoskeletal:` 127, `Respiratory:` 92, `Cardiovascular:` 90, `Constitutional:` 70, `Neurological:` 58.
- Dialogue: 493 ten-grams shared by ≥3 encounters — the `virtassist` scribe script (`i'd like to tell dax a little bit about you` ×20, `so let's talk a little bit about my assessment and plan` ×19, `the nurse told you about dax` ×14) and closing lines. `dax` preamble in 36 encounters, `hey , dragon` commands in 55 (all `virtassist`).
- Risk: a token classifier trained on 140 encounters will learn "the token after `tell dax a little bit about you . <X> is a`" and "the capitalized bigram before `is a pleasant NN-year-old`" — real recipes for this corpus, useless on a differently formatted test set. Augmentation must break these frames (see Decisions c).

### 8.2 Note ⊄ dialogue (notes are *not* copies) [regex + verified]

- Only 2–4% of note sentences (≥6 words) appear verbatim in the dialogue (`aci` 60/2874, `virtscribe` 51/1208 — dictated exam blocks —, `virtassist` 6/1533). Median share of note 8-grams present in the dialogue: `aci` 0.3%, `virtassist` 1.1%, `virtscribe` 4.8%.
- Consequence: cross-field entity linking cannot rely on string identity. The same fact is re-phrased (`april 10th` → `04/10`, `my mom` → `the patient's mother`, `eight` → `8`, `two thousand eighteen` → `2018`, `nephew` → `cousin`), and the note contains facts absent from the dialogue (`Montelukast … seizure-like reaction` `D2N138`, surname in 102 encounters, computed age in `D2N105`/`D2N099`).

### 8.3 Near-duplicates

- Pairwise Jaccard on 5-gram shingles: **1 pair** above 0.15 (`D2N179`/`D2N180`, note 0.18 — both hand-numbness consults with the same referring-physician frame; different patients `elijah cook` / `shirley taylor`). `D2N181` vs `D2N182` share the name `michelle king` but nothing else (Jaccard 0.00; DOB `01/05/1986` vs `05/07/1982`, codes K21.9 vs K59.09/Z28.310). No duplicated encounters in the corpus.

### 8.4 Templated openings and uniformity [regex + verified]

- 155 distinct 8-token opening templates after masking names/numbers. `virtassist`: `hi , NAME . how are you ?` 16 + variants (31 of 55 start with a greeting + first name); `virtscribe`: `next patient is NAME NAME . date of …` and `NAME NAME , date of birth , N/N/N`; `aci`: `hey NAME good to see you today so` ×6, `hi NAME how are you the medical assistant …` ×4, `so NAME is a N -year-old male who` — 74/207 dialogues open with a greeting + name in the first clause.
- Notes: `<Name> is a NN-year-old <sex>` sentence in 140/207 (virtassist 49/55, virtscribe 33/40, aci 58/112); `pleasant` in 56 HPIs, `presents to the clinic today` in 61.
- Age distribution (n=166 with explicit age): min 3, p10 27, median 48, p90 67, max 100; <18: 6 (`9-month-old`, `3 -year-old`, `10`, `14` ×2, `15`), ≥80: 6, ≥90: 2. Decades: 40s 41, 50s 40, 30s 26, 60s 26, 20s 13. Sex: 83 male / 85 female / 39 not stated (all `aci` notes without HPI). → No "all patients are 59-year-old males" artifact, but ages cluster in 30–65 and a 5-year bucket already gives k ≥ 2 for all but 11 encounters (see §10).
- Vocabulary is *implausibly clean* in notes (12.9% OOV against the system dictionary, almost all legitimate medical terms: `denies`, `musculoskeletal`, `rhonchi`) and *moderately noisy* in dialogue (7.0% OOV, mostly `okay` ×4431, `kinda`, `gotcha`, `hmmm`, `mkay`, plus true ASR garbles, §9).

### 8.5 Train / eval overlap

- 71 note sentences identical across splits (boilerplate: `General Examination: Neck is supple without lymphadenopathy`, `The patient will follow up in 3 months`); 16 identical dialogue lines (`okay have you taken anything for the pain`). 3 full patient names and 24/58 eval first names are also in train (§4.6).
- Codes: eval has 112 distinct codes, **79 never seen in train** (82 of 153 slots); 49/67 eval encounters contain ≥1 unseen code; only 4 of 66 eval code *sets* occur in train. Train has **no** chapter V/W/X/Y code; eval has 24 slots of them. Consequence for the local utility proxy: a supervised ICD classifier fit on train cannot score eval; the proxy must be zero-shot (description similarity / lexical evidence) or measure *evidence retention* (the §7 overlap before vs after anonymization) rather than F1 of a trained model.
