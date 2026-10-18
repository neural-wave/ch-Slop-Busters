# Dataset analysis for the anonymization pipeline (Task 1)

Corpus: 207 patient–doctor encounters (train 140, eval 67; `aci` 112 / `virtassist` 55 / `virtscribe` 40). Scripts: `scripts/eda/` (stdlib only, `uv run python scripts/eda/<nn>_<phase>.py`). Machine-readable artifacts and `hard_cases.jsonl` are in `outputs/eda/`. Tags: **[regex]** = automated count (approximate), **[verified]** = read by eye. 12 encounters were read end to end (§0).

## Executive summary — the 10 biggest risks for the anonymizer

1. **Dialogue has no casing and, in `aci`, no punctuation and no digits**: 207/207 dialogues are lowercase (1 capitalized name token in ~900), 112/112 `aci` dialogues have no sentence punctuation and spell numbers out (26 digit tokens in 112 dialogues: `fifty nine`, `one twenty over fifty`, `seven . four`, `twenty eighteen`). Cased NER and `\d`-based rules are blind on 54% of the dialogues.
2. **The patient's name is a bare vocative with no anchor in ~8% of dialogues** (`alright scott so`, `okay raymond`, `howard` as a one-word turn) and 38 of the 209 name tokens are also ordinary or medical words (`white`, `young`, `cook`, `price`, `mark`, `walker`, `allen`, `murphy`). Punctuation-anchored patterns reach 90% encounter recall; a first-name gazetteer alone has ~70% token precision (`ray` 185×, `may` 97×, `hope`, `summer`).
3. **The two fields are asymmetric**: 102 encounters carry the surname only in the note, 66 carry the first name only in the dialogue, 45 notes (39 `aci`) have no name at all, and the note contains facts the dialogue never states (`100-year-old` and `9-month-old` computed from DOB + date of service; `Montelukast … reaction`). Entities must be linked per `encounter_id`, not by string identity (only 2–4% of note sentences are copied from the dialogue).
4. **`virtscribe` opens with a dictated header** in 36/40 encounters that packs full name + DOB (33 encounters, 6 formats) + MRN (11 encounters, 5 formats: `mrnr472348`, `mrn 3-8-4-9-2-0`, `mrn : abc123`, `0000123`, `113322 . 44-year-old`) + date of service, and the absolute dates continue in the note (`ECG, 12/26/2020.` inside an all-caps line). 82% of all absolute dates are in `virtscribe`.
5. **Speaker tags are not a reliable role cue**: ≥6 encounters (3%, 4 in eval) have `[doctor]`/`[patient]` swapped for the whole dialogue, 14 lines have no tag (including a first line), `[patient_guest]` exists only in 5 train encounters. "The name after `hi` in a doctor line is the patient" fails 3% of the time.
6. **~10% of encounters name a third party**: 48 name a clinician (`dr. reyes`, `doctor isabella`, `dr. jacob ward`), staff first names appear with no title (`helen`, `madison`), 7 have named relatives/friends (`my friend diane`, `my son sean`), 1 a nickname (`dino`), 1 a celebrity simile (`dawn knots`). The doctor's own biography leaks too (`that was my first fall here`).
7. **Quasi-identifiers are code evidence**: hobbies/activities/places appear in 117 encounters and are the evidence for the 24 external-cause/activity code slots (`Y93.53 golf`, `W00.0XXA ice and snow`, `V89.2XXA motor-vehicle`), all of which are **eval-only**. Removing `skiing` or `car accident` deletes ~5% of code slots; `icd10_codes` alone already makes 80% of encounters unique (97% with decade + sex), so k-anonymity is not a meaningful target — text-level leakage is.
8. **Train/eval label shift**: 79/112 eval codes never occur in train (82/153 slots, 49/67 encounters), 108/146 train codes are singletons, chapter V/W/X/Y exists only in eval. A supervised ICD classifier is not a valid utility proxy; evidence retention or zero-shot description matching is.
9. **5,380 numeric tokens, 41% spoken**: only ~250 are privacy numbers (11 MRNs, ~200 dates, 344 exact ages) against >2,000 must-keep clinical numbers; look-alikes go both ways (`250/50` dose, `18-20/28` score, `4/6 murmur` vs `4/28/07` DOB, `113322 . 44-year-old`, `you are 100`).
10. **Noise that creates false negatives**: 350 ASR-garble tokens (`hey document`, `positive long ... lyme`, `mrnr472348`), a name typo (`jaqueline`/`Jacqueline`), 35 narrow no-break spaces and other non-ASCII characters inside notes, stutters next to names (`n- date of birth`), split dots in titles and decimals, 8 false all-caps headers. Normalization must be 1:1 and offset-mapped.

## Contents

- Executive summary — the 10 biggest risks for the anonymizer
- 0. Encounters read end-to-end by eye (12: 4 per dataset) [verified]
- 1. Structure and format
- 2. Normalization quirks of `dialogue` (vs `note`)
- 3. PII inventory
- 4. Names deep-dive
- 5. Ambiguous and adversarial tokens
- 6. Temporal information
- 7. Clinical content vs ICD-10 codes
- 8. Synthetic-data artifacts
- 9. Typos and noise
- 10. Re-identification risk
- 11. Train vs eval shift
- Trap catalogue
- Decisions
- Open questions (for the organizers or the team)

## 0. Encounters read end-to-end by eye (12: 4 per dataset) [verified]

### D2N083 — aci, train — M77.01 medial epicondylitis right elbow
- Patient first name only in dialogue line 1 (`hey lawrence how're you doing`); last name **only** in note (`Lawrence Butler is a pleasant 45-year-old male`). Age **only** in note. Nothing else identifying in dialogue except hobby.
- Quasi-identifiers: `during this pandemic i really got into ceramics and doing pottery` → note SOCIAL HISTORY `began making pottery during the COVID-19 pandemic`. Hobby is also clinically relevant (overuse → epicondylitis): removing it loses the causal story but not the code.
- Dialogue has zero punctuation; dosing abbreviation `q.6 h.` (missed by the abbreviation regex). Clinical slip: `resistance against flexion of that left wrist` while everything else is right elbow — laterality noise that a "fix laterality" heuristic could not resolve.
- Note contains `6/10 pain` (slash number ≠ BP) and first-person `The patient and I discussed`.

### D2N088 — virtassist, train — J06.9, F32.A, E11.9, I10
- `andrew` in dialogue lines 1, 7 (`andrew is a 59-year-old male …`), 7 again; note `Andrew Campbell` twice (HPI first sentence and the identical sentence re-pasted at the top of ASSESSMENT AND PLAN).
- Two product names used as if they were people: `i know the nurse told you about dax . i'd like to tell dax a little bit about you` and 6× voice commands `hey , dragon , show me the vital signs / chest x-ray / diabetic labs`, `hey , dragon . order lisinopril`, `hey , dragon . finalize the note`. These are not in the note.
- Relatives/hobbies/events: `my wife got me into barre classes`, `some house parties`, `we were doing some landscaping`, `saturday or sunday` — all copied into the note HPI (`His wife got him into barre classes`, `except for some house parties`).
- Numbers spelled in dialogue (`elevated at eight`, `around six or seven`), digits in note (`elevated at 8`). E11.9 "without complications" is assigned although A1c = 8: the organizers' codes are not threshold-driven here.
- Boilerplate: `• Medical Reasoning:` / `• Additional Testing:` / `• Medical Treatment:` per problem, `Follow up: I would like to see him back in approximately 4 months.`, `Patient Agreements: The patient understands and agrees with the recommended medical treatment plan.`

### D2N029 — virtscribe, eval — C43.59 melanoma of trunk
- Dictated header: `next patient is sophia jackson , mrnr472348 . she's a 57 year old female` — the MRN is glued to `mrn` (ASR), no hyphen in the age. Later `mrs. jackson`. Note: `Sophia Jackson`, `Mrs. Jackson`.
- Date format changes across fields: dialogue `april 10th` → note `on 04/10` (no year in either). `last month` kept.
- Relatives without names: `my mom`, `her sister`, `my grandfather`, `your wife` (female patient with a wife). Family history section lists them (`Mother: melanoma. Maternal aunt: melanoma. Maternal grandfather: pancreatic cancer.`) — code-irrelevant but medically useful; not identifying on its own.
- Occupation: `i own an auto repair shop` → note `Patient owns an auto repair shop.` Smoking `since my teens`.
- Dictated pathology block is dense with must-keep numbers: `clark level 4`, `breslow thickness 0.7 millimeters`, `stage 1`, `one to two centimeter margin`. Contains an ASR error that survived into the note: `infiltrating , uh , lymphocytes , breast` → `Infiltrating lymphocytes: breast.` (should be *brisk*); `central a melanotic` → `amelanotic` (fixed in note).
- Organization name `national comprehensive cancer network guidelines` — a cased NER will tag it ORG; it is not PII.
- Note uses curly apostrophes (`patient’s`, `There’s`).

### D2N123 — aci, train — R13.10, E11.9, I10
- `tyler` ×3 in dialogue (line 1 `so tyler is a 56 -year-old male`, then vocatives); note `Tyler Green` — last name only in note. Note starts with `SUBJECTIVE` instead of CHIEF COMPLAINT.
- Clinical numbers spoken in words with a split decimal: `hemoglobin a1c last time was seven . four` (note `7.4`), `one fifty i'm sorry one twenty over fifty` (note `120/50`), `five hundred milligrams`, `eighty percent of the time`. ASR noise at the end: `thanks for coming in to them`.
- No quasi-identifier at all beyond name/age/sex: a "minimal" encounter.

### D2N131 — virtassist, train — I10, Z94.0
- First line has **no speaker tag** (`hi , susan , how are you ?`), two more untagged doctor continuations later. `susan` ×4 in dialogue; note `Ms. Susan Watson` ×2 (title + full name). Age 26 in both.
- `stress at work` (no employer), `kidney transplant a few years ago`, `scar on the left flank` — transplant status is the code (Z94.0) and a strong quasi-identifier at once. 6 `hey , dragon` commands.

### D2N105 — virtscribe, train — R01.1, I49.3, I10, E78.5, E03.9, R42
- Dictated header carries DOB **and** date of service: `patient's name is diana scott . date of birth , 12/8/1920 . the date of service is 7/9/2021`. The note has neither date but states `Diana Scott is a 100-year-old female` — the age was **computed from DOB and DOS**. Age 100 is itself identifying (HIPAA: ages > 89 must be aggregated). Dialogue also says `well , so you are 100`.
- First name reused inside the dictated assessment (`diana has a 4/6 systolic murmur`) and in the note PLAN (`Diana has a 4/6 systolic murmur`) — names are not confined to the opening.
- **Staff first name in lowercase mid-dialogue**: `the medicines that you went through with madison` (medical assistant). Nothing marks it as a name.
- Family history with relations only (`my son`, `my brother`, `my eldest sister`); `walker`, `the culdesac` → note `cul-de-sac`, `the driveway`; weights `120 pounds` / `130 pounds`.
- Dictation-system commands inside the dialogue: `for physical exam , please use the regular template`, `end of dictation`.
- Clinical ASR/dictation errors that reach the codes: dialogue says `hypolipidemia` and `your hyperthyroidism , uh , i see you're taking levothyroxine`; note says `Hyperlipidemia` and `Hypothyroid`; code is E03.9 hypothyroidism. Functional decline `70 to 80 %` in dialogue becomes `20-30%` in note.

### D2N114 — aci, train — R51.9
- **Speaker tags swapped for the entire encounter**: `[patient] good morning Julie how are you doing today` is the doctor, `[doctor] i'm doing okay i'm just a little nervous about … my headaches` is the patient. `Julie` is the only capitalized name in all dialogues and it is the *patient* (`Julie Jenkins is a pleasant 33-year-old female`). Age only in note.
- Referring physician: `as doctor white said` → note `Dr. White` ×2. Brands as quasi-identifiers: `new balance`, `asics`, `nike`, `tumor guy` (ASR garble of a shoe brand) → note `wears either New Balance or Asics shoes`. `insurance company` generic.
- Note typo `CV: Normal head rhythm`; Roman numerals `Cranial nerves I through XII`.

### D2N092 — virtassist, train — S63.501A right wrist sprain
- Patient known only by surname with title: `mrs . martinez` ×2 (split dot) and note `Ms. Martinez` — **no first name anywhere**. Doctor named by the patient: `hey , dr . gomez`. Dragon command includes the name: `hey , dragon , i'm here seeing mrs . martinez . she's a 43-year-old female`.
- Note uses short headers `CC:` / `HPI:` / `EXAM` / `IMPRESSION`. **Laterality contradiction**: dialogue + HPI + IMPRESSION + code = right; EXAM and RESULTS say `left wrist`.
- Quasi-identifiers: `playing tennis`, `fell doing lawn work … on my rake` (told as "surgical history").

### D2N099 — virtscribe, train — B34.9, L21.0 (infant)
- `[patient_guest]` is the mother; the patient is `randy gutierrez . date of birth , 8/10/2020` → note `9-month-old male` (age again derived from DOB; DOB absent in note). `randy` ×4 in dialogue (`how many days has randy been feeling sick`, `big breath randy`, `bye randy`).
- **Nickname / pet name**: mother says `dino , you're doing so good`, also `buddy`, `little one`, `where's mama`. A name list will not contain `dino`.
- Relatives: `his older sister`, `my nephew` (note turns it into `a cousin`), mother. Dictation commands: `please use review of symptoms . all text to write . physical exam , auto text uri .`
- Brands galore, lowercase: `tylenol`, `baby vick's`, `aveno` (→ `Aveeno`), `zarbee's`, `head & shoulder's`, `selsun blue`, `nizoral ad`, `honey nut cheerios`, `agave`. Temporal: `supper time last night`, `this morning`, `if this was a monday`, `the weekend`.

### D2N058 — aci, eval — I25.10, I34.0, I50.1, I10
- **Full name in dialogue line 1** (`russell ramirez is a 45 -year-old male …`), `russell` ×4 later; the note contains **no name at all** (`Patient reports …`). Age only in dialogue.
- ASR artifacts: `hey document` (= hey doc) ×3, `dawn knots legs` (= *Don Knotts* legs, a celebrity simile → a public-figure name), `the rex center` (rec center, a place), `i do n't want my sense to close up` (stents). Years spelled: `twenty eighteen`, `two thousand eighteen` → note `2018`. Dose with spoken decimal: `aldactone twelve . five milligrams` → `12.5 mg`.
- Employer hint: `pretty good insurance there through the plant`; hobbies `walking the trails`, `gardening`, `photography` → note SOCIAL HISTORY `enjoying walking outside, gardening, and nature photography`.
- Must-keep numbers: `ejection fraction thirty five percent` (drives I50.1), `three out of six six systolic ejection murmur`, `aspirin eighty one milligrams`.

### D2N134 — virtassist, train — A69.20 Lyme, I10, E11.9
- `michelle` ×3 in dialogue; note **never names her** (`The patient is a 31-year-old female`). First line starts with a cut-off fragment `we're gon na go right to the front- . so when you touch that- hi , michelle`.
- Eponym that must survive: `lyme titer`, `lyme disease` (code A69.20); ASR variant `positive long ... lyme titer`. `hiking in the woods a couple of weeks ago`, `bought a cuff last year`. One untagged continuation line (`so for your second problem`).

### D2N138 — virtscribe, train — J45.40, J30.9, T78.01XA
- Header: `next patient is emily hernandez . date of birth , march 26 2001 . this is a 20 year old female`; `written consent is obtained today to use dax for documentation`. Note `Emily Hernandez` ×2, `Mrs. Hernandez` ×2.
- **Staff**: `i'll have helen come in`, `did you update it with helen`. **Employer**: `you worked with enterprise rent-a-car` → note `cleaning an Enterprise rental car`. **Insurer + pharmacy with a street**: `kaiser` ×3, `kaiser permanente broadway`. **Military**: `going into the military`, `enlisted early spring of 2022` → note `join the military` / `joining the army`.
- Geographic leak without a place name: `there were so many fires … all you saw was the orange and the smoke`, `smoke season` (West-coast wildfire year 2020) and the doctor's own biography `that was my first fall here`.
- Numbers that look like PII but are not: `250/50 dose` (Wixela strength, looks like BP), `act score is 21`, `80 mcg`, `follow-up 25 minutes` (billing). `[ inaudible 00:09:25 ]` timestamp. Note typo `Welixa`; note ALLERGIES lists `Montelukast sodium, “seizure-like” reaction`, which is not in the dialogue (note ⊄ dialogue).

### Things caught only by reading (regex missed)
1. Swapped speaker tags (`D2N114`); 2. staff first names with no title (`madison`, `helen`); 3. nicknames (`dino`); 4. celebrity simile (`dawn knots`); 5. employer named as `the plant`, `enterprise rent-a-car`; 6. insurer/pharmacy with street (`kaiser permanente broadway`); 7. age derived from DOB + DOS (`100-year-old`, `9-month-old`); 8. wildfire/smoke-season as geography; 9. doctor's own biography (`my first fall here`); 10. the dialogue `[doctor]` header line that packs name + DOB + MRN + DOS in one sentence; 11. clinical contradictions (laterality, hypo/hyper) that no anonymizer should "fix" but that confuse a model which uses the note to disambiguate the dialogue.

## 1. Structure and format

Script: `scripts/eda/01_structure.py` → `outputs/eda/headers_by_dataset.json`. Tags: **[regex]** = automated count, **[verified]** = read by eye.

### 1.1 Schema and sizes [regex]

- All 207 records (train 140 / eval 67) share the same 5 keys with the same types; `icd10_codes` entries always `{code, description}`; no duplicate `encounter_id`, no empty field, id pattern always `D2N###`.
- Split × dataset: train `aci` 77 / `virtassist` 35 / `virtscribe` 28; eval `aci` 35 / `virtassist` 20 / `virtscribe` 12. `aci` is 54% of both splits.
- Lengths (median chars, min–max): dialogue train 6196 (3115–13047), eval 6247 (2938–14186); note train 2648 (867–5347), eval 2537 (852–5712). `virtscribe` is the long tail: eval `virtscribe` dialogue median 8793 chars vs 5248 for `virtassist` (two eval virtscribe encounters > 13k chars). Median 50 dialogue turns, 52 note lines.
- ICD-10: train 285 code slots / 146 distinct, eval 153 / 112 distinct. 64/140 train encounters have exactly 1 code (`aci` median 1, `virtassist` median 3). Code shapes range from `A99` to `A99.99AA` (7-char codes with laterality/encounter suffixes, e.g. `S52.501A`, `M17.11`-style). Same code always has the same description; no code repeated inside an encounter.

### 1.2 Speaker tags [regex, verified]

| tag | lines | encounters |
|---|---|---|
| `[doctor]` | 5897 | 207 |
| `[patient]` | 5406 | 205 |
| `[patient_guest]` | 132 | 5 (all `virtscribe`, **train only**) |
| `[ inaudible 00:09:25 ]` | 1 | `D2N138` (mid-line transcription artifact, spaces inside brackets) |

- 26 lines carry **no speaker tag**: 14 are empty lines (13 encounters, e.g. `D2N129` has two) and 12 are real text continuing the previous `[doctor]` turn (`D2N095` l.25 `and then in terms of your blood pressure , how are you doing ?`, `D2N131` l.0 `hi , susan , how are you ?` — the *first* line has no tag; `D2N019` l.25–26 two consecutive untagged doctor lines). All 12 are `virtassist`. A parser must treat a tagless line as "continuation of previous speaker" (or doctor if first) rather than crash or drop it.
- 228 consecutive lines with the same speaker (turn merging is not guaranteed). 2 encounters have no `[patient]` line at all (`D2N058`-style monologue dictations).
- Only 3 tokens in all 207 dialogues contain an uppercase letter: `PanOxyl` ×2 (`D2N114`, brand name) and `Julie` ×1 (`D2N114`, `[patient] good morning Julie how are you doing today`). Reading `D2N114` end to end shows the **speaker tags are swapped for the whole encounter** (the `[doctor]` lines describe the headaches, the `[patient]` lines perform the exam), so `Julie` is the *patient* (`Julie Jenkins` in the note). Casing is not a usable feature in dialogue, and speaker tags are not a reliable cue for "who is the patient".
- Other words that look like names but are products **[verified]**: `dax` (virtassist scribe preamble, ~70×), `dragon` (`D2N092`: `hey , dragon , i'm here with …` — Nuance Dragon dictation product).

### 1.3 Punctuation / tokenization differs by dataset [verified]

- `virtassist` and `virtscribe` dialogues have **tokenized punctuation** (`hi , andrew . how are you ?`, `mr . james`, `mrs. anderson` also occurs un-split in `D2N096`).
- `aci` dialogues have **no sentence punctuation at all** in most encounters (`[doctor] hey lawrence how're you doing`, `D2N083`; `[doctor] so tyler is a 56 -year-old male …`, `D2N123`) and write ages as `56 -year-old` (space before hyphen). Sentence segmentation is impossible from punctuation in `aci`; a model must rely on the turn boundary.
- `virtscribe` dialogues frequently start with a **dictated header** read by the doctor: `next patient is emily hernandez . date of birth , march 26 2001 . this is a 20 year old female …` (`D2N138`), `next patient is sophia jackson , mrnr472348 . she's a 57 year old female …` (`D2N029`), `patient is pamela cook . medical record number is 123546 .` (`D2N024`), `patient's name is diana scott . date of birth , 12/8/1920 . the date of service is 7/9/2021 .` (`D2N105`), `maria martin , date of birth 03/05/1946 , has been informed and consents to have this visit recorded` (`D2N178`). This single first line can contain full name + DOB + MRN + date of service.

### 1.4 Note section headers [regex + verified]

Header detection = line that is entirely upper-case (≥3 letters). Per dataset (percentage of notes containing the header):

| header | aci (112) | virtassist (55) | virtscribe (40) |
|---|---|---|---|
| CHIEF COMPLAINT | 88% | 78% | 100% |
| HISTORY OF PRESENT ILLNESS | 52% | 82% (+18% `HPI`) | 100% |
| REVIEW OF SYSTEMS | 95% | 78% | 15% (+5% `REVIEW OF SYMPTOMS`) |
| PHYSICAL EXAM / PHYSICAL EXAMINATION / EXAM | 99% / – / – | – / 82% / 18% | 85% / – / – |
| VITALS / VITALS REVIEWED | 65% / – | – / 15% | 10% / – |
| RESULTS | 76% | 95% | 52% |
| ASSESSMENT + PLAN (separate) | 52% | 11% / 29% | 98% |
| ASSESSMENT AND PLAN | 48% | 69% | 2% |
| INSTRUCTIONS | 63% | 4% | 85% |
| SOCIAL HISTORY | 56% | 0% | 48% |
| MEDICAL HISTORY / PAST MEDICAL HISTORY / PAST HISTORY | 51% / – / – | – / 9% / – | 2% / 2% / 60% |
| MEDICATIONS / CURRENT MEDICATIONS | 47% / – | – / 9% | 2% / 60% |
| FAMILY HISTORY | 15% | 0% | 35% |
| SURGICAL HISTORY / PAST SURGICAL HISTORY | 14% / – | – / 7% | – |
| ALLERGIES | 4% | 0% | 25% |
| IMPRESSION | – | 16% | – |

- 80 distinct header sequences in `aci`, 18 in `virtassist`, 35 in `virtscribe`: section order is **not** fixed. Most common `aci` sequence appears only 7×.
- Spelling/synonym variants to normalize: `PHYSICAL EXAM`/`PHYSICAL EXAMINATION`/`EXAM`; `REVIEW OF SYSTEMS`/`REVIEW OF SYMPTOMS`; `VITALS`/`VITALS REVIEWED`; `MEDICAL HISTORY`/`PAST MEDICAL HISTORY`/`PAST HISTORY`; `MEDICATIONS`/`CURRENT MEDICATIONS`; `HISTORY OF PRESENT ILLNESS`/`HPI`; `CHIEF COMPLAINT`/`CC`.
- 10 `virtassist` notes use short-form headers with a colon: `CC:` / `HPI:` (`D2N092`, `D2N094`, `D2N096`, `D2N132`, `D2N069`). Two-letter headers are missed by a "≥3 capitals" rule.
- **False headers** (all-caps content lines that a naive header detector swallows) **[verified]**: `ECG, 12/26/2020.` (`D2N103`), `CMP, 03/22/2021.` (`D2N185`), `BSA 2.65` (`D2N101`), `HIV` (`D2N023`), `PCOS.` (`D2N032`), `GERD.` (`D2N089`), `COPD.` (`D2N015`), `MSK`, `OTC`, `EKG`. Two of them contain an **absolute date** inside an all-caps line, so date detection must also run on "header" lines.
- Inline sub-headers (`NEURO: …` 19×, `HENT: …` 13×, `RESPIRATORY: …`, `SKIN:`, `NECK:`, `GAIT:`) and Title-case sub-headers (`Cardiovascular`, `Respiratory`, `Musculoskeletal`, `Neck`, `Gastrointestinal` as standalone lines, 50–56× each) live inside `REVIEW OF SYSTEMS` / `PHYSICAL EXAM`. They are not PII but a truecaser/NER must not read `Neck` or `Head and Face` as a name.
- 8 empty sections (e.g. `D2N203 RESULTS`, `D2N015 ASSESSMENT AND PLAN`). 30 header lines end with `:`.

### 1.5 Bullets and lists [regex]

- `•` (U+2022): 789 lines in 83 notes, only `virtassist` (655) and `virtscribe` (134). `-` bullets: 604 lines in 56 notes, 54 of them `aci`. Numbered items `1.`/`2.`/`3.` in 61/112 `aci` notes (ASSESSMENT AND PLAN problem list: `1. Hypertension`), rare elsewhere.
- Bullets appear in `ASSESSMENT AND PLAN` (91 notes), `PHYSICAL EXAM(INATION)` (99), `REVIEW OF SYSTEMS` (48), `ASSESSMENT` (46). Every note contains blank lines; none is indented.
- Impact: a line-based detector must strip the bullet before matching; `•` and `-` are also where `Medical Reasoning:` / `Patient Education and Counseling:` boilerplate lives (see §8).

### 1.6 Non-ASCII and control characters [regex, all occurrences listed]

| char | name | n | encounters | field / dataset | example |
|---|---|---|---|---|---|
| U+2022 | BULLET | 789 | 83 | note; virtassist 655, virtscribe 134 | `• Constitutional: Denies fever.` |
| U+2019 | RIGHT SINGLE QUOTE | 39 | 25 | note; virtscribe 29, aci 10 | `Neer’s test` (`D2N098`), `Zarbee’s` (`D2N099`) |
| U+202F | NARROW NO-BREAK SPACE | 35 | 8 | note; virtscribe 33, virtassist 2 | `Inspection reveals asymmetrical breasts` (`D2N101`, used *as a word separator*) |
| U+201C/U+201D | CURLY DOUBLE QUOTES | 21+23 | 17–19 | note; virtscribe mostly | `described “like he is fighting it”.` (`D2N099`) |
| U+00A0 | NO-BREAK SPACE | 4 | 4 | note; virtassist | `chills.  Endorses fatigue.` (`D2N093`) |
| U+200B | ZERO WIDTH SPACE | 1 | 1 | note `D2N174` | `lately. ​His recent knee x-ray` |
| U+00B2 | SUPERSCRIPT TWO | 1 | 1 | note `D2N185` | `1.73 m²` |
| U+2013 | EN DASH | 1 | 1 | **dialogue** `D2N200` | `start a – diet` |
| U+00DF | SHARP S | 1 | 1 | note `D2N010` | `right lower extremity.ß` (stray keystroke) |

- No `\r`, no `\t`, no double spaces, no combining characters (all text is NFC). 7 note lines have trailing whitespace.
- Why it matters: (a) ` `/` `/`​` are not matched by `\s` in some tokenizers and `\b` word boundaries behave differently around them → a regex like `\bMs\. \w+` silently misses; (b) `’` inside `Neer’s`/`Zarbee’s` breaks a `'s` possessive rule written for ASCII; (c) if a pre-processing step replaces these characters, the output must be mapped back so span offsets refer to the original text (1-to-1 replacements ` →' '`, `’→'`, `“”→"` keep offsets identical; deleting `​`/`ß` shifts them by 1).

## 2. Normalization quirks of `dialogue` (vs `note`)

Script: `scripts/eda/02_dialogue_normalization.py`. All counts **[regex]** unless marked; examples checked by eye **[verified]**.

### 2.1 Three different "dialogue dialects" [verified]

| property | aci (112) | virtassist (55) | virtscribe (40) |
|---|---|---|---|
| sentence punctuation | **absent** (`hey lawrence how're you doing`) | tokenized (`hi , andrew . how are you ?`) | tokenized |
| digits in dialogue | **26 total** (numbers spelled out: `ninety nine percent`, `two thousand eighteen` `D2N058`) | 274 | 564 |
| `um`/`uh` fillers | 128 | 1267 | 651 |
| repeated words `the the`, `i i` | **1781** | 25 | 6 |
| stutters `w- we` / trailing `that-` | 0 / 2 | 78 / 153 | 36 / 105 |
| ellipsis `...` | 0 | 185 | 105 |
| split contractions `do n't`, `wan na`, `gon na` | 852 / 974 | 225 / 190 | 321 / 63 |
| age written as | `56 -year-old` (space-hyphen, 19×, only in the doctor's opening summary) | `59-year-old` | `57 year old` / `20 year old` |

- Consequence: one tokenizer config does not fit all three. In `aci` neither punctuation nor casing nor digits are available: an age is `fifty nine` and a BP is `one twenty over eighty`; a regex written for `\d` finds nothing. In `virtassist`/`virtscribe` the text is PTB-tokenized (`do n't`, `mr .`, `i'm`), so a model trained on normal English sees out-of-vocabulary pieces (`n't`, `na`).
- All 207 dialogues are lowercase (`i` 9781×, `I` 0×); notes use first person with capital `I` 664× (`Medical Reasoning: I believe …`).

### 2.2 Titles [regex + verified]

| form | dialogue hits (datasets) | encounters | note hits |
|---|---|---|---|
| `mr . james` / `dr . gomez` (split dot) | 26 (virtassist 5, virtscribe 21) | 7 | 0 |
| `mr. james` / `dr. reyes` (unsplit dot) | 87 (virtassist 24, virtscribe 63) | 36 | 131 (`Mr. James`, `Dr. Reyes`) |
| `ms gomez` (no dot) | 2 (virtscribe) | 2 | – |
| `mister jackson` / `miss` / `ma'am` / `sir` | 34 (aci 19, virtscribe 13, virtassist 2) | 24 | 0 |
| `doctor isabella` (word + name) | 45 (aci 34) | 38 | 19 `Dr. X` in notes |

- Both split and unsplit dots coexist, sometimes inside the same encounter (`D2N096`: `mrs. anderson` and `D2N092`: `mrs . martinez`). `aci` spells `mister` out. Rule must accept `(mr|mrs|ms|miss|mister|dr|doctor)\s?\.?\s` + name.
- **Trap**: `next radiographs , mr arthrogram of the right hip` (`D2N032`) = *MR arthrogram* (magnetic resonance), not Mr. Arthrogram. Title rule must exclude a short imaging/med vocabulary.
- `doctor put you on` (`D2N110`), `doctor said` are not names: the `doctor <x>` rule needs a stop-list of verbs.

### 2.3 Abbreviations [regex]

- `a.m.`/`p.m.` only in notes (1 each: `5 a.m.` `D2N172`, `2:00 am` `D2N026`); no `b.i.d.`-style dotted abbreviations anywhere; `po`/`iv`/`prn` 13 dialogue + 5 note hits (`lisinopril 10 milligrams po daily` `D2N095`, `iv drug use` `D2N113`).
- Frequency expressions are spelled: `three times a day` (201 dialogue / 96 note), `3 times a week`.
- `x-ray` 209 dialogue / 131 note, always hyphenated; disease acronyms (`copd`, `gerd`, `chf`, `cad`, `ckd`, `pe`) 43 dialogue / 39 note; test acronyms (`ekg`, `ecg`, `ct`, `mri`, `cbc`, `cmp`) 186 / 145. `pe` = physical exam in `let's use my default pe` (`D2N102`), not pulmonary embolism.
- `a1c`: `hemoglobin a1c` 94 dialogue / 82 note; value sometimes spelled (`elevated at eight`, `D2N088`).
- No `i.e.`, `e.g.`, `vs.`, no street abbreviations (`st.`, `ave.`) → no addresses in the corpus at all.

### 2.4 Decimals and units [regex]

- Decimals `2.5` : dialogue 30 (virtassist 7, virtscribe 23, aci 0), note 117. Spaced decimal `8 . 3`: **1** (`medical record number 113322 . 44-year-old`, `D2N184`, which is an MRN followed by a sentence boundary, not a decimal). `8 point 3`: 0.
- Slash numbers `\d+/\d+`: dialogue 18 (all virtscribe), note 108. Only ~half are BP; the others are dates (`4/28/07`), murmur grades (`2/6`), and pupil/reflex scores. `130 over 90` spoken form: 9 dialogue hits (`D2N142`). `140 / 90` with spaces: 0.
- Units: `milligrams` spelled in dialogue (`1000 milligrams`, `20 milligrams`) vs `mg` in note (400 hits). Glued `800mg` 2 dialogue / `4mg` 1 note. `pounds` in dialogue, `pounds`/`lbs` in note. `%` vs `percent` both present (`0.1 %` with a space in dialogue `D2N100`).
- Thousands with comma: 14 dialogue (`1,792 grams` `D2N101`), 2 note (`23,000` WBC `D2N182`, `50,000 IU` `D2N027`).

### 2.5 Dates and times — every format found [regex + verified]

| format | dialogue (datasets) | note | examples |
|---|---|---|---|
| `mm/dd/yyyy` | 33 (virtscribe only) | 39 (virtscribe 37, aci 2) | `date of birth 9/29/1957` `D2N098`; `completed on 08/03/2020` `D2N103` |
| `mm/dd/yy` | 8 (virtscribe) | 1 | `date of birth , 4/28/07` `D2N100`; `On 09/17/20 we saw Mrs. Sanchez` `D2N103` |
| `m-d-yy` | 1 | 0 | `date of birth 8-23-97` `D2N143` |
| `month d(th) , yyyy` | 16 (virtscribe) | 0 | `february 15th , 1979` `D2N102`; `march 26 2001` `D2N138` |
| `month d(th)` no year | 9 (virtscribe) | 5 | `at last visit on april 16th` / `April 16` `D2N142` |
| `month yyyy` / `month of yyyy` | 9 | 3 | `august of 2019` `D2N140`; `since May 2020` `D2N141` |
| bare year `19xx/20xx` | 70 (virtscribe only) | 100 (virtscribe 84, aci 16) | `in 2019`, `early spring of 2022` `D2N138`; `07/2021` `D2N101` |
| year spelled out | 3 (aci) | 0 | `november of two thousand twenty` `D2N186`; `two thousand eighteen` `D2N058` |
| `the 24th` (ordinal only) | 2 | 21 (mostly `the 5th MTP joint`, `the 7th grade`) | `scheduled on the 24th for surgery` `D2N025` |
| `date of birth` phrase | 33 (all virtscribe, 33 encounters = 16% of corpus, **82% of virtscribe**) | **0** | – |
| `date of service` / `today is` | 9 | 9 | `the date of service is 7/9/2021` `D2N105` |
| weekday | 36 | 13 | `if its saturday or sunday` `D2N088`; `Monday through Thursday` `D2N140` |
| clock time | 4 | 1 | `wake up at about 5:00 in the morning` `D2N172`; `[ inaudible 00:09:25 ]` `D2N138` |
| season | 90 (ambiguous: `had a bit of a fall` `D2N090` is not a season) | 40 | `end of the summer` `D2N088` |
| holiday | 14 (5 enc.) | 3 | `we just had easter` `D2N117`; `getting ready for thanksgiving` `D2N177` |
| `… ago` | 167 (102 enc.) | 112 | `about three weeks ago` `D2N090` |
| `last/this past …` | 137 (92 enc.) | 67 | `in the last week or so` `D2N088` |
| `NN-year-old` | 93 (92 enc.) | 211 | `59-year-old male` |
| `NN years old` | 1 | 9 | `anemic since i was 13 years old` `D2N073` |
| `at age NN` | 2 | 4 | `Appendectomy at age 7` `D2N094`; `breast cancer at age 40` `D2N101` (family history) |
| `in his 40s` | 4 | 0 | `died of dialysis in his 40s` `D2N185` |
| covid / pandemic anchor | 38 / 9 | 26 / 2 | `during the pandemic` `D2N160`; `pottery during the COVID-19 pandemic` `D2N083` |

- Absolute dates are concentrated in `virtscribe` (dictated header + dated results: `ECG, 12/26/2020.`); `virtassist` has almost none; `aci` has them only spelled out or as years in notes.
- No `o'clock`, no `d of month`, no ISO dates.

### 2.6 Disfluencies, stutters, contractions [regex]

- `um/uh` 2046 (142 enc.), `yeah/okay/alright` 7248, `you know/i mean/kind of` 1655. Stutter fragments: `we- we're`, `l- like`, `med- ,`, `t- ,` (310 hyphen-terminated tokens in 81 encounters; 1 in a note: `Hemoglobin- 8.2` `D2N020`). A name can be stuttered too (`an- another`), so a name regex anchored on `\b` must tolerate a preceding `x- ` fragment.
- Split negation `do n't`/`ca n't` 1398 vs unsplit `don't` 3; `wan na`/`gon na` 1227 vs `wanna` 1. `'em` 3 (`D2N138`). Curly apostrophes never in dialogue, 39 in notes.
- Number words 2458 in dialogue (aci 1871) vs digits 864; `fifty-nine` hyphenated compound: 0 (aci writes `fifty nine`). Ordinal words 404.

### 2.7 Note casing [regex]

- 19 sentences start lowercase (`. his endo…` `D2N093`, `email, or call if symptoms worsen` `D2N098`); 40 mixed-case tokens are units/chemistry (`dL`, `pH`, `mmHg`); 1986 Title-case tokens after a lowercase word include names (`Andrew Campbell`) but also weekdays, drugs (`Tylenol`, `Robitussin`), eponymous tests (`Neer’s`, `Hawkins-Kennedy`), brands (`Zarbee’s`, `Vick's`), and holidays. Cased NER on notes will produce PERSON false positives on eponymous tests/brands unless those are whitelisted.

### 2.8 Conclusion: detokenizer / truecaser? [judgment]

- **Detokenization** (`do n't`→`don't`, `mr .`→`mr.`, `, `→`,`) is cheap and safe for `virtassist`/`virtscribe`, and useless for `aci` (no punctuation to fix). It only matters if the NER model is a cased transformer trained on normal English. Keep an offset map: every merge removes exactly one space, so a per-character index array (`new_idx → old_idx`) is enough.
- **Truecasing** dialogue is risky: a truecaser trained on general text will capitalize `may`, `bill`, `mark`, `will`, `grace` inconsistently and will not know `dax`/`dragon` are products. It would also capitalize sentence starts in `aci` where sentences are not delimited. Better: use a casing-insensitive model (GLiNER-style / lowercase-finetuned token classifier) or lowercase the *training* data for a token classifier so that train and inference match. Fine-tuning on lowercase text makes the truecaser unnecessary.
- **Number words**: for `aci` a spelled-number normalizer (`fifty nine` → `59`) is required before any age/DOB/vitals rule, or the rules must be written on number words directly. Normalize only for detection, never rewrite the text.
- What a pre-processor must NOT do: delete characters (ZWSP, `ß`), merge lines, or change bracketed tags — those shift offsets and break the span → original mapping.

## 3. PII inventory

Script: `scripts/eda/03_pii_inventory.py` → `outputs/eda/pii_inventory.json` (patterns, counts, surface forms, examples) and `outputs/eda/pii_samples.txt` (20 random hits per category, hand-checked). Gazetteers live in `scripts/eda/namelib.py`. Precision below = fraction of the 20 random hits that are a true instance of the category **[verified]**; recall is estimated with probes (encounters where a near-universal category has no hit, then read) **[verified]**.

### 3.1 Inventory table [regex + verified]

| category | hits (dlg / note) | encounters (%) | aci / virtassist / virtscribe (encounters) | where | precision (20 hits) | surface forms / notes |
|---|---|---|---|---|---|---|
| patient first name, greeting (`hi , andrew`) | 159 / 5 | 151 (73%) | 76/112 · 52/55 · 23/40 | 156/159 in dialogue lines 0–2 | 18/20 (`good afternoon nap`, `morning aldosterone`) | 144 distinct names |
| patient name, intro / header (`andrew is a 59-year-old`, `next patient is emily hernandez`, note `Andrew Campbell is a …`) | 102 / 212 | 173 (84%) | 80 · 55 · 38 | note HPI (129) + ASSESSMENT AND PLAN (29) + ASSESSMENT (20); dialogue line 0–4 | 20/20 | note: `Mr. John Perry is a 61-year-old`, `Ashley is a 14-year-old` (first name only in ASSESSMENT) |
| title + name (`mr . james`, `Ms. Martinez`, `mister jackson`) | 156 / 132 | 84 (41%) | 22 · 26 · 36 | note HPI 85 | 19/20 (`miss swimming`) | mixes patient (`mr. allen`) and clinicians (`dr. evelyn`, `dr. martha rivera` = PT) |
| doctor / staff name | 66 / 19 | 46 (22%) | 14 · 10 · 22 | dialogue line 0–2 (31/66), note HPI | 18/20 (`and doxycycline can`) | `dr. reyes`, `doctor white`, `dr. jacob ward` (full name), `dr. evelyn` (first name); staff first names without title (`helen`, `madison`) are **not** matched by any pattern |
| relative / friend (relation word) | 328 / 134 | 112 (54%) | 69 · 18 · 25 | HPI, FAMILY HISTORY, SOCIAL HISTORY | 17/20 (`FAMILY HISTORY` header) | `wife` 44, `dad` 27, `kids` 26, `sister` 24, `mom` 23 … relation words alone are not identifying |
| **named** relative / friend | 7 / 2 | 7 (3%) | 2 · 0 · 5 | | 9/9 | `my son sean` (`D2N124`), `my daughter , theresa` + note `his daughter, Theresa` (`D2N141`), `my sister , hannah`, `your friend , tyler` (`D2N027`), `my friend diane` (`D2N035`) |
| DOB phrase | 33 / 0 | 33 (16%) | 0 · 0 · **33/40 (82%)** | always dialogue line 0 (33/33) | 20/20 | `date of birth` 32, `birth date` 1; no DOB in any note |
| `birthday` (not DOB) | 6 / 3 | 4 | | | — | `happy birthday coming up on monday` (`D2N043`) = relative date leak |
| exact age | 114 / 229 | 171 (83%) | 79 · 55 · 37 | note HPI 143, A&P 38; dialogue line 0–4 | 20/20 | `59-year-old` 93, `57 year old`, `56 -year-old`, `9-month-old`, `in his 40s` (relative's age), `13 years old`, `at age 7` |
| absolute date | 91 / 111 | 50 (24%) | 13 · **0** · 37 | note HPI 60, RESULTS 18; dialogue line 0 (56/91) | 19/20 (`18-20/28` score) | 145 distinct forms: `9/29/1957`, `4/28/07`, `8-23-97`, `march 26 2001`, `february 15th , 1979`, `04/2020`, `in 2018`, `two thousand eighteen`, `on the 24th` |
| month / season / holiday | 163 / 49 | 71 (34%) | 30 · 15 · 26 | | 19/20 (`back summer salt`) | `summer` 43, `spring` 23, `march` 15, `the holidays` 9, `easter`, `thanksgiving` |
| weekday | 43 / 16 | 23 (11%) | 7 · 9 · 7 | | 19/20 (`friday's` restaurant) | `saturday` 18, `monday` 12 |
| relative date | 948 / 571 | 204 (99%) | 109 · 55 · 40 | everywhere | 20/20 | 523 forms: `about three weeks ago`, `last week`, `in 2 weeks`, `since i was 13`, `during the pandemic` |
| clock time (digits) | 4 / 2 | 4 (2%) | 0 · 1 · 3 | | 6/6 | `2:00 am` (`D2N026`), `5:45`, `7:00` (`D2N027`), `5:00` / `5 a.m.` (`D2N172`), `00:09:25` (inaudible tag) |
| time-of-day phrase (`in the morning`, `at night`) | 130 / 57 | 65 (31%) | | | not PII (dosing times) | keep |
| location | 78 / 29 | 45 (22%) | 33 · 3 · 9 | SOCIAL HISTORY, HPI | ~14/20 | `trip to Columbus, Georgia` (`D2N080`), `lives in dallas , georgia` (`D2N076`), `skiing in vermont` (`D2N126`), `denver`, `west coast`, `san francisco`, `main street` (`D2N039`), `kaiser permanente broadway`, `the woods`, `the trails`, `County Hospital` (`D2N025`), `county surgical` (`D2N025`) |
| employer / occupation / school | 87 / 50 | 61 (29%) | 32 · 14 · 15 | SOCIAL HISTORY 22, HPI 29 | ~13/20 | `auto repair shop`, `bakery` (`D2N129`, `D2N152`), `enterprise rent-a-car`, `the plant`, `farmer` (`D2N056`), `veteran`, `military`/`army`, `retired from my office job`, `7th grade`, `high school` (sports history), `coach`, `desk at work` |
| hobby / sport | 309 / 148 | 117 (57%) | 74 · 27 · 16 | HPI, SOCIAL HISTORY | ~15/20 | `soccer`, `skiing`, `swimming`, `hiking`, `pottery`, `tennis`, `golf`, `pickleball`, `barre`, `mountain biking`, `photography`, `gardening`; many are also the injury mechanism |
| event | 81 / 21 | 43 (21%) | 31 · 9 · 3 | | 17/20 | `vacation`, `birthday party`, `house parties`, `easter egg hunt`, `super bowl last weekend`, `thanksgiving`, `holiday weekend … on my boat` |
| sports team | 8 / 2 | 6 (3%) | 6 · 0 · 0 | SOCIAL HISTORY | 10/10 | `denver nuggets fan` (`D2N189`), `san francisco 49ers fan` (`D2N157`), `dallas cowboys`, `chiefs`, `bulls`, `packers`, `red wings` / `detroit is a hockey town` (`D2N154`) |
| religion / ethnicity / nationality | 1 / 2 | 3 (1%) | | | 2/3 | `Elderly Caucasian male` (`D2N026`), `people of East Asian descent` (`D2N125`, generic) — essentially absent |
| phone / email / MRN / IDs / insurance | 50 / 15 | 21 (10%) | 6 · 0 · 15 | dialogue line 0 (19/50) | IDs 11/11 | **MRN in 11 encounters**: `mrnr472348`, `mrn 3-8-4-9-2-0`, `mrn : 5484367`, `mrn 869723`, `mrn 385729`, `mrn49282721`, `mrn 0000123`, `mrnq27631`, `mrn : abc123`, `medical record number is 123546`, `medical record number 113322`. Insurers: `blue cross blue shield` (`D2N101`), `kaiser` (`D2N138`), `medicare` (`D2N025`), `the va` (`D2N186`). **No phone numbers, no e-mails, no SSN, no addresses** |
| vehicle | 47 / 13 | 25 (12%) | | | — | `car accident` ×8 (an event, `D2N127` "multiple car crashes"), `tandem bicycle`, `motor boat`, `truck` |
| pet | 29 / 4 | 23 (11%) | 20 · 1 · 2 | | 15/20 (`a cat scan` ×5 before fix) | `my dog`, `my three cats`, `a dog , bird , and bunnies` (`D2N100`); no pet names |
| product / brand (non-drug) | 408 / 21 | 85 (41%) | 21 · 55 · 9 | | 19/20 (`flare ups`) | `dragon` 280 (all virtassist), `dax` 72, `mcdonald's`, `brooks`, `cvs`, `panoxyl`, `rex center`, `nike`/`asics`/`new balance`, `kaiser permanente` |
| other quasi-identifier | 39 / 25 | 24 (12%) | 14 · 4 · 6 | SOCIAL HISTORY | ~12/20 | `twin` (`D2N076`, `D2N126`), `adopted` (`D2N046`), `lives with her parents and sister`, `roommate`, `renovating my basement`, `married for 25 years` (`D2N074`), `zoom meeting` (telehealth), `iv drug use` (`D2N113`) |

### 3.2 Recall probes [verified]

- **Patient name in dialogue**: patterns hit 186/207 encounters (90%). Of the 21 misses, reading the first lines shows a bare vocative name in 17 (`alright scott so so i understand`, `alright brittany so`, `so anna good to see you`, `okay raymond it looks like`, `hi good afternoon joseph`) — all `aci`, where there is no comma to anchor on; 1 has the surname spoken alone as a one-word turn (`[doctor] howard` answering `how do you say your last name`, `D2N190`); 3 have no name in the opening (`D2N043`, `D2N046`, `D2N056`). A **first-name gazetteer applied to the first 3 turns** is therefore mandatory for `aci`; punctuation-anchored rules alone cap recall at ~90%.
- **Patient name in note**: patterns hit 158/207. Of the 49 misses, 4 are pattern gaps (`Jerry Nguyen a 54-year-old` — missing "is", `D2N091`; `Karen` only in PLAN of `D2N055`; `Murphy` in `D2N146`/`D2N192`) and **45 notes contain no patient name at all** (39/112 `aci`, 4 `virtassist`, 2 `virtscribe`): they use `Patient reports …` / `The patient is a 31-year-old female`. So the patient name is a *dialogue-side* problem in `aci`, and a *note-side* problem (full name, title+surname, repeated in HPI and A&P) in `virtassist`/`virtscribe`.
- **Age**: 171/207 encounters have an explicit age; the 36 without are `aci` notes with no HPI (and `D2N026`, `D2N031`, `D2N075` virtscribe). Ages derived from DOB+DOS (`100-year-old`, `9-month-old`) appear only in the note.
- **Absolute dates**: 0 in `virtassist`; `virtscribe` 37/40 encounters (dictated header + dated results); `aci` 13/112 (years in SURGICAL HISTORY: `RCA stent 2018`).
- **Third-party names**: 7 encounters with a named relative/friend by pattern, plus staff first names found only by reading (`madison` `D2N105`, `helen` `D2N138`), the referring doctor full name (`dr. jacob ward`, `D2N180`), the doctor's first name (`doctor isabella` `D2N098`, `dr. evelyn` `D2N076`, `doctor ruth` `D2N197`), and the patient's doctor-surname line (`howard`, `D2N190`). Expect ~10% of encounters to contain a name that is **not** the patient's.
- **Swapped speaker tags**: a cue-based heuristic (doctor phrases under `[patient]`, patient phrases under `[doctor]`) flags `D2N082`, `D2N057`, `D2N059`; reading adds `D2N114`, `D2N056`, `D2N190`. ≥6/207 (3%, 4 of them `aci` eval) — "the name after `hi` in a `[doctor]` line is the patient" is wrong 3% of the time.

### 3.3 Per-dataset profile of PII [regex]

| | aci (112) | virtassist (55) | virtscribe (40) |
|---|---|---|---|
| patient name in dialogue | first name as vocative, no punctuation; full name in opening summary in ~20% (`russell ramirez is a 45 -year-old`) | `hi , <first>` line 0 + `<first> is a NN-year-old` line ~4 (scribe preamble) | dictated header `next patient is <first> <last>` + `mr./mrs. <last>` throughout |
| patient name in note | **absent in 35%**; else `First Last is a pleasant NN-year-old` | `First Last is a NN-year-old` in HPI and again in A&P; `Ms. First Last` | `First Last` + `Mrs. Last` several times, also in PLAN |
| DOB / MRN / DOS | none | none | DOB 82%, MRN 25%, DOS ~10% — all in dialogue line 0 |
| absolute dates | years only (`2018`) | none | dense (`08/03/2020`, `ECG, 12/26/2020`) |
| quasi-identifiers | richest: hobbies, teams, pets, events, occupations in SOCIAL HISTORY | `dax`/`dragon` commands, hobbies in HPI | employer, insurer, military, staff names |
| numbers in dialogue | spelled out | digits | digits |

## 4. Names deep-dive

Script: `scripts/eda/04_names.py` → `outputs/eda/name_lexicon.json` (per-encounter names by role and field, resolved patient first/last, doctor/staff/relative names, shared names) and `outputs/eda/collisions.json`. Extraction = phase-3 patterns + a first-name gazetteer on the first 3 turns and on `so|alright|okay <name>` vocatives. Numbers are from the script **[regex]**, examples checked **[verified]**.

### 4.1 Cross-field consistency (patient name)

| relation between dialogue and note | encounters |
|---|---|
| same first name in both fields | 125 |
| first name only in dialogue (note has no name or only `Ms. Last`) | 66 |
| first name only in note (`D2N114` swapped tags → `Julie` said by the "patient"; `D2N179` `elijah`) | 2 |
| first name in neither (patient addressed only as `mrs . martinez`/`mr . ward`, or never named) | 10 (`D2N092`, `D2N096`, `D2N132`, `D2N005`, `D2N017`, `D2N125`, `D2N043`, `D2N046`, `D2N056`, `D2N059`) |
| spelling variant / typo | 1: `jaqueline` (dialogue) vs `Jacqueline Price` (note), `D2N188` |
| surname in both fields | 46 |
| surname only in note | **102** (the note adds the surname the dialogue never says: `andrew` → `Andrew Campbell`) |
| surname only in dialogue | 6, all `aci` (`stephanie morales`, `mister jackson`, `karen nelson`, `russell ramirez`, `cruz`, `thompson`) — the note has no name |
| surname in neither | 53 |

- 101 encounters have *first name only* in the dialogue and *first + last* in the note: the note is the richer source, so a model that sees the fields independently gets no help from the dialogue for the surname, and a surrogate strategy must link `andrew` ↔ `Andrew Campbell` ↔ `Mr. Campbell` across fields.
- Note mentions per patient: median 2, max 9 (`D2N185`: `Ashley James` + `Ashley` ×3 + `Ms. James`). 5 notes reuse the **bare first name** after the full name (`Diana has a 4/6 systolic murmur`, `Kayla reports`, `Matthew`, `Ashley`, `Katherine`) → a note rule that only catches `First Last is a` misses the later bare mentions.
- No nickname/formal pairs were found by pattern (`bob`/`robert`, `bill`/`william`); the only variant is the `jaqueline` typo. Reading found one pet name (`dino` for `randy`, `D2N099`).

### 4.2 Name positions in dialogue [regex]

- 202/207 dialogues mention the patient's name; in **195 of them the first mention is in line 0** (`aci` 104/107, `virtassist` 53/55, `virtscribe` 38/40); the latest first-mention is line 4 (`virtassist` scribe preamble `so , martha is a 50-year-old`).
- Mentions per dialogue: median 4, max 15. **59 encounters (29%) have ≥1 mention at line ≥5** (106 such mentions): vocatives mid-visit (`okay , tyler , i'm examining you now`, `thanks , susan`, `bye randy`) and dictated assessments (`diana has a 4/6 systolic murmur`, `D2N105`). A "first N lines only" strategy misses ~30% of encounters.
- Dictated header (`next patient is …`, `date of birth`, `mrn`, `<first> <last> is a NN-year-old`): `virtscribe` 36/40 (90%), `aci` 2/112, `virtassist` 1/55. In `aci` the opening is instead a spoken summary `so tyler is a 56 -year-old male who presents …` (≈25 encounters), in `virtassist` the scribe preamble `<first> is a NN-year-old <sex> with a past medical history …` (≈50 encounters).

### 4.3 `mr . james`: first or last name? [regex + verified]

- Of 126 title+name tokens in dialogue resolved against the note, **100 are title + surname**, 20 unresolved (note has no name), 6 "title + first" are regex artifacts of `hi , andrew` (no real title). Verified: `mr . james` = *Jose James* (`D2N098`), `mrs. jackson` = *Sophia Jackson*, `ms. ross` = *Grace Ross*, `mrs. wood` = *Kelly Wood*. **Rule: a token after `mr/mrs/ms/miss/mister` is a surname**, even when it is also a common first name (`james`, `taylor`, `allen`, `scott`, `howard`, `ross`, `cook`).
- The inverse holds for `dr`/`doctor`: 10 of ~50 doctor names are first names (`dr. evelyn`, `doctor isabella`, `dr. alice`, `doctor ruth`, `dr. martha rivera`, `doctor michael`, `dr. jacob ward`, `dr. juan`, `dr. tyler`, `dr. vincent`).
- Title forms per dataset: `aci` uses `mister`/`miss` spelled out (`mister jackson is a 45 -year-old`), `virtassist`/`virtscribe` use `mr .`/`mr.`/`ms`/`mrs.`.

### 4.4 Doctors, staff, relatives [regex + verified]

- Doctor names: 50 distinct in dialogue (`miller`, `moore`, `reyes`, `white`, `hughes`, `harris`, `sanchez`, `scott`, `baker`, `flores`, `howard`, `gomez`, `lopez`, `martinez`, `gutierrez`, `phillips`, `rogers`, `gonzalez`, `taylor`, `cooper`, `evelyn`, `isabella`, `ruth`, `alice` …), 16 distinct in notes (`Dr. White` ×3, `Dr. Reyes` ×2 …). 48 encounters (23%) name at least one clinician. Doctor surnames are drawn from the same pool as patient surnames (`martinez`, `sanchez`, `taylor`, `cooper`, `scott`), so surname ≠ role.
- Staff first names (no title, lowercase, mid-dialogue): `helen` (`D2N138`), `catherine`, `madison` (`D2N105`, found by reading only). Named relatives/friends: `sean`, `theresa`/`Theresa`, `hannah`, `tyler`, `diane` (7 encounters).
- No encounter has doctor surname == patient surname.

### 4.5 Collisions: names that are also words [regex + verified]

38 of the 209 distinct name tokens in the corpus collide with ordinary or medical vocabulary. Those that actually occur in the non-name sense in this corpus (counts = occurrences as a word, note / dialogue):

| token | name role in data | word sense in data (note / dlg) | example in word sense | how to tell apart |
|---|---|---|---|---|
| `white` | `Dr. White` ×6 (note), `doctor white` | 2 / 24 | `white blood cell count`, `white coat` | title before (`dr .`) vs noun after (`blood cell`, `coat`) |
| `young` | patient surname `Young` | 2 / 18 | `she is young and otherwise healthy` | capitalized + `Mr./Ms.` or adjective position |
| `walker` | patient surname `Walker` ×11 | 2 / 2 | `requires a walker to ambulate` | determiner `a walker` → device |
| `brown` | surname `Brown`, `Dr. Brown` | 2 / 6 | `black and brown nevus`, `brown liquor` | colour context |
| `cook` | surname `Cook` ×6 | 0 / 5 | `prepare and cook meals` | verb position |
| `green` | surname `Green` (`Tyler Green`) | 2 / 3 | `green phlegm`, `green tea` | noun after |
| `price` | surname `Price` ×7 | 2 / 2 | `reduced price`, `if the price improves` | `the price` |
| `frank` | first name `Frank` | 1 / 2 | `frank obstruction` (medical adjective) | adjective before noun |
| `hill` | surname `Hill`, eponym (Hill-Sachs) | 0 / 3 | `ski down the hill`, `walking up a hill` | `a/the hill` |
| `murphy` | surname `Murphy` | 0 / 3 | `positive for murphy's sign` | `'s sign` |
| `thompson` | surname `Thompson` (`mister thompson`) | 0 / 4 | `your thompson test is negative` | `test` after |
| `taylor` | surname `Taylor`, `Miss Taylor` | – | – | – |
| `mark` | first name `mark` (`hey there mark`) | 0 / 2 | (all name uses; `mark` as verb absent) | – |
| `wood` | surname `Wood` (`kelly wood`) | 0 / 2 | `knock on wood` | idiom |
| `baker` | surname, eponym (Baker's cyst) | 0 / 1 | `around the mount baker area` (place) | – |
| `allen` | surname `Allen` ×12, eponym (Allen test) | 0 / 1 | `allen test` not present; 15 eponym hits are `Allen` the patient | – |
| `jones` | surname `Jones`, eponym (Jones fracture) | – | `jones fracture` absent | – |
| `smith` | surname `Smith` ×9 | 0 / 1 | `smith magenis syndrome` (`D2N076`, genetic syndrome) | `syndrome` after |
| `ray` | – (not a name here) | 185 dialogue | `x-ray`, `x - ray` | always `x-`/`x -` before |
| `may` | – | 97 | modal verb `it may be` | verb position; as month `in may` |
| `hope` | – | 28 | `i hope` | verb |
| `summer` | – | 26 | season | – |
| `april`, `june` | – | 10 / 7 | months (`june 3rd 2021`) | followed by day/year |
| `grace`, `dawn`, `candy`, `sunny`, `brooklyn`, `madison`, `nick`, `rich` | mostly not names here (`madison` once a staff name) | 1 each | `by the grace of`, `rich foods`, `nick` (cut) | – |
| `lyme` (33), `cushing` (5), `wilson's` (1), `achilles` (16), `lachman` (16), `lisfranc` (11), `mcmurray` (9), `tinel` (8), `colles` (4), `spurling` (4), `homan` (4), `barre` (4), `tourette` (2), `guillain` (1), `hawkins-kennedy` (2), `neer` (2), `pap` (16) | **never a person** in this corpus | all | `lyme titer`, `cushing's syndrome`, `achilles tendon`, `lachman test`, `barre classes` (exercise, not Guillain-Barré) | eponym whitelist; `'s`, `test`, `sign`, `fracture`, `disease`, `syndrome`, `tendon`, `titer` after |

- A pure first-name gazetteer on dialogue would produce ~400 false positives (`ray` 185 from `x-ray`, `may` 97, `hope` 28, `summer` 26, `april`, `june`) against ~900 true name tokens: precision ≈ 70% without context rules. The false positives are concentrated in 6 tokens, so a stop-list (`ray`, `may`, `hope`, `summer`, `april`, `june`, `will`, `grace`, `candy`, `rich`, `nick`, `sunny`) plus a cue requirement (greeting, title, intro, vocative comma, line 0) fixes it.
- Medical eponyms present as diseases/tests: `lyme` 33 (code A69.20 in 2 encounters), `cushing(’s)` 5, `wilson's` 1, `tourette's` 2, `guillain(-barre)` 1 + 15 orthopaedic test names. None of them is a person in the corpus, and none of the patient/doctor surnames is an eponymous disease name (closest: `Allen`, `Jones`, `Baker`, `Hill`, `Murphy`, `Harris`, `Thompson`, `Bennett` — surnames here, tests elsewhere).

### 4.6 Names shared across encounters [regex + verified]

- 143 distinct full names, **10 reused** across 2–3 encounters. In every case age, sex-consistent codes and history differ: `michelle king` = 44F M25.561 (`D2N126`) / 10F K21.9 (`D2N181`) / 39F K59.09 (`D2N182`); `diana scott` = 100F murmur (`D2N105`) / 61F wrist fracture (`D2N077`); `brittany edwards` = 32F IBS / 76F ankle fracture; `bruce ward` = 52M pituitary / 60M hepatitis. → **synthetic name reuse, not the same patient**. 3 full names appear in both train and eval (`brittany edwards`, `bruce ward`, `sophia jackson`), 24 of 58 eval first names also occur in train.
- First names: 174 distinct for 207 patients (`matthew` ×6, `elizabeth`/`betty`/`michelle`/`anna`/`sophia` ×4); surnames: `hernandez` ×5, `walker`/`gutierrez`/`ward`/`allen`/`price` ×4 — a top-100 US census distribution.
- Impact: a "consistent surrogate per patient" must be keyed on `encounter_id`, never on the name string; the same real name maps to different surrogates in different encounters (and must), and the eval split contains names seen in train, so memorizing train names is not cheating but also not useful.

## 5. Ambiguous and adversarial tokens

Script: `scripts/eda/05_ambiguous.py` (+ `scripts/eda/numwords.py`, a spelled-number normaliser used only for analysis) → `outputs/eda/numeric_patterns.json`. Eponym/name collisions are in §4.5 and `collisions.json`.

### 5.1 Token classes that are PII in one context and clinical in another [verified]

| class | PII reading | clinical reading | examples in data |
|---|---|---|---|
| eponymous disease / test ↔ surname | `Mr. Allen`, `Dr. Baker`, `Ms. Jones`, `mister thompson`, `Hill`, `Murphy` | `allen test`, `baker's cyst`, `jones fracture`, `thompson test` (`D2N154`), `hill-sachs`, `murphy's sign` (`D2N146`), `lyme titer`, `cushing's`, `wilson's`, `tourette's`, `guillain-barre`, `achilles`, `lachman`, `lisfranc`, `mcmurray`, `tinel's`, `colles`, `spurling's`, `homan's`, `hawkins-kennedy`, `neer's`, `smith magenis syndrome` (`D2N076`) | in this corpus every eponym occurrence is the *disease/test* sense, and every colliding surname occurrence is the *person* sense; the separating cues are the title before (`mr`, `dr`) and the noun after (`test`, `sign`, `fracture`, `syndrome`, `disease`, `titer`, `tendon`, `'s`) |
| colour / adjective ↔ surname | `Dr. White`, `Mr. Young`, `Ms. Green`, `Mr. Brown` | `white blood cell`, `white coat`, `she is young`, `green phlegm`, `brown liquor`, `frank obstruction` | title or capital + `is a NN-year-old` vs determiner/adjective slot |
| occupation / object ↔ surname | `Ms. Walker`, `Ms. Cook`, `Mr. Price`, `Mr. Wood` | `requires a walker`, `cook meals`, `the price improves`, `knock on wood` | `a walker`, verb position |
| month ↔ first name | (none in data) | `may` 97×, `april`, `june` as months | – |
| `ray` | – | `x-ray` 185× in dialogue (`x - ray` variants) | always after `x-` |
| product ↔ person | – | `dax` 72 (`tell dax a little bit about you`), `dragon` 280 (`hey , dragon , show me the ekg`), `brooks` (shoes), `enterprise`, `kaiser` | `hey , dragon` + command verb (`show` 148, `order` 61, `finalize` 48); `dax` always with `tell`/`about`/`using` |
| imaging ↔ title | `mr . james` | `mr arthrogram` (`D2N032`), `mra`, `ct`, `pe` (= physical exam, `D2N102`) | next token is a modality, not a name |
| age ↔ other number | `56 -year-old`, `you are 100`, `at age 7` | `type two`, `two out of six`, `grade three`, `15-20 pounds` | suffix `-year-old` / prefix `age` only |
| DOB / MRN ↔ clinical numbers | `9/29/1957`, `mrn 3-8-4-9-2-0`, `mrnq27631`, `113322 . 44-year-old` | `2/6 murmur`, `6/10 pain`, `4/5 strength`, `250/50 dose` (Wixela), `120/50` BP, `1/23/2020 ct` | see §5.2 |
| date ↔ score | `18-20/28` (MoCA, `D2N141`) looks like a date; `12/16` score | `on 09/17/20` | a slash-number with a 3rd component or a 4-digit year is a date; `/28`, `/10`, `/6`, `/5` are scores |
| relative's age ↔ patient age | `died in his 40s`, `brother: 9 years old` | – | subject of the sentence |
| staff first name ↔ word | `helen`, `madison`, `catherine` | `madison` (street / word) | only a gazetteer + verb frame (`have X come in`, `went through with X`) |
| lowercase everything | `i'm doctor isabella`, `hi mr . james`, `kelly wood . date of birth` | – | 1 capitalized name token (`Julie`) in all dialogues; **casing carries zero signal in dialogue** |

### 5.2 Numeric inventory [regex, approximate classes]

5,380 numeric tokens (digits or spelled-out) in the corpus; 41% of dialogue numbers are spelled out (`aci`: nearly all). Classes assigned by a context rule (left 60 / right 40 chars); the `other` bucket (1,359, mostly spelled `two`/`three` in dialogue: `two of the three memory words`, `a week or two`) was not hand-labelled.

| class | n (dlg / note) | spoken | label | example |
|---|---|---|---|---|
| dose (`81 mg`, `1000 milligrams`, `lortab 5 milligram`) | 808 (365 / 443) | 205 | **must-keep** | `aspirin 81 mg daily` |
| duration / relative time (`4 days ago`, `next 30 days`) | 791 (445 / 346) | 428 | must-keep (timeline) | `approximately 4 days ago` |
| exact age | 344 (114 / 230) | 11 | **privacy → bucket**; keep for age-specific codes | `i know that you are 74` (`D2N199`), `since i was 13 years old` |
| lab value | 313 (185 / 128) | 132 | must-keep | `creatinine 0.7`, `BNP 3000`, `wbc 23,000`, `ejection fraction thirty five percent` |
| score / grade (`6/10`, `2/6`, `4/5`, `11 out of 10`, `clark level 4`) | 309 (133 / 176) | 102 | must-keep | `three out of six systolic ejection murmur` |
| blood pressure (`120/50`, `one twenty over fifty`, `146 to 155`) | 270 (86 / 184) | 55 | must-keep (I10 vs I11/I13 context) | `130 over 90` |
| stage / grade / type (`type 2`, `stage 3`, `grade three`) | 139 | 46 | must-keep (E11 vs E10, N18.3) | `type two diabetes` |
| frequency (`three times a day`) | 123 | 64 | must-keep | |
| HbA1c (`8`, `7.4`, `seven . four`) | 87 (34 / 53) | 29 | must-keep (E11.65) | `hemoglobin a1c … elevated at eight` |
| vitals: temperature 52, heart rate 49, weight 46, resp. rate 26, height 20, SpO2 2, BMI 3 | 198 | ~90 | must-keep; weight/height extreme values are fingerprints (`120 pounds` at 100 y, `1,792 grams` breast tissue) | `ninety nine point seven` |
| counts (`1 pack of cigarettes`, `two cigarettes a day`, `3 kids`) | 82 | 30 | must-keep for pack-years / drinks; `3 kids` is a quasi-identifier | |
| measurement (`0.7 millimeter`, `4.6 cm`) | 74 | 27 | must-keep (melanoma staging, stone size) | |
| percent clinical (`EF 35%`, `sat 99%`) | 53 | 17 | must-keep | |
| year (`2018`, `twenty eighteen`) | 47 (32 / 15) | 7 | privacy-low; keep the clinical interval (`stent in 2018` → "3 years ago") | |
| absolute date (`08/03/2020`, `may 17`) | ~200 in §6 | 1 | **privacy**; generalize | |
| MRN / record number (`0000123`, `3-8-4-9-2-0`, `123546`, `abc123`) | 11 | 0 | **privacy** | |
| clock time | 7 | 5 | privacy-low | `five am` |
| money (`$ 850 after insurance`, `$ 9 for 90 days`) | 14 | 1 | neutral (but reveals insurance status) | |
| phone / e-mail / SSN | **0** | | | none in the corpus |

- Numbers that **look like PII but are clinical**: `250/50` (Wixela dose), `18-20/28` and `12/16` (cognitive scores), `2/6`, `4/5`, `6/10`, `1,792 grams`, `0000123`-style zero padding never occurs for a lab. Numbers that **look clinical but are PII**: `113322 . 44-year-old` (MRN glued to age), `3-8-4-9-2-0`, `you are 100` (no `-year-old`), `4/28/07`.
- Numbers that could **flip a code** if perturbed: HbA1c (E11.65 `D2N090` is assigned with A1c 10+; E11.9 `D2N088` with A1c 8 — the organizers did not code by threshold consistently), BP (I10 vs I11.0 `D2N156`/`D2N068`/`D2N084` — those depend on heart failure, not on the number), eGFR/creatinine (N18.30 `D2N157`, `D2N161`, `D2N207`), BMI (none of the codes here is obesity, but `bmi greater than 27 / 30` thresholds are discussed in `D2N021`), EF (I50.1 `D2N058`: `ejection fraction thirty five percent`), hemoglobin (D50.9/D64.9), melanoma thickness/stage (C43.59 `D2N029`). Keep all of them unchanged.

### 5.3 Dates: clinically anchored vs identifying [regex + verified]

| kind | dialogue | note | example |
|---|---|---|---|
| DOB | 36 | 0 | `date of birth 9/29/1957` |
| date of service | 1 | 0 | `the date of service is 7/9/2021` (`D2N105`) |
| anchored to a procedure / test / visit | 50 | 98 | `mitral valve repair done on 8/3/2020`, `ECG, 12/26/2020.`, `CT on 1/23/2020`, `last seen on november 24th 2020`, `stent in 2018` |
| unanchored | 12 | 15 | `enlisted early spring of 2022`, `thyroid profile from march 1st , 2021`, `changed insurance in 07/2021` |

- 148 of 212 dated expressions are *clinically anchored*: they give the age of a surgery/test. The clinical value is the **interval** to the visit, not the calendar date, so a per-encounter date shift preserves utility; converting to year only (`2020`) also preserves "chronic vs acute" (every code with acute/chronic in the description is listed in §6.2). The dictated header is the only place where the absolute date is needed to compute the age (`12/8/1920` + `7/9/2021` → `100-year-old`), i.e. DOB and DOS leak the age even when no age is written.

### 5.4 Medication / brand names that look like names or places [regex + verified]

- 76 distinct capitalized tokens precede a dose in notes (`Norvasc` 19, `Protonix` 11, `Lasix` 10, `Tylenol` 7, `Toprol`, `Crestor`, `Motrin`, `Bumex`, `Cozaar`, `Flovent`, `Ultram`, `Brilinta`, `Lipitor`, `Prozac`, `Zoloft` …). **None of them is in the first/last-name or place gazetteers**: drug names do not collide with person names in this corpus, but they *are* capitalized mid-sentence in notes (291 brand/drug tokens vs 218 person-name tokens), so a cased NER that relies on capitalization will over-fire on drugs unless it knows them.
- In dialogue the same drugs are lowercase (`metformin` 82, `tylenol` 80, `lisinopril` 72, `norvasc` 32 …).
- Non-drug products with name/place shape: `brooks` (shoe brand = surname `Brooks` of patient `D2N072`), `enterprise rent-a-car` (employer), `kaiser permanente broadway` (insurer + street), `mount baker` (place; `Baker` is also a surname), `friday's` (restaurant), `mountain dew`, `honey nut cheerios`, `head & shoulder's`, `zoom` (telehealth), `rex center` (ASR for "rec center"), `tumor guy` (ASR garble of a shoe brand).

### 5.5 What capitalization would give a cased NER [regex]

- Notes: 218 mid-sentence capitalized tokens are person names, but 1,661 are other capitalized words (drugs 291, tests, days, eponyms, `Patient`, `COVID`, section words), so capitalization has ~12% precision for names in notes.
- Dialogue: 2,400 lowercase tokens are in the first/last-name gazetteer (after removing `ray/may/hope/summer/april/june/will/...`), 379 are brand words, 31 place words — none capitalized. A cased model receives no signal; a lowercase-tolerant model or lowercased training data is required (see Decisions).

## 6. Temporal information

Script: `scripts/eda/06_temporal.py` → `outputs/eda/temporal_expressions.json` (inventory + code dependencies). Counts **[regex]**, code lists derived from the descriptions **[regex]** and checked against the encounters **[verified]**.

### 6.1 Inventory of temporal expressions

| type | hits (dlg / note) | encounters | formats found | generalization that keeps the codes |
|---|---|---|---|---|
| absolute date, numeric | 194 (75 / 119) | 38 (37 virtscribe) | `9/29/1957`, `08/03/2020`, `4/28/07`, `8-23-97`, `10/2020`, `03/2021` | DOB → remove or age bucket; others → year only or per-encounter shift (±k days, same k for both fields) |
| month day year (spoken) | 18 (18 / 0) | 11 | `august 1st , 2020`, `march 26 2001`, `february 15th , 1979`, `october 31st 1949`, `january 15th , '21` | same |
| month + day, no year | 14 | 4 | `april 16th`/`April 16` (`D2N142`), `may 17`, `june 14th`, `february 9th` | → month or "N weeks ago" |
| month + year | 12 | 5 | `august of 2019`, `June 2020`, `04/2020` | → year |
| year only | 99 (57 / 42) | 25 | `in 2018`, `2019`, `twenty eighteen`, `two thousand twenty`, `early spring of 2022` | keep year or shift; "stent in 2018" is clinically useful only as "3 years ago" |
| month only | 48 (41 / 7) | 22 | `in march`, `of june`, `last december` | keep (relative) or drop |
| season / holiday | 117 (83 / 34) | 44 | `summer` 43, `spring` 23, `the holidays`, `easter`, `thanksgiving`, `super bowl`, `holiday weekend` | keep seasons (allergy codes J30.x depend on them), generalize named holidays to "a holiday" |
| weekday | 59 (43 / 16) | 23 | `saturday`, `monday morning`, `thursday or friday` | neutral; can stay |
| clock time | 7 | 4 | `2:00 am`, `5:00`, `5 a.m.`, `7:00`, `00:09:25` | neutral |
| time of day | 264 | 86 | `at night`, `in the morning`, `this morning`, `last night` | keep (dosing, symptom timing) |
| relative "ago" | 284 (172 / 112) | 118 | `a week ago`, `two years ago`, `about three weeks ago` | **keep verbatim** |
| relative last/next/today/now | 2,501 | 206 | `now` 852, `today` 748, `last week`, `next visit`, `recently` | keep |
| duration | 897 (573 / 324) | 192 | `in two weeks`, `for a full week`, `about six months`, `the last 4 to 5 weeks` | keep (acute vs chronic) |
| frequency | 1,389 | 196 | `daily`, `twice a day`, `3 times a week`, `q.6 h.` | keep |
| exact age | 367 (129 / 238) | 174 | `59-year-old`, `57 year old`, `9-month-old`, `3 -year-old`, `100-year-old`, `at age 40` (relative) | bucket to 5 years for adults; **do not bucket < 2 years** (`9-month-old` drives pediatric management but no code here is age-specific below 18), 90+ → "90 or older" |
| age bucket / life stage | 32 | 17 | `elderly`, `in his 40s`, `when i was younger`, `as a child`, `newborn`, `postmenopausal` | keep |
| pregnancy timing | **0** | 0 | – | no obstetric encounter in the corpus |
| era anchor | 12 | 10 | `during the pandemic`, `with covid`, `during smoke season`, `the fires` (`D2N138`) | `pandemic` is harmless; `smoke season`/`the fires` is a geographic hint |

### 6.2 Codes in the data whose correctness depends on demographics or numbers [regex over descriptions, verified on examples]

| dependency | codes (slots) | examples | where the information lives |
|---|---|---|---|
| **laterality** | 64 codes (85 slots) | `M17.11` right knee, `M25.512` left shoulder, `S63.501A` right wrist, `H35.3211` right eye, `M25.572` left ankle | `right/left/bilateral` 1,759× in dialogue, 1,023× in note (CHIEF COMPLAINT, HPI, EXAM, ASSESSMENT); contradictions exist (`D2N092` EXAM says left, code says right; `D2N083` `left wrist` slip) |
| **encounter type** (`initial`/`subsequent` 7th character) | 42 codes (51 slots) | `S33.5XXA`, `S13.9XXD` (subsequent), `T78.01XA` | cues: `new patient`, `this happened yesterday` vs `follow-up`, `since the surgery`; 109 initial-cues and 115 follow-up cues in dialogues; relative dates carry this, absolute dates do not |
| with/without modifier | 14 codes (79 slots) | `E11.9` vs `E11.65` (hyperglycemia) vs `E11.621` (foot ulcer) vs `E11.22` (CKD); `K21.9` without esophagitis; `I25.10` without angina; `I11.0` with heart failure | numbers (A1c, creatinine/eGFR) and findings (`foot ulcer`, `heart failure`, `angina`) |
| severity / acuity / stage | 23 codes (32 slots) | `J45.21` mild intermittent with exacerbation vs `J45.40` moderate persistent, `I50.21` acute systolic, `N18.30` CKD stage 3, `G31.84` mild cognitive impairment | durations and frequencies (`three to four days per week`, `act score 21`), lab thresholds (`eGFR`), timing words (`acute`, `chronic`, `newly`) |
| diabetes type | 6 codes (50 slots) | `E10.9` (`D2N093`) vs `E11.*` | `type two` / `type 2` (139 stage/type numbers), `insulin since childhood` |
| **age-dependent** | 4 codes | `M81.0` age-related osteoporosis (54-year-old), `H35.32xx` age-related macular degeneration, `I49.3` (100-year-old) | HPI first sentence (138 notes), ASSESSMENT AND PLAN (37); the *exact* age is never needed — "50s"/"elderly" suffices for every code here |
| **sex-dependent** | 5 codes | `N62` breast hypertrophy, `N60.01/02` breast cyst, `D05.1` DCIS, `N50.89` male genital | `female/male` in 152/207 notes (HPI 140); 54 notes only via pronouns `she/her/he/his`; dialogue via `mrs/ms/mr`, `she's a 57 year old` |
| lab / vital threshold | 7 codes (10 slots) | `E11.65` hyperglycemia, `N18.30`, `D50.9`/`D64.9` anemia, `R73.9` | A1c, creatinine, hemoglobin values in RESULTS and dialogue |
| BMI threshold | 0 codes | (`bmi greater than 27/30` discussed in `D2N021` for weight-loss drugs) | – |
| **external cause / activity / place** (ICD-10 chapter 20) | 23 codes (24 slots) | `Y93.53 Activity, golf`, `Y93.54 bowling`, `Y93.68 volleyball (beach)`, `Y93.02 running`, `Y93.E9`, `W00.9XXA fall due to ice and snow`, `V89.2XXA motor-vehicle accident`, `W21.09XA struck by thrown ball`, `Y92.009 place of occurrence` | **the hobby/event/place quasi-identifiers in SOCIAL HISTORY/HPI are literally the code**: `i was playing tennis`, `skiing in vermont`, `on my boat skiing`, `car accident in february`. Removing the activity removes a code (24 slots ≈ 5% of all); generalizing `vermont` is fine, generalizing `skiing` is not |
| substance use (sensitive) | 3 | `F17.200` nicotine dependence, `F10.90` alcohol use, `Z79.899` | SOCIAL HISTORY (`smokes 1 pack per day`, `four to five drinks`) |
| mental health (sensitive) | F32.A ×17 + `G47.00`, `G31.84` | depression is the 3rd most frequent code | HPI/A&P (`Regarding her depression`) |
| HIV / hepatitis (sensitive) | 2 + `D2N023` (`hiv positive gentleman`, hypogonadism) | `B19.20`, `B17.1` | HPI, SOCIAL HISTORY (`iv drug use`) |
| unspecified (no laterality/severity needed) | 71 codes (131 slots) | `I10`, `E11.9`, `F32.A`, `K21.9`, `I50.9`, `M06.9` | – |

- Conclusion: exact ages and calendar dates can be generalized with **zero effect** on every code in the data (no code is pediatric-age-specific, no obstetric code; `age-related` codes need only "older adult"). Durations, frequencies, laterality, sex, `type`, `stage`, `acute/chronic`, activity/external-cause words and all clinical numbers must survive. The only temporal expressions with privacy value are DOB, DOS, procedure dates and holiday/era anchors; the rest (2,500 relative expressions) are both harmless and useful.

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

## 9. Typos and noise

Script: `scripts/eda/09_typos.py` (dictionary = macOS `web2` + corpus drug/brand list). **[regex]** counts, **[verified]** classifications.

### 9.1 Misspellings and ASR errors

- Dialogue: 671 non-dictionary word types (6,945 tokens); 350 of them never occur in any note (ASR-garble candidates). Most are colloquial (`kinda` 173, `hmmm` 50, `gotcha`, `mkay`, `meds`, `chemo`, `satting`, `carbs`) and harmless. True ASR/transcription errors found **[verified]**: `hey document` ×3 (= *doc*, `D2N058`), `dawn knots legs` (= *Don Knotts*, `D2N058`), `the rex center` (= *rec center*, `D2N058`), `my sense to close up` (= *stents*), `positive long ... lyme titer` (`D2N134`), `tumor guy` (shoe brand, `D2N114`), `distar- distal` (`D2N094`), `acro acromioclavicular` (`D2N116`), `echocardiocardiogram` (`D2N058`), `eyebugers` (`D2N106`), `metro cream` (= *MetroGel*, `D2N145`), `the venti` (Starbucks size, `D2N158`), `infiltrating lymphocytes , breast` (= *brisk*, `D2N029`), `central a melanotic` (= *amelanotic*), `aveno` (= *Aveeno*, `D2N099`), `jaqueline` (= *Jacqueline*, `D2N188`), `mrnr472348` (MRN glued, `D2N029`), `hypolipidemia`/`hyperthyroidism` for hyperlipidemia/hypothyroidism (`D2N105`), `thanks for coming in to them` (`D2N123`), `i wan na just continue the motrin at 800mg`.
- Notes: 422 non-dictionary types, nearly all valid medical vocabulary (`denies` 502, `musculoskeletal` 182, `rhonchi` 45, `thyromegaly`, `tendinopathy`). Real typos **[verified]**: `coughor` (missing space, `D2N093`), `ff this` (`D2N063`), `is is` (`D2N200`), `Welixa` for Wixela (`D2N138`), `CV: Normal head rhythm` (`D2N114`), `Infiltrating lymphocytes: breast.` (`D2N029`), `Normal empty tan` → `Normal empty can test` (`D2N098`), `she hitting her arm` (`D2N092`), `it is seems` (`D2N131`), `arter stent` (`D2N047`), `regiment` for regimen, `mammaplasty`/`mammoplasty` both used (`D2N024`), `condromalacia` (`D2N032`).

### 9.2 Inconsistent abbreviations (same concept, several forms) [regex]

| concept | forms (dialogue / note counts) |
|---|---|
| HbA1c | `hemoglobin a1c` 64/72, `a1c` 30/9, `hba1c` 0/1, spelled values `seven . four`, `elevated at eight` |
| x-ray | `x-ray` 185/80, `x-rays` 24/51, never `xray` |
| ECG | `ekg` 33/17, `ecg` 8/3, `electrocardiogram` 0/7, `ekgs` 2 |
| blood pressure | `blood pressure` 334/172, `pressure` 32/5, `bp` 1/12, spoken `one twenty over fifty` |
| twice daily | `twice a day` 89/48, `twice daily` 14/44, `twice per day` 6/5, `2 times daily`, `two times per day`, `b.i.d.` 0 |
| milligrams | `milligrams` 269 (dialogue) vs `mg` 378 (note); `800mg` glued 2 |
| follow-up | `follow up` 49/137, `follow-up` 44/80, `followup` 10/2, `recheck`/`re-check` 14 |
| PCP | `pcp` 33/7, `primary care` 26/34, `family doctor`, `primary doctor` |
| ultrasound / echo | `ultrasound` 40/32, `echocardiogram` 51/43, `echo` 3/1, `us` 66 (mostly the pronoun) |
| shortness of breath | `shortness of breath` 103/110, `short of breath` 29/3, `dyspnea` 0/39, `winded` |
| COVID | `covid` 38/26, `covid-19`, `coronavirus`, `the pandemic` |

### 9.3 Casing and punctuation errors inside notes [regex]

- Lowercase sentence starts 14 (`. his endo…`, `. she states`), double punctuation 4 (`breath..`), space before comma 1, double word 1, stray characters 2 (`​`, `ß`), 9 long prose lines without a final period, `mmHg` glued 20 (legitimate), lowercase `i` as pronoun 0, `Cranial nerves I through XII` (Roman numerals).
- Note sentences beginning with a lowercase token are a trap for a sentence-start heuristic ("a capitalized token not at sentence start is a name"): `email, or call if symptoms worsen` (`D2N098`).

### 9.4 Broken turns and bracketed artifacts in dialogue [regex]

- 149 stutter fragments (`w- we`), 109 turns cut off with a trailing `-` (`since this episode of your-`), 24 ending in `...`, 64 turns starting with punctuation (`. refer you to psychotherapy`), 869 one-word turns (`okay` 306, `no` 111, `yeah` 99, `yes` 71, `mm-hmm` 68), 17 empty lines, 1 bracketed artifact (`[ inaudible 00:09:25 ]`), 14 lines without a speaker tag, ≥6 encounters with swapped tags (§3.2).
- A name can sit next to a fragment: `hi- / nice to meet you- / doctor gutierrez . how are you ?` (`D2N018`), `front- . so when you touch that- hi , michelle` (`D2N134`), `an- another`.

### 9.5 Which noise is harmless and which causes PII false negatives [verified]

| noise | effect on PII detection |
|---|---|
| fillers, one-word turns, stutters, `...` | harmless (no PII inside), but stutters next to names break `\b` anchoring (`t- , for your second problem`) |
| name typo `jaqueline` vs `Jacqueline` | **FN** for gazetteer and for cross-field linking; needs fuzzy match (edit distance ≤ 2) between dialogue first name and note full name |
| glued MRN `mrnr472348`, dashed `3-8-4-9-2-0`, alphanumeric `abc123`, zero-padded `0000123` | **FN** for `\d{5,}` or `mrn \d+` rules; rule must be `mrn\w*` + any token up to the next ` . ` |
| spelled numbers in `aci` (`fifty nine`, `twenty eighteen`, `seven . four`) | **FN** for every digit-based age/date/BP rule in 54% of dialogues; need a number-word normaliser (phase 5 helper) |
| split dot in titles (`mr . james`) and in decimals (`113322 . 44-year-old`) | FN for `Mr\.` rules; the `. ` also makes sentence splitters cut inside names |
| swapped speaker tags | wrong role attribution (patient vs doctor name); harmless if every person name is treated the same |
| dictation commands (`please use the regular template`, `end of dictation`, `auto text uri`) | harmless, but a model may learn them as section cues |
| ASR homophones (`document` for doc, `dawn knots`) | `dawn knots` is a *celebrity name* that no gazetteer/NER will flag in lowercase — low risk (simile), unfixable by rules |
| non-ASCII spaces (` `) | FN for `\s`-based tokenizers; normalize to ASCII space (1:1, keeps offsets) |
| `[ inaudible 00:09:25 ]` | the time stamp matches a clock-time rule → harmless false positive |

## 10. Re-identification risk

Script: `scripts/eda/10_reid.py` → `outputs/eda/reid_profiles.json`. Quasi-identifier tokens come from the phase-3 regexes (hobbies/occupations/locations/teams/pets/relatives/brands/staff), so counts are approximate **[regex]**; k is computed inside the 207-encounter corpus (the organizers' test set and the real population are unknown).

### 10.1 k-anonymity on attribute combinations

| attributes available to the attacker | unique encounters | k ≤ 2 | k ≤ 5 | median k |
|---|---|---|---|---|
| sex | 1 | 1 | 1 | 104 |
| age decade + sex | 6 (3%) | 14 | 25 | 20 |
| age 5-year bucket + sex | 11 (5%) | 21 | 38 | 12 |
| exact age + sex | **50 (24%)** | 74 | 140 | 3 |
| code set alone (preserved by design) | **167 (80%)** | 179 | 207 | 1 |
| only the 6 most frequent codes | 5 | 7 | 10 | 127 |
| age decade + sex + code set | 201 (97%) | 207 | 207 | 1 |
| age 5y + sex + code set | 205 (99%) | 207 | 207 | 1 |
| exact age + sex + code set | 207 (100%) | | | 1 |
| + occupation, hobbies | 205–207 | | | 1 |
| sex + codes + occupation + hobbies, **no age** | 199 (96%) | | | 1 |

- The `icd10_codes` field, which must be preserved byte-for-byte, already makes 80% of encounters unique on its own and 97% together with decade + sex. **Within this corpus, k-anonymity is unattainable because of the labels**; the privacy metric can therefore only be about what the *text* reveals beyond the codes: direct identifiers (names, DOB, MRN, dates, staff, employers, places) and rare narrative details.
- Exact age alone is a strong quasi-identifier (24% unique with sex); the 5-year bucket reduces that to 5%, the decade to 3%. Ages ≥ 90 (2) and < 5 (2) are unique at any bucket → use open-ended buckets (`90+`, `under 5`).

### 10.2 Rare details

- 140/207 encounters contain ≥1 detail token that occurs in no other encounter (median 1; `virtscribe` 2.9, `aci` 1.9, `virtassist` 0.8 on average). Corpus-unique tokens by category: hobby 160, staff/doctor names 53, locations 46, employers/occupations 45, brands 30, pets 11, named relatives 9, teams 7, events 5, vehicles 5.
- Most identifying bundles **[verified]**: `D2N138` (army enlistment spring 2022 + Enterprise rent-a-car + Kaiser Permanente Broadway + peanut anaphylaxis + wildfire smoke), `D2N076` (twin infant with Smith-Magenis syndrome, family lives in Dallas, Georgia), `D2N185` (14-year-old with renal hypoplasia, softball, Jacksonville/Atlanta, father had a kidney transplant, relative died on dialysis in his 40s), `D2N157` (San Francisco 49ers fan from the west coast, lives with his dog, retiring), `D2N189` (Denver Nuggets fan, swimming), `D2N154` (triathlon + hockey + Red Wings + Pittsburgh), `D2N074` (computer programmer, married 25 years, pituitary tumour), `D2N029` (owns an auto repair shop, melanoma, mother and aunt with melanoma), `D2N058` (insurance through "the plant", photography, gardening, trails behind the rec center, EF 35%), `D2N105` (100-year-old, walker, cul-de-sac, son had an MI).
- Details that are *also* the code evidence (see §7): `golf`, `bowling`, `soccer`, `skiing`, `running`, `car accident`, `sidewalk`, `snow` — cannot be removed in the 24 external-cause slots without losing the code; they are low-cardinality words (k ≥ 3 for most sports) and only become identifying in combination with a place or team.

### 10.3 Attacker model (feeds the attack bonus)

An attacker holding the original encounter and the anonymized output can link them by:
1. **Codes + sex + age bucket** — always available; 97% unique. This is why the attack bonus should be scored on *text-only* features, with codes removed from both sides, or it is trivially won.
2. **Structural fingerprint** — number of turns, note section sequence, boilerplate variant, clinical numbers (A1c 7.4, creatinine 0.7, EF 35%, weights, doses). Text similarity (TF-IDF over the clinical content) will re-link ~100% of pairs even after perfect PII removal, because the clinical content *is* kept by design. A meaningful attack must therefore try to recover the *removed* values (name, DOB, employer) — e.g. by asking an LLM to guess the name from context — rather than to match encounters.
3. **Surviving quasi-identifiers** — the 706 corpus-unique tokens above; a name/date-only anonymizer leaves all of them. The adversarial evaluation should count how many of these survive (dictionary-based) and whether an LLM can produce the employer/city/team from the anonymized text.
4. **Cross-field leaks** — if the dialogue is anonymized but the note still carries `Mrs. Hernandez`, or the note is anonymized but the dialogue still has `date of birth , march 26 2001`, one field re-identifies the other. Every entity must be mapped jointly per `encounter_id`.

## 11. Train vs eval shift

Script: `scripts/eda/11_shift.py` → `outputs/eda/train_eval_shift.json`. **[regex]**.

| dimension | train (140) | eval (67) | shift / implication |
|---|---|---|---|
| dataset mix | aci 77 / virtassist 35 / virtscribe 28 (55/25/20%) | 35 / 20 / 12 (52/30/18%) | same mix; the hidden test set is presumably drawn the same way |
| dialogue words (median) | 1240 | 1240 | none (eval virtscribe is longer: 1821 vs 1332) |
| note words (median) | 420 | 404 | none |
| codes per encounter | 2.04 | 2.28 | slightly more codes in eval |
| distinct codes | 146 | 112 | **79 eval codes unseen in train (82/153 slots, 49/67 encounters)**; 108 train codes are singletons |
| ICD chapters | no V/W/X/Y at all | V 1, W 11, X 2, Y 10 (24 slots) | external-cause / activity / place codes only in eval → the activity/place vocabulary is utility-critical in eval but invisible in train |
| top codes | I10 37, E11.9 29, F32.A 12 | E11.9 12, I10 7, I50.9 5, F32.A 5, N20.0 5 | hypertension less dominant in eval |
| patient-name density (per 1k words / % encounters) | greeting 0.46 / 74%, intro 1.57 / 84%, title 1.32 / 41% | 0.45 / 70%, 1.53 / 84%, 1.27 / 40% | identical |
| doctor/staff names | 0.40 / 22% | 0.39 / 22% | identical |
| DOB | 0.10 / 17% | 0.08 / 13% | all virtscribe in both |
| exact age | 0.99 / 84% | 0.89 / 79% | identical |
| absolute dates | 0.99 / 24% | **1.67** / 24% | eval virtscribe is date-dense (6.4 per 1k words) |
| locations | 0.35 / 25% | 0.22 / 15% | fewer in eval |
| employer / occupation / school | 0.36 / 24% | **0.49 / 40%** | more occupations in eval (aci 0.73) |
| hobbies / sports | 1.44 / 56% | 1.59 / 58% | similar; eval aci 2.47 (sports injuries → external-cause codes) |
| MRN / insurance | 0.21 / 11% | 0.12 / 7% | MRN in 9 train vs 2 eval encounters |
| named relatives / friends | 0.04 / 4% | 0.03 / 3% | rare in both |
| product brands (dax/dragon) | 1.17 / 39% | 1.27 / 46% | virtassist only |
| note headers | 38 distinct | 29 distinct | eval-only: `COPD.`, `HIV`, `PCOS.` (false headers); train-only: `BIRTH HISTORY`, `SUBJECTIVE`, `ORDERS`, `OTC`, `REVIEW OF SYMPTOMS`, `PAST SURGICAL HISTORY` |
| `[patient_guest]` | 5 encounters | **0** | pediatric/caregiver dialogues untested by eval; the hidden test set may contain them |
| untagged lines / swapped tags | 7 / 3 | 5 / 3 | present in both |
| aci dialogues without punctuation | 77/77 | 35/35 | universal |
| notes without any name | 37 | 21 | same share |
| non-ASCII characters | 617 | 298 | same rate |

- What this implies for the hidden test set: formats, PII densities and dialects are stable across the public splits, so rules tuned on train transfer. The *labels* do not: 70% of eval codes are unseen, singletons dominate, and a whole ICD chapter appears only in eval. Any local utility proxy that fits on train codes is invalid; use code-description similarity, evidence retention, or a frozen general-purpose ICD model. For privacy, the eval split is a faithful proxy. The 5 `[patient_guest]` encounters and the dictated-header encounters (36) are the formats most likely to be *under*-represented in eval and should be covered by the unit tests regardless.

## Trap catalogue

| trap | example (`encounter_id`) | why it fools a model | detection strategy |
|---|---|---|---|
| bare vocative, no punctuation (`aci`) | `alright scott so so i understand` (`D2N150`), `okay raymond it looks like` (`D2N062`) | no comma, no capital, no title; `scott` is also a surname | **both**: first-name gazetteer restricted to lines 0–2 and to `so|alright|okay|hey <w> (so|i|it's|…)` frames, confirmed by a lowercase-trained token classifier |
| swapped speaker tags | `[patient] good morning Julie how are you` (`D2N114`), `[patient] hi kyle` (`D2N082`) | role heuristics assign the name to the doctor | **rule**: treat all person names symmetrically; detect swap by doctor-cue count per speaker |
| one-word surname turn | `[doctor] howard` after `how do you say your last name` (`D2N190`) | no context in the line itself | **model** (dialogue-level context) + cross-field link (`Howard` in note) |
| title + surname that is a first name | `mr . james` (`D2N098`), `Ms. James` (`D2N094`), `miss taylor` (`D2N142`) | gazetteers tag `james` as first name | **rule**: token after `mr/mrs/ms/miss/mister` = surname |
| title that is not a title | `mr arthrogram` (`D2N032`), `let's use my default pe` (`D2N102`) | regex `mr \w+` | **rule**: imaging/exam stop-list after titles |
| doctor by first name | `doctor isabella` (`D2N098`), `dr. evelyn` (`D2N076`), `doctor michael` (`D2N124`) | surname-only doctor rules miss it | **rule** `dr/doctor + any name` with verb stop-list |
| staff first name without title | `with madison` (`D2N105`), `have helen come in` (`D2N138`) | nothing marks it; `madison` is a place | **model** + verb frames (`have X come in`, `went through with X`) |
| nickname / pet name | `dino , you're doing so good` (`D2N099`) | not in any lexicon | **model** (vocative position in `[patient_guest]` turn) |
| celebrity simile | `dawn knots legs` (`D2N058`) | lowercase ASR garble of a public figure | flag only (not patient PII) |
| name = common/clinical word | `white blood cell` vs `Dr. White`; `requires a walker` vs `Ms. Walker`; `green phlegm` vs `Tyler Green`; `frank obstruction`; `hey there mark` | same token, two senses | **both**: cue rules (title before, noun after) + classifier |
| eponym | `lyme titer`, `cushing's`, `thompson test`, `murphy's sign`, `smith magenis syndrome`, `allen test` | surname lexicon contains them | **rule**: eponym whitelist + `'s|test|sign|syndrome|disease|fracture|titer` suffix |
| product as person | `tell dax a little bit about you`, `hey , dragon , order lisinopril` (280×) | greeting frame + name slot | **rule**: `dax`/`dragon` stop-list |
| name typo across fields | `jaqueline` vs `Jacqueline Price` (`D2N188`) | exact linking fails | **rule**: edit-distance ≤2 link between dialogue first name and note full name |
| bare first name later in the text | `diana has a 4/6 systolic murmur` (`D2N105`), `Ashley` ×3 (`D2N185`), `bye randy` | "first lines only" strategies | **rule**: propagate every known patient name string through both fields |
| dictated header bundle | `next patient is melissa sanchez . date of birth , 9/23/1962 . mrn : 5484367` (`D2N103`) | 3 entity types in one sentence, split dots | **rule**: header grammar on line 0 of `virtscribe` |
| DOB formats | `9/29/1957`, `4/28/07`, `8-23-97`, `february 15th , 1979`, `march 26 2001`, `october 31st 1949`, `n- date of birth 4/14/1960` | 6 formats, stutter fragments | **rule**: `date of birth|birth date|dob` + any date grammar within 40 chars |
| MRN formats | `mrnr472348`, `mrn 3-8-4-9-2-0`, `mrn : abc123`, `mrn 0000123`, `mrn49282721`, `medical record number 113322 . 44-year-old` | glued, dashed, alphanumeric, zero-padded, followed by a split period | **rule**: `mrn\w*|medical record number` + consume the next token(s) up to ` . ` |
| age without marker | `you are 100` (`D2N105`), `i know that you are 74` (`D2N199`) | no `year-old` | **both**: `you are|i'm|he's NN` rule bounded 1–110 + classifier |
| age in words / space-hyphen | `fifty nine`, `56 -year-old`, `57 year old`, `9-month-old`, `3 -year-old` | digit regex, hyphen regex | **rule** on a number-word-normalized shadow string |
| computed age | note `100-year-old`, `9-month-old` from DOB + DOS | not in dialogue, >89 | **rule**: age bucket with `90+` and `<2 y` exceptions |
| date inside all-caps line | `ECG, 12/26/2020.`, `CMP, 03/22/2021.` (`D2N103`, `D2N185`) | header detector skips the line | **rule**: run date rules on every line; false-header filter (digits/commas) |
| clinical slash numbers | `4/6 murmur`, `6/10 pain`, `4/5 strength`, `18-20/28`, `250/50 dose` | look like dates / BP | **rule**: score grammar (`/10 /6 /5 /28`, `out of`, `grade`) before date grammar |
| spoken decimals | `seven . four` (A1c), `twelve . five milligrams`, `ninety nine point seven` | split dot = sentence boundary | **rule**: number-word normalizer |
| seasons vs words | `had a bit of a fall` (`D2N090`), `back summer salt` (`D2N067`), `in the fall` | `fall`/`summer` ambiguity | **rule**: season only after `in the|this|last|every|of` |
| weekday vs brand | `friday's restaurant` (`D2N140`) | weekday lexicon | **rule**: possessive + `restaurant` |
| pet vs CT | `a cat scan` ×5 | pet lexicon | **rule**: `cat scan` stop |
| team = common noun | `chiefs`, `bucks` (dollars), `red wings`, `nuggets`, `heat` | lexicon false positives | **both**: team list + `fan|game|play|watch` cue |
| hobby = injury mechanism / code | `playing tennis` (S63.501A), `golfing` (Y93.53), `skiing in vermont` (Y93.23), `car accident` (V89) | removing it removes a code | **policy**: keep activity nouns, generalize venue/place/team/brand |
| employer named as brand/place | `enterprise rent-a-car`, `kaiser permanente broadway`, `through the plant`, `working in a bakery` | brand lexicon ≠ employer | **model** (phrase spans) + `work(s) at/for/as` frames |
| insurer | `blue cross`, `kaiser`, `medicare`, `the va` | not names, not places | **rule**: insurer list |
| geography without a place name | `smoke season`, `the fires` (`D2N138`), `detroit is a hockey town` | no gazetteer hit | **model** / flag |
| doctor biography | `that was my first fall here` (`D2N138`) | speaker ≠ patient | **model**; flag first-person doctor statements with dates |
| sensitive + code-bearing | `depression` (F32.A), `hiv positive gentleman` (`D2N023`), `iv drug use` (B19.20), `alcohol use disorder` (F10.90) | PII detectors over-remove | **policy**: keep diagnosis + quantity, strip narrative |
| relatives' diagnoses | `Mother: melanoma.`, `twin sister … autism`, `nephew … sinus infection` | third-party health data | **policy**: keep relation + disease, remove names/ages |
| contradictions | `left wrist` vs `right wrist` (`D2N092`), `hyperthyroidism` vs E03.9 (`D2N105`), `lymphocytes: breast` (`D2N029`) | an LLM "fixes" them → changes clinical content | **policy**: never edit non-PII spans |
| Unicode spaces / stray chars | `asymmetrical breasts` (`D2N101`), `extremity.ß` (`D2N010`), `​His` (`D2N174`), `Neer’s` | `\b`, `\s`, `'s` rules break; deletions shift offsets | **rule**: 1:1 normalization map + index array for deletions |
| untagged / cut turns | first line without tag (`D2N131`), `hi- / nice to meet you- / doctor gutierrez` (`D2N018`), `[ inaudible 00:09:25 ]` | parsers drop or merge lines | **rule**: tagless line = continuation; never merge lines |

## Decisions

### a. Neural vs neuro-symbolic → **neuro-symbolic (rules first, model second, LLM last)**

- **Rules with ~100% precision on this corpus** (phase 3 hand-check): DOB/DOS phrases (33, all in dialogue line 0), MRN/record numbers (11, 5 formats), absolute date grammars (6 formats, 19/20 precision), explicit ages (`NN-year-old`, `NN year old`, `NN -year-old`, `NN-month-old`, spelled; 20/20), titles + name (`mr/mrs/ms/miss/mister/dr/doctor` with split/unsplit dot; 19/20 with the `arthrogram` stop), the `virtscribe` dictated-header grammar, note HPI/A&P intro (`First Last is a NN-year-old`; 20/20), `Dr. Last` in notes, insurer list, `dax`/`dragon`/eponym/`cat scan`/`x-ray` stop-lists, speaker-tag and section parsing, Unicode normalization, propagation of every found patient-name string through both fields. These cover all DOB/MRN/DOS, ~95% of absolute dates, ~95% of ages, ~90% of patient-name *encounters* and ~85% of patient-name *mentions*.
- **Model needed** (rules < 70% precision or < 80% recall): bare vocatives in `aci` (17 encounters), staff/relative/friend first names and nicknames (~10% of encounters), bare first-name repeats in prose, locations without a gazetteer hit (`the rex center`, `main street`, `district regional hospital`), employers/occupations/schools (137 hits, ~65% rule precision), hobbies/events/teams as *spans* (457 hits, ~70%), third-party biography, ASR-damaged names (`jaqueline`). A small token classifier trained on lowercase, PTB-tokenized text (so that `aci`/`virtassist` input matches training) with the weak labels of (c); GLiNER-style zero-shot as the fallback for span-level categories with too few examples (locations, employers, events).
- **Both** (rule proposes, model/context confirms): patient first name in line 0–2 (gazetteer + frames + classifier), dates in prose without a year (`april 10th`, `last december`, `the 24th`), occupations, colliding names (`white`, `young`, `mark`, `price`), `twin`/`adopted`/`veteran` social flags.
- **LLM (≤3B, local) only for**: generalizing free-text quasi-identifiers into clinically equivalent phrases (`enterprise rent-a-car` → `a rental-car company`, `barre classes` → `exercise classes`) and for the attack bonus (guess removed values). Never for detection of names/dates/IDs, never allowed to touch non-flagged spans (contradictions must survive).
- Trade-off: rules give determinism and explainability for 80% of the privacy mass at zero VRAM; the model adds ~10 points of name recall and the quasi-identifier classes at the cost of label noise. Pure neural would lose the exact-format IDs/dates (rare, 11 MRNs) and would over-fire on drugs/eponyms in notes; pure rules would miss ~8% of patient names and most quasi-identifiers.

### b. Pre-processing script → **yes, minimal, offset-preserving, no detokenization/truecasing for the model**

1. **Unicode map, 1:1**: ` `/` ` → space, `’`→`'`, `“”`→`"`, `²`→`2`, `–`→`-`; deletions (`​`, stray `ß`) are recorded in an index array `norm_idx → orig_idx` so every span maps back to the original string. Validate with `assert orig[map[s]:map[e]]` round-trips.
2. **Speaker-tag parser**: `^\[(doctor|patient|patient_guest)\]\s?`; a line without a tag is a continuation of the previous speaker (or `doctor` on line 0); unknown bracket tags (`[ inaudible 00:09:25 ]`) are text; never merge or drop lines. Compute a "swapped-tags" score (doctor cues per speaker) and expose it; the anonymizer treats all human names identically so a swap cannot cause a miss.
3. **Note sectioner**: all-caps line (≥3 letters) = header, with a false-header filter (contains digits/commas/period-only endings like `GERD.`, `BSA 2.65`, `ECG, 12/26/2020.`) and a synonym map (`PHYSICAL EXAMINATION`→`PHYSICAL EXAM`, `HPI`/`CC:` short forms, `REVIEW OF SYMPTOMS`, `PAST HISTORY`); rules still run on header lines.
4. **Number-word shadow**: `spans()` from `scripts/eda/numwords.py` produces digit values and offsets for `fifty nine`, `one twenty over fifty`, `seven . four`, `twenty eighteen`; age/date/BP rules run on the shadow but **the text is never rewritten**.
5. **Light detokenization only for an off-the-shelf cased NER baseline** (`do n't`→`don't`, `mr .`→`mr.`, `, `→`,`), tracked by the same index map; it is not applied to the fine-tuned model, whose training data is lowercased and PTB-tokenized instead (train = inference distribution). **No truecasing**: it would capitalize `may`, `will`, `grace`, `ray` inconsistently and cannot help `aci` (no sentence boundaries).
- What it would break if done naively: any deletion or merge shifts offsets (the decisions log and the final replacement need original offsets); detokenizing `113322 . 44-year-old` into `113322.44` destroys the MRN/age boundary; truecasing `hi mr . james` → `Hi Mr. James` makes `James` look like a first name.

### c. Fine-tuning plan

- **Label schema** (BIO, token level, applied to both fields): `PATIENT_NAME` (first/last merged; role resolved later), `OTHER_PERSON` (doctor, staff, relative, friend, nickname, celebrity), `AGE`, `DOB`, `DATE` (absolute, incl. month+day and year-only when anchored to the patient), `ID` (MRN, insurance member), `LOCATION` (city/state/street/facility/landmark), `EMPLOYER_ORG` (employer, school, insurer, pharmacy chain), `OCCUPATION`, `ACTIVITY` (hobby/sport/team/event — kept or generalized by policy, labelled anyway), `PRODUCT` (brand, scribe products), `SENSITIVE` (optional flag label for mental health/substance/HIV/legal narrative, used for the policy layer, never for removal by itself). Relative dates, durations, clinical numbers and eponyms are explicit negatives.
- **Weak labeling**: (1) the phase-3/4 rules produce silver spans on all 207 encounters (the eval *text* is a legitimate unlabeled input; its codes are not used); (2) names found anywhere are propagated per encounter (`andrew` ↔ `Andrew Campbell` ↔ `Mr. Campbell` ↔ `Campbell`), fuzzy for typos; (3) a first pass of the classifier re-labels, disagreements go to a manual review queue; (4) manual review of ≥30 encounters (the 12 already read + 18 stratified) and of all 167 `hard_cases.jsonl` spans, which are also the regression test suite.
- **Expected label counts** (from phase 3/4, both fields): `PATIENT_NAME` ≈ 1,100 mentions (≈ 900 dialogue, ≈ 470 note mentions incl. titles), `OTHER_PERSON` ≈ 170, `AGE` ≈ 370, `DOB` 33, `DATE` ≈ 230, `ID` 11 (+ ~10 insurer mentions), `LOCATION` ≈ 110, `EMPLOYER_ORG`/`OCCUPATION` ≈ 140, `ACTIVITY` ≈ 450, `PRODUCT` ≈ 430 (350 are `dax`/`dragon`), `SENSITIVE` ≈ 300.
- **Are 140 encounters enough?** Yes for names, ages, dates, titles (hundreds of highly regular examples, and the rules already cover them); borderline for `OTHER_PERSON` (≈170, varied); **no** for `LOCATION`, `EMPLOYER_ORG`, events (tens of examples, open vocabulary) → use gazetteers + zero-shot span model + augmentation, and accept a lower bar there.
- **Augmentation** (deterministic, seeded by `encounter_id`): name swapping from census-style lists applied consistently to both fields and all mention forms (`andrew`/`Andrew Campbell`/`Mr. Campbell`), including names that collide with words (`mark`, `may`, `grace`, `bill`, `white`) and typo variants; date/DOB/MRN format swapping (all 6 date formats, the 5 MRN shapes); age jitter ±5 with every surface form (`NN-year-old`, `NN year old`, `NN -year-old`, spelled); digit ↔ number-word rewriting to make `virtassist`/`virtscribe` look like `aci`; title variants (`mr .`/`mr.`/`mister`/none); inserting a `virtscribe`-style dictated header into `aci`/`virtassist` dialogues and `CC:`/`HPI:` short headers into notes; moving a vocative into a mid-dialogue turn; swapping speaker tags; random `[patient_guest]` turns with a relative's name. Do **not** augment clinical numbers.
- **Evaluation split**: by encounter, stratified by dataset: train 140 → 110 fit / 30 dev (the 12 read encounters go to dev); eval 67 untouched for the final number and never used for model selection; `hard_cases.jsonl` as a must-pass suite; report dialogue and note recall separately and per dataset.

### d. Policies for the ambiguous classes

- **Clinical numbers** (doses, labs, vitals, BP, A1c, EF, scores, grades, stages, measurements, frequencies, durations, counts of cigarettes/drinks): never modified. Extreme fingerprints (weight at 100 years, `1,792 grams`) are kept; they are needed for the codes and k-anonymity is lost anyway through the codes.
- **Ages**: adults → 5-year bucket (`59-year-old` → `55-59-year-old` or `late 50s`), ≥90 → `90 or older`, <2 years → keep months, 2–17 → keep the exact age only if a pediatric code depends on it (none here) else bucket; relatives' ages → decade. Apply identically to the computed ages in notes.
- **Dates**: DOB and date of service → remove (`[DOB]`); other absolute dates → per-encounter random shift (same offset in both fields, ±30–365 days) or year-only when the year suffices for the code (`stent in 2018` → `stent 3 years ago`); relative expressions, durations, frequencies, seasons, weekdays, times of day → keep; named holidays → `a holiday`; era anchors (`pandemic`) → keep, `smoke season`/`the fires` → `a period of poor air quality`.
- **Names**: patient → one consistent surrogate per encounter in both fields and all forms (or `[PATIENT]`), including bare repeats, typos and the swapped-tag cases; doctors/staff/relatives/friends/nicknames → role-typed surrogates (`[DOCTOR]`, `[RELATIVE]`); eponyms and products never touched.
- **Quasi-identifiers**: employers, schools, insurers, hospitals, cities/states/streets, teams, brands → generalize to the class (`a rental-car company`, `a city in the southeast`, `an NBA team`); hobbies/activities → keep the activity noun when it is the injury mechanism or an external-cause candidate (`tennis`, `golf`, `skiing`, `car accident`, `fell on the sidewalk`), generalize otherwise (`barre classes` → `exercise classes`, `pottery` → `repetitive hand work` only if the team accepts the utility risk; default keep); household/pets/twins/adoption/marital duration → generalize (`lives with family`), keep what is clinically functional (`uses a walker`, `lives alone` for discharge planning).
- **Sensitive-but-code-relevant content**: keep every diagnosis term and quantity (`depression`, `alcohol use disorder`, `hepatitis C`, `hiv positive`, `smokes 1 pack per day`); remove the narrative specifics (`hep c partners`, `used to party a lot`, `iv drug use over 15 years ago` → `remote history of IV drug use`), lawyer/court/disability details, and relatives' names/ages in FAMILY HISTORY while keeping relation + disease. Log each such span with reason `sensitive-kept` for the explainability file.
- **Never "repair"** clinical contradictions, typos or ASR errors outside PII spans.

## Open questions (for the organizers or the team)

1. Which categories count as "privacy-related information" in the organizers' metric — only direct identifiers (names, DOB, MRN, dates), or also quasi-identifiers (occupation, hobbies, places, relatives) and sensitive conditions? The answer changes the hobby/activity policy and the eval score by ~5% of code slots.
2. Are consistent surrogate names acceptable, or are placeholders expected? Does a surrogate count as "removed"?
3. Does the privacy metric accept age buckets and shifted dates as removal, or does any residual age/date count as leakage?
4. Do external-cause/activity codes (`W/X/Y`, eval-only) count in the utility F1? If yes, activity words must stay verbatim.
5. Should sensitive-but-coded conditions (depression F32.A, HIV, hepatitis C, alcohol) be treated as PII or as clinical content?
6. Will the hidden test set contain `[patient_guest]` turns, dictated headers, swapped tags and untagged lines in the same proportions (eval has 0 `[patient_guest]`)?
7. May the eval split *text* be used for unsupervised self-training / weak labeling of the detector?
8. Explainability format: is a sidecar `decisions_*.jsonl` enough, or do they expect inline markers in the anonymized text?
9. Team decision: fine-tune a small token classifier (needs a few GPU-hours and the weak-label pipeline) vs. GLiNER zero-shot + rules only, given the 48 h; and which local LLM (if any) for generalization.
10. Team decision: date shift vs year-only; `[PATIENT]` vs fake names; keep or generalize hobbies by default.
