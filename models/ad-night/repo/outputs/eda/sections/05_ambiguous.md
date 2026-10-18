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
