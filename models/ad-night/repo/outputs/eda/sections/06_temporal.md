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
