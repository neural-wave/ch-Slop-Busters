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
