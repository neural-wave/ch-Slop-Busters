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
