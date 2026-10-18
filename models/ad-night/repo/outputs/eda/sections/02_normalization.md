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
