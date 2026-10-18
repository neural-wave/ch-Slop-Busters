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
