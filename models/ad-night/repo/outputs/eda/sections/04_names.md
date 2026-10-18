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
