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
