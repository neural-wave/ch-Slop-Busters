# Annotation guidelines (ground truth for self-evaluation / teacher labels)

Each encounter has two free-text fields: `dialogue` (lowercase ASR transcript with [doctor]/[patient] turns)
and `note` (clinical note). Annotate **privacy-related information (PHI)** and, when asked, **medical concepts**.

General rules
- Every `text` value MUST be copied verbatim (same characters, spacing, punctuation; case may differ) from the field
  named in `field`. The validator rejects anything that is not a substring.
- List each distinct surface form **once per field** (do not repeat duplicates; the evaluator finds all occurrences).
- Use the shortest span that carries the identifying content (max ~6 words). Do NOT include titles (mr./ms./dr.)
  in names; do NOT include generic words around a value (for "she is 59 years old" annotate "59").
  Exception for ages: annotate the age expression as it appears ("59-year-old", "59", "fifty nine").
- Do NOT annotate: the ambient-scribe product "dax"/"dragon"; generic relation words alone ("my wife", "son");
  sex/gender; relative time expressions needed clinically ("two weeks ago", "since last year");
  medical eponyms used as disease names ("parkinson's disease", "bell's palsy", "crohn's").
  BUT a person whose surname is an eponym ("mr. parkinson") IS a name.

## PHI types
Tier 1 (direct identifiers):
- `NAME_PATIENT`   patient's first/last/full name in any form (each distinct form separately: "Andrew Campbell", "andrew", "campbell")
- `NAME_RELATIVE`  names of family members, friends, partners, pets' names count as NAME_OTHER
- `NAME_CLINICIAN` names of doctors, nurses, therapists, other staff
- `NAME_OTHER`     any other person/pet name
- `AGE`            exact ages of the patient or other people ("59-year-old", "age 59", "fifty-nine", "3 year old")
- `DATE`           absolute or calendar-anchored dates: full dates, dates of birth, month+year, years ("2019", "june 2020",
                   "3/6", "march 3rd", "9/29/1957"); specific holidays used as dates count as TEMPORAL (tier 2)
- `LOCATION`       cities, states, countries, addresses, streets, zip codes, named venues/landmarks/parks/lakes
- `ORGANIZATION`   employers, companies, schools, universities, hospitals/clinics by name, teams, churches, stores by name
- `CONTACT`        phone/fax numbers, emails, URLs
- `ID`             MRN, SSN, insurance/account/license/device numbers
Tier 2 (quasi-identifiers / rare non-medical context):
- `OCCUPATION`     job, profession, role, workplace type ("teacher", "landscaper", "post office", "construction")
- `ACTIVITY`       hobbies, sports, leisure activities, rare habits ("barre classes", "hunting", "triathlon", "knitting")
- `LIFE_EVENT`     trips/vacations, weddings, moves, births/deaths of others, legal/financial events, military service
- `TEMPORAL`       holidays, named events, weekdays/seasons used to pin down timing ("christmas", "fourth of july", "super bowl")
- `PERSONAL_DETAIL` other rare personal facts that narrow down identity (possessions, pets, nationality, religion,
                   living situation details like "big plot of land", "venison")

Output per item: {"text": ..., "field": "dialogue"|"note", "type": ..., "tier": 1|2}

## Medical concepts (only when the task asks for them)
Types: `DIAGNOSIS`, `SYMPTOM`, `MEDICATION`, `PROCEDURE`, `TEST` (labs, imaging, measurements), `FINDING` (exam/test findings).
Surface strings verbatim, max ~6 words, distinct per field. Include all clinically meaningful concepts (typically 20-60
per encounter). Set "icd_relevant": true if the concept supports or specifies one of the encounter's ICD-10 codes
(the diagnosis itself, its defining symptoms/findings/labs, laterality, severity, complications).
Output per item: {"text": ..., "field": ..., "type": ..., "icd_relevant": true|false}

## File format
Write one JSON file per encounter to the path given in the task:
{"encounter_id": "...", "phi": [...], "medical_concepts": [...]}   (medical_concepts = [] if not requested)
Then run the validator given in the task and fix every reported problem.

## Teacher labels for the training split (extra field)
For the training split, every PHI item additionally carries `"generalized"`: a short, generic, non-identifying phrase that
could replace the text *in its sentence* while keeping the sentence grammatical and keeping any clinically useful meaning
(e.g. "barre classes" -> "exercise classes", "landscaping" -> "yard work", "vermont" -> "another state",
"post office" -> "a public-sector job", "lake tahoe" -> "a lake", "christmas" -> "a holiday", "triathlon" -> "an endurance event").
Use "" for names, ages, dates, contact and IDs (those are handled by rules).
