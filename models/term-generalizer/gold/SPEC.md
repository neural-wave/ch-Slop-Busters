# Annotation spec: PII, medical concepts and ICD evidence

Used for the reference annotation of the eval split (`gold/eval/`) and the silver annotation of the train split
(`gold/train/`). Annotators are LLM agents; the files are validated by `ann_tool.py check` and consolidated by
`build_gold.py --split <split>`.

## Tools

```bash
python3 ../term-generalizer/gold/ann_tool.py show  D2N088 --split train   # dialogue, note and icd10_codes of one encounter
python3 ../term-generalizer/gold/ann_tool.py check D2N088 --split train   # validates gold/train/D2N088.json, prints problems
```

## File format: `../term-generalizer/gold/<split>/<encounter_id>.json`

```json
{"encounter_id": "D2N088",
 "pii": [{"text": "andrew", "field": "dialogue", "category": "PATIENT_NAME", "comment": "patient first name"}],
 "medical_concepts": [{"text": "upper respiratory infection", "field": "dialogue", "type": "diagnosis",
                       "icd_relevant": true}],
 "ambiguous_terms": [{"text": "dax", "field": "dialogue", "is_pii": false, "comment": "documentation software"}],
 "evidence": [{"code": "J06.9", "spans": [{"text": "upper respiratory infection", "field": "dialogue"}]}]}
```

- `text` is a **verbatim substring** of that field (copy it exactly, including the lowercase and spaced punctuation
  of the dialogue: `hi , andrew .`). List each distinct string **once per field**; the evaluator counts all its
  mentions. Annotate `dialogue` and `note` separately: the same name in both fields is two entries.
- Keep spans minimal: `andrew`, not `hi , andrew`. Add the longer forms that also occur (`Andrew Perez`, `mr. perez`)
  as separate entries, so every surface form is covered.

## `pii`: what identifies the patient or another person

| category | tier | include | examples from the eval reference |
|---|---|---|---|
| `PATIENT_NAME` | direct | first, last, full name, with title if used | `martha`, `Martha Collins`, `ms. hill` |
| `CLINICIAN_NAME` | direct | doctors, nurses, PAs named in the text | `dr. gutierrez`, `doctor taylor`, `ruth sanchez` |
| `OTHER_PERSON_NAME` | direct | relatives, friends, staff, any other named person (also mis-transcribed celebrities) | `tyler` (cycling buddy), `roy` (scheduler) |
| `DATE` | direct | calendar dates: day + month, month + year, years, DOB, holidays/birthdays as anchors | `01/20/2020`, `two thousand eighteen`, `january 13th , 1982`, `his birthday` |
| `CONTACT` | direct | phone, email, address, URL | |
| `ID` | direct | MRN, account, insurance, device serial numbers | `mrnr472348` |
| `AGE` | quasi | exact ages | `50-year-old`, `age 62` |
| `LOCATION` | quasi | named places: states, cities, venues, parks, streets, regions | `florida`, `northern cascades`, `rex center` |
| `ORGANIZATION` | quasi | named employers, clinics, pharmacies, schools, teams | `cvs pharmacy`, `point may orthopedics`, `the plant` (employer) |
| `OCCUPATION` | quasi | the person's job or role | `landscaper`, `student` |
| `OTHER_QUASI` | quasi | rare personal details that narrow down who it is: unusual hobbies with specifics, family size, veteran status, life events, distinctive routines | `rides his bike for about 20 to 30 miles`, `5 children and multiple grandchildren`, `i was adopted`, `5:45 swim` |

**Not PII:** `dax`, `dragon` and other software; `[doctor]`/`[patient]` tags and generic roles (`my wife`, `the
nurse`); bare weekdays; relative time and durations (`two years ago`, `the last 48 hours`, `3-month follow-up`);
doses, vitals, lab values and pain scores (`40 mg`, `180 over 95`, `7/10`); guideline ages (`over the age of 50`);
gestational ages; diagnoses and other medical content.

## `medical_concepts`: what an ICD-10 coder needs

Every diagnosis, symptom, finding, lab or test, medication (with dose if written together), procedure and anatomy
mention that carries clinical meaning. `type` is one of `diagnosis, symptom, finding, lab_or_test, medication,
procedure, anatomy`. Set `icd_relevant: true` when the concept supports, refines or rules out one of the encounter's
`icd10_codes` (laterality, acuity, severity, cause and complications count).

**External-cause and Z-code context is a medical concept, never PII.** Activities, mechanisms and places of
occurrence that support a V00-Y99 code (`W18.49XA` slipped, `Y93.53` golf, `Y92.xxx` place) or a Z code (`Z57`
occupational exposure, `Z72` lifestyle, `Z91` history) are labelled `type: "finding"`, `icd_relevant: true`, even if
they look personal: `i was golfing`, `fell off the ladder`, `slipped on the ice`, `at work in the warehouse`. If the
same text also names a specific place or employer (`at the walmart warehouse`), put the generic part in
`medical_concepts` (`warehouse`) and the name in `pii` (`walmart`).

## `ambiguous_terms`: strings that could be read either way

Terms where the PII vs. medical decision needs a reason: eponyms (`parkinson's`, `Cushing's`, `Lisfranc fracture`),
surnames that are also diseases or words, place names that look like names (`washington`), brand/software names,
abbreviations (`cvs`). `is_pii` gives the decision, `comment` the reason. When one encounter uses a term in both roles
(`mr. parkinson` and `parkinson's disease`), add two entries with the longer verbatim context as `text`.

## `evidence`: links from each code to its support

One entry per code in the encounter's `icd10_codes`, with the concept spans (verbatim, from either field) that
support it. Every evidence span must also be listed in `medical_concepts` with `icd_relevant: true`. If the text has
no support for a code, give `"spans": []`.

## Process

1. `show` the encounter. Read the dialogue and the note completely.
2. Write the JSON file. Be exhaustive on PII (missing a name is the costly error) and on ICD-relevant concepts.
3. Run `check` and fix every reported problem until it prints `OK`.
