# Labeling guidelines (silver privacy labels for the local evaluation harness)

Purpose: give the local harness an exact, independent list of the private information in each
encounter so that `unified-harness evaluate` can measure **the proportion of privacy-related
information removed**, per category and per field, and catch **over-removal** of clinical
look-alikes. These labels are silver: produced at development time (allowed by the challenge rules) and
are never read by the anonymization pipeline.

## Output file

One JSON file per encounter: `labels/upstream/<split>/<encounter_id>.json` (git-ignored)

```json
{
  "encounter_id": "D2N000",
  "pii": [
    {"field": "dialogue", "text": "nora", "category": "PATIENT_NAME", "risk": "high"},
    {"field": "note", "text": "Ellison", "category": "PATIENT_NAME", "risk": "high"},
    {"field": "note", "text": "59-year-old", "category": "AGE", "risk": "high"}
  ],
  "keep": [
    {"field": "dialogue", "text": "lyme titer", "category": "EPONYM", "why": "disease test, not a person"}
  ]
}
```

- `field`: `dialogue` or `note` (list the same string once per field in which it occurs).
- `text`: an **exact substring** of that field (copy it, same spelling, same spaces such as
  `mr . alvarez`). Matching is case-insensitive and whole-word, and **every occurrence** of the string
  in the field is counted, so list each distinct surface string once per field.
- The validator (`scripts/labels/validate.py`) must report 0 errors for the file.

## How to choose `text`

1. **Names: one item per name token**, not per full name: `Nora Ellison` → `Nora` and
   `Ellison`; dialogue `nora` → `nora`. Possessives are covered automatically (`nora's`
   contains the word `nora`). Include nicknames, typo variants (`katharyn`), and names said
   with a title (`mr . alvarez` → item `alvarez`).
2. **Disambiguate collisions.** If the bare token also appears in the same field with a
   non-PII meaning (`white` in `white blood cell`, `walker` the device, `may` the verb, `price`,
   `young`, `cook`, `hill`), use the shortest exact string that only matches the PII use by adding
   the neighbouring word: `doctor white`, `Dr. White`, `ms. walker`. Check with the validator.
3. **Ages**: the age expression as written, including the unit, so that a bare number elsewhere is
   not counted: `59-year-old`, `56 -year-old`, `57 year old`, `fifty nine year old`, `9-month-old`.
   For markerless ages use a 2-word string: `are 74`, `you are 100`, `age 7`.
4. **Dates**: the whole date as written (`7/14/1962`, `june 3 1998`, `october 9th , 1983`,
   `05/11`, `may 11th`, `11/30/2021`). Bare years go in category `YEAR` with the year only if
   unique in the field (`2016`), else with a neighbour word (`in 2016`).
5. **IDs**: the identifier as written (`mrnq815203`, `5-1-7-2-9-4`, `xyz789`, `778899`).
6. Multi-word places/organisations as one phrase (`san francisco`, `enterprise rent-a-car`,
   `kaiser permanente`, `blue cross blue shield`).

## Categories (`category`) and default `risk`

| category | what | risk |
|---|---|---|
| `PATIENT_NAME` | patient's first/last name, nickname, typo variant | high |
| `PERSON_NAME` | any other real person: clinician, staff, relative, friend (by name) | high |
| `DOB` | date of birth | high |
| `ID` | MRN, record/account/insurance member numbers, phone, e-mail, SSN, URL | high |
| `DATE` | absolute calendar date: day+month(+year), month+year, date of service, dated tests | high |
| `AGE` | exact age of the patient or of a named/identified relative (incl. ages ≥ 90, months) | high |
| `YEAR` | bare calendar year (`in 2016`, `twenty sixteen`) | medium |
| `LOCATION` | city, state, country, street, neighbourhood, named landmark/facility, named hospital/clinic | medium |
| `ORGANIZATION` | employer, school, insurer, pharmacy chain, sports team, named company/church | medium |
| `OCCUPATION` | job / profession / employment detail (`auto repair shop` owner, `farmer`, `army`) | medium |
| `OTHER_QI` | other rare personal detail usable for re-identification: specific hobby or event that is **not** clinically needed, named holiday tied to the visit, pet, family structure (`twin`, `adopted`, `married for 25 years`), brands of personal items, celebrity references | low |

Use the default risk unless there is a clear reason (e.g. a very common hobby like `walking` is not
PII at all — leave it out; a public figure simile is `low`).

## What is NOT private information (do not list in `pii`)

- Relative times and durations (`three weeks ago`, `last summer`, `in 2 weeks`), weekdays,
  seasons, times of day, frequencies.
- Clinical content: diagnoses (also sensitive ones like depression, HIV, hepatitis C), symptoms,
  medications, doses, lab values, vitals, scores, laterality, sex, procedures, family history
  diseases (`Mother: melanoma`) — relation words alone (`my wife`, `his son`) are not PII.
- Product/system names that are not about the patient: `dax`, `dragon` (dictation software),
  guideline bodies (`national comprehensive cancer network`), drug brands.
- Eponymous diseases, tests and signs (`lyme`, `cushing's`, `murphy's sign`, `thompson test`,
  `achilles`, `lachman`, `smith magenis syndrome`).
- Generic places (`the hospital`, `the er`, `home`, `work`, `the gym`).

## `keep` list (over-removal traps)

List strings that **look like PII but are clinical** and must survive anonymization, especially:

- Eponyms (`EPONYM`): `lyme`, `cushing's`, `parkinson's`, `baker's cyst`, `murphy's sign` …
- Code evidence (`CODE_EVIDENCE`): if an `icd10_codes` entry is an external-cause / activity /
  place code (V/W/X/Y chapters) or otherwise needs a "personal" detail, list the words that support
  it (`golfing`, `playing tennis`, `skiing`, `car accident`, `on the sidewalk`, `ice`). Such a detail
  goes **only** to `keep`, never to `pii`.
- Clinical numbers that look like dates/IDs (`CLINICAL_NUMBER`): `4/6`, `250/50`, `6/10`, `18-20/28`,
  `120/80` … (only the look-alikes, not every number).
- Words that are names elsewhere but clinical here (`NOT_A_NAME`): `walker` (device), `white blood
  cell`, `mr arthrogram`.

Be exhaustive for `high` risk items; be reasonable (not exhaustive) for `low`.
