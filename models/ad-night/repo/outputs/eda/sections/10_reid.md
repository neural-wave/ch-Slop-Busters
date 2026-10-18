## 10. Re-identification risk

Script: `scripts/eda/10_reid.py` → `outputs/eda/reid_profiles.json`. Quasi-identifier tokens come from the phase-3 regexes (hobbies/occupations/locations/teams/pets/relatives/brands/staff), so counts are approximate **[regex]**; k is computed inside the 207-encounter corpus (the organizers' test set and the real population are unknown).

### 10.1 k-anonymity on attribute combinations

| attributes available to the attacker | unique encounters | k ≤ 2 | k ≤ 5 | median k |
|---|---|---|---|---|
| sex | 1 | 1 | 1 | 104 |
| age decade + sex | 6 (3%) | 14 | 25 | 20 |
| age 5-year bucket + sex | 11 (5%) | 21 | 38 | 12 |
| exact age + sex | **50 (24%)** | 74 | 140 | 3 |
| code set alone (preserved by design) | **167 (80%)** | 179 | 207 | 1 |
| only the 6 most frequent codes | 5 | 7 | 10 | 127 |
| age decade + sex + code set | 201 (97%) | 207 | 207 | 1 |
| age 5y + sex + code set | 205 (99%) | 207 | 207 | 1 |
| exact age + sex + code set | 207 (100%) | | | 1 |
| + occupation, hobbies | 205–207 | | | 1 |
| sex + codes + occupation + hobbies, **no age** | 199 (96%) | | | 1 |

- The `icd10_codes` field, which must be preserved byte-for-byte, already makes 80% of encounters unique on its own and 97% together with decade + sex. **Within this corpus, k-anonymity is unattainable because of the labels**; the privacy metric can therefore only be about what the *text* reveals beyond the codes: direct identifiers (names, DOB, MRN, dates, staff, employers, places) and rare narrative details.
- Exact age alone is a strong quasi-identifier (24% unique with sex); the 5-year bucket reduces that to 5%, the decade to 3%. Ages ≥ 90 (2) and < 5 (2) are unique at any bucket → use open-ended buckets (`90+`, `under 5`).

### 10.2 Rare details

- 140/207 encounters contain ≥1 detail token that occurs in no other encounter (median 1; `virtscribe` 2.9, `aci` 1.9, `virtassist` 0.8 on average). Corpus-unique tokens by category: hobby 160, staff/doctor names 53, locations 46, employers/occupations 45, brands 30, pets 11, named relatives 9, teams 7, events 5, vehicles 5.
- Most identifying bundles **[verified]**: `D2N138` (army enlistment spring 2022 + Enterprise rent-a-car + Kaiser Permanente Broadway + peanut anaphylaxis + wildfire smoke), `D2N076` (twin infant with Smith-Magenis syndrome, family lives in Dallas, Georgia), `D2N185` (14-year-old with renal hypoplasia, softball, Jacksonville/Atlanta, father had a kidney transplant, relative died on dialysis in his 40s), `D2N157` (San Francisco 49ers fan from the west coast, lives with his dog, retiring), `D2N189` (Denver Nuggets fan, swimming), `D2N154` (triathlon + hockey + Red Wings + Pittsburgh), `D2N074` (computer programmer, married 25 years, pituitary tumour), `D2N029` (owns an auto repair shop, melanoma, mother and aunt with melanoma), `D2N058` (insurance through "the plant", photography, gardening, trails behind the rec center, EF 35%), `D2N105` (100-year-old, walker, cul-de-sac, son had an MI).
- Details that are *also* the code evidence (see §7): `golf`, `bowling`, `soccer`, `skiing`, `running`, `car accident`, `sidewalk`, `snow` — cannot be removed in the 24 external-cause slots without losing the code; they are low-cardinality words (k ≥ 3 for most sports) and only become identifying in combination with a place or team.

### 10.3 Attacker model (feeds the attack bonus)

An attacker holding the original encounter and the anonymized output can link them by:
1. **Codes + sex + age bucket** — always available; 97% unique. This is why the attack bonus should be scored on *text-only* features, with codes removed from both sides, or it is trivially won.
2. **Structural fingerprint** — number of turns, note section sequence, boilerplate variant, clinical numbers (A1c 7.4, creatinine 0.7, EF 35%, weights, doses). Text similarity (TF-IDF over the clinical content) will re-link ~100% of pairs even after perfect PII removal, because the clinical content *is* kept by design. A meaningful attack must therefore try to recover the *removed* values (name, DOB, employer) — e.g. by asking an LLM to guess the name from context — rather than to match encounters.
3. **Surviving quasi-identifiers** — the 706 corpus-unique tokens above; a name/date-only anonymizer leaves all of them. The adversarial evaluation should count how many of these survive (dictionary-based) and whether an LLM can produce the employer/city/team from the anonymized text.
4. **Cross-field leaks** — if the dialogue is anonymized but the note still carries `Mrs. Hernandez`, or the note is anonymized but the dialogue still has `date of birth , march 26 2001`, one field re-identifies the other. Every entity must be mapped jointly per `encounter_id`.
