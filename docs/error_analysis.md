# Error analysis on the tune split (M4)

Run date 17 October 2026. Silver labels (`labels/merged/tune`, 140 encounters); outputs of the M1
baseline (`balanced` and `aggressive` presets, `full` mode). Produced by
`scripts/error_analysis.py`; patterns are abstract shapes (digits as 9, month names as MONTH, other
words as W or W+), never the original strings. Causes come from the decisions sidecar:

- **no detector**: no decision overlaps the labeled occurrence;
- **kept: ...**: a detector found it and the medical shield or the preset kept it;
- **edited elsewhere**: an edit overlaps but the labeled string survives (none were found).

## Run: tune_balanced_full on tune

### Surviving labeled PII occurrences by category and cause

| category (risk) | occurrences | left | causes | top shapes |
|---|---:|---:|---|---|
| OTHER_QI (low) | 146 | 95 | kept: shield:activity_context 44, no detector 29, kept: preset:balanced keeps OTHER_QI 22 | `W` 65, `W+` 24, `the W` 2, `W+ 99 years` 2, `in W+` 1, `W NUM NUM NUM W` 1 |
| OCCUPATION (medium) | 33 | 17 | no detector 17 | `W` 8, `W+` 6, `in W` 2, `W in W+` 1 |
| ORGANIZATION (medium) | 54 | 12 | no detector 11, kept: shield:drug 1 | `W+` 5, `W` 4, `the W` 2, `W on W` 1 |
| LOCATION (medium) | 51 | 8 | no detector 8 | `W+` 4, `W` 3, `the W` 1 |
| AGE (high) | 239 | 5 | no detector 4, kept: shield:clinical_number 1 | `W 99` 3, `9 - year - old` 1, `W 99 W` 1 |
| DATE (medium) | 4 | 3 | no detector 3 | `MONTH` 3 |
| PERSON_NAME (high) | 82 | 1 | no detector 1 | `W` 1 |
| DATE (high) | 73 | 1 | no detector 1 | `early MONTH` 1 |
| LOCATION (low) | 1 | 1 | no detector 1 | `W of the W` 1 |

### Keep-trap occurrences removed, by trap category and the edit that removed them

| trap category | removed by (detector -> category) |
|---|---|
| CODE_EVIDENCE | gazetteer:activity -> ACTIVITY 2 |

## Run: tune_aggressive_full on tune

### Surviving labeled PII occurrences by category and cause

| category (risk) | occurrences | left | causes | top shapes |
|---|---:|---:|---|---|
| OTHER_QI (low) | 146 | 29 | no detector 29 | `W` 16, `W+` 9, `the W` 2, `in W+` 1, `W NUM NUM NUM W` 1 |
| OCCUPATION (medium) | 33 | 17 | no detector 17 | `W` 8, `W+` 6, `in W` 2, `W in W+` 1 |
| ORGANIZATION (medium) | 54 | 12 | no detector 11, kept: shield:drug 1 | `W+` 5, `W` 4, `the W` 2, `W on W` 1 |
| LOCATION (medium) | 51 | 8 | no detector 8 | `W+` 4, `W` 3, `the W` 1 |
| AGE (high) | 239 | 5 | no detector 4, kept: shield:clinical_number 1 | `W 99` 3, `9 - year - old` 1, `W 99 W` 1 |
| DATE (medium) | 4 | 3 | no detector 3 | `MONTH` 3 |
| PERSON_NAME (high) | 82 | 1 | no detector 1 | `W` 1 |
| DATE (high) | 73 | 1 | no detector 1 | `early MONTH` 1 |
| LOCATION (low) | 1 | 1 | no detector 1 | `W of the W` 1 |

### Keep-trap occurrences removed, by trap category and the edit that removed them

| trap category | removed by (detector -> category) |
|---|---|
| CODE_EVIDENCE | gazetteer:activity -> ACTIVITY 34 |

## Rules mode for comparison (balanced)

The encoders and the LLM in `full` mode add the removal of 14 organization, 6 location, 17 low-risk
and 2 name occurrences over `rules`. They add nothing for occupations: 17 of 33 occurrences survive in
both modes, and no detector fires on any of them.

## What this says

1. **The keep-trap floor is missed by two occurrences.** Balanced loses 2 keep-trap occurrences, both
   code-evidence activities removed by the activity gazetteer. Aggressive loses 34 the same way.
   Protecting activities that support the encounter's external-cause codes clears the 99.5% floor.
   This is a prerequisite for every other feature, because the gate floor is absolute.
2. **Occupations have no working detector.** 17 of 33 occurrences survive in every mode. Organizations
   (11 of 54) and locations (8 of 51) are the next medium-risk gaps, all "no detector". This is the
   targeted LLM detector (F2).
3. **Low-risk details are a policy question.** Of 95 surviving low-risk occurrences in balanced, 44 are
   kept by the activity shield and 22 by the preset, and 29 have no detector. Aggressive removes the 66
   kept ones but costs the 34 trap occurrences. Separating the two needs the agreement rule (F4), with
   code evidence as the tie-breaker.
4. **Ages and dates are nearly solved on tune.** 5 of 239 age and 4 of 77 date occurrences survive.
   The date gap seen on validation (15 of 17 high-risk leaks) comes from dates that only Ketai's
   annotators labeled. The tune labels follow the upstream convention, so they cannot measure that gap.
   F1 needs tune labels for those date forms before it can be gated.
5. **Cross-field propagation has no evidence on tune.** No labeled string survived next to an edit of
   the same string ("edited elsewhere" = 0); the identity graph already covers names. F3 moves last.

## Revised M5 order

| Order | Feature | Evidence above | Expected effect |
|---|---|---|---|
| F0 (new) | Code-evidence guard for the activity gazetteer | finding 1 | Trap survival ≥ 99.5% in both presets |
| F2 | Targeted LLM detector (occupation, organization, location) + replay cache | finding 2 | Medium-risk removal up, quasi removal toward 75% |
| F4 | Agreement rule, `privacy` preset | finding 3 | Low-risk removal toward aggressive at the trap floor |
| F5 | Rare-token verifier | attack A3: 9 rare tokens survive | Residual identifying tokens |
| F1 | Temporal pass | finding 4; needs date labels on tune first | Dates toward Ketai's 91% on validation |
| F3 | Quasi propagation across fields | finding 5 | Kept as a safety net only if it passes the gate |
