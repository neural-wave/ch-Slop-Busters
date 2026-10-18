## 11. Train vs eval shift

Script: `scripts/eda/11_shift.py` → `outputs/eda/train_eval_shift.json`. **[regex]**.

| dimension | train (140) | eval (67) | shift / implication |
|---|---|---|---|
| dataset mix | aci 77 / virtassist 35 / virtscribe 28 (55/25/20%) | 35 / 20 / 12 (52/30/18%) | same mix; the hidden test set is presumably drawn the same way |
| dialogue words (median) | 1240 | 1240 | none (eval virtscribe is longer: 1821 vs 1332) |
| note words (median) | 420 | 404 | none |
| codes per encounter | 2.04 | 2.28 | slightly more codes in eval |
| distinct codes | 146 | 112 | **79 eval codes unseen in train (82/153 slots, 49/67 encounters)**; 108 train codes are singletons |
| ICD chapters | no V/W/X/Y at all | V 1, W 11, X 2, Y 10 (24 slots) | external-cause / activity / place codes only in eval → the activity/place vocabulary is utility-critical in eval but invisible in train |
| top codes | I10 37, E11.9 29, F32.A 12 | E11.9 12, I10 7, I50.9 5, F32.A 5, N20.0 5 | hypertension less dominant in eval |
| patient-name density (per 1k words / % encounters) | greeting 0.46 / 74%, intro 1.57 / 84%, title 1.32 / 41% | 0.45 / 70%, 1.53 / 84%, 1.27 / 40% | identical |
| doctor/staff names | 0.40 / 22% | 0.39 / 22% | identical |
| DOB | 0.10 / 17% | 0.08 / 13% | all virtscribe in both |
| exact age | 0.99 / 84% | 0.89 / 79% | identical |
| absolute dates | 0.99 / 24% | **1.67** / 24% | eval virtscribe is date-dense (6.4 per 1k words) |
| locations | 0.35 / 25% | 0.22 / 15% | fewer in eval |
| employer / occupation / school | 0.36 / 24% | **0.49 / 40%** | more occupations in eval (aci 0.73) |
| hobbies / sports | 1.44 / 56% | 1.59 / 58% | similar; eval aci 2.47 (sports injuries → external-cause codes) |
| MRN / insurance | 0.21 / 11% | 0.12 / 7% | MRN in 9 train vs 2 eval encounters |
| named relatives / friends | 0.04 / 4% | 0.03 / 3% | rare in both |
| product brands (dax/dragon) | 1.17 / 39% | 1.27 / 46% | virtassist only |
| note headers | 38 distinct | 29 distinct | eval-only: `COPD.`, `HIV`, `PCOS.` (false headers); train-only: `BIRTH HISTORY`, `SUBJECTIVE`, `ORDERS`, `OTC`, `REVIEW OF SYMPTOMS`, `PAST SURGICAL HISTORY` |
| `[patient_guest]` | 5 encounters | **0** | pediatric/caregiver dialogues untested by eval; the hidden test set may contain them |
| untagged lines / swapped tags | 7 / 3 | 5 / 3 | present in both |
| aci dialogues without punctuation | 77/77 | 35/35 | universal |
| notes without any name | 37 | 21 | same share |
| non-ASCII characters | 617 | 298 | same rate |

- What this implies for the hidden test set: formats, PII densities and dialects are stable across the public splits, so rules tuned on train transfer. The *labels* do not: 70% of eval codes are unseen, singletons dominate, and a whole ICD chapter appears only in eval. Any local utility proxy that fits on train codes is invalid; use code-description similarity, evidence retention, or a frozen general-purpose ICD model. For privacy, the eval split is a faithful proxy. The 5 `[patient_guest]` encounters and the dictated-header encounters (36) are the formats most likely to be *under*-represented in eval and should be covered by the unit tests regardless.
