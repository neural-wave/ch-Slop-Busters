# The merged ACI-eval gold

The main suite, `aci_eval` (the 67 challenge evaluation encounters), has three PII golds. Each model team built and
tuned against "its" gold, so scoring every model on one of them favours its team. `python3 -I eval gold`
(`eval/gold/merge_aci_gold.py`) merges them into one gold with votes. The scorer reports three levels of it, plus each
gold on its own so the old numbers stay reproducible.

```bash
python3 -I eval gold                 # data/eval/gold/aci_eval/merged.jsonl, merged_stats.json, merged_review.md
python3 -I eval gold --audit         # + audit.md / audit.json (or: python3 -I eval/gold/audit_gold.py)
python3 -I eval suites --only aci_eval              # labels from the merged gold (rebuilt when it changes)
python3 -I eval suites --only aci_eval --no-merged  # the old labels: harness `pii` + VM `pii_alt`
```

`eval all` runs `gold` first. The merge is deterministic: the same inputs give byte-identical files, so an unchanged
gold does not rebuild the suite.

## Sources

| Gold | Path (under `neuralwave-container/`) | Made by | Eval split | Label set |
|---|---|---|---|---|
| `harness` | `neuralwave/data/converted/aci_bench/labels.jsonl` (imported by `scripts/convert/aci_bench.py`) | A+D team's annotators | 532 PII items, 194 keep items | the harness's 11 categories, per-item risk |
| `vm` | `models/anonymous/annotations/eval/<id>.json` (`GUIDELINES.md` there) | VM teacher annotation for the anonymous model | 815 PHI items (tier 1/2), 5,211 concepts | 15 PHI types, 6 concept types |
| `ketai` | `ketai-eval/gold/eval_gold.jsonl` (`build_gold.py` there) | ketai's reference annotation | 484 PII items, 4,334 concepts, 125 non-PII ambiguous terms | 11 PII categories, 7 concept types |

Item counts are distinct (field, lowercased text) pairs with at least two alphanumeric characters, the unit the suites
score. All three golds were made with LLM assistance, from the same text and with similar instructions, so their votes
are less independent than they look: two golds agreeing is weaker evidence than two human annotators agreeing.

Measured exact agreement on (encounter, field, lowercased text), before any nesting: 289 items in all three golds,
490 in at least two, 1,052 in the union. Most disagreement is about span length, not about what is private: 231
harness items are not in the ketai gold, and 173 of them are whole-word substrings of a ketai item (`martha` and
`collins` vs `Martha Collins`, `gutierrez` vs `dr. gutierrez`). 129 harness items are in neither other gold, 107 of
them inside a ketai item.

## Label harmonization (`gold/taxonomy.py`)

Every label maps to the harness's categories plus `CONTACT`, as the paper (Cornelius & Rinaldi 2026, Table 3) maps
its four corpora to PERSON, DATE, CONTACT and ADDRESS. The source label stays in the item (`label`, `source_labels`).

| Category | Risk | Direct | VM types | ketai categories |
|---|---|---|---|---|
| `PATIENT_NAME` | high | yes | `NAME_PATIENT` | `PATIENT_NAME` |
| `PERSON_NAME` | high | yes | `NAME_RELATIVE`, `NAME_CLINICIAN`, `NAME_OTHER` | `CLINICIAN_NAME`, `OTHER_PERSON_NAME` |
| `DOB` | high | yes | `DATE` with a DOB cue (below) | `DATE` with a DOB cue |
| `DATE` | high | yes | `DATE` | `DATE` |
| `YEAR` | medium | no | `DATE` that is a bare 4-digit year | `DATE` that is a bare 4-digit year |
| `ID` | high | yes | `ID` | `ID` |
| `CONTACT` | high | yes | `CONTACT` (was `ID` in the old VM labels) | `CONTACT` |
| `AGE` | high | no | `AGE` | `AGE` |
| `LOCATION`, `ORGANIZATION`, `OCCUPATION` | medium | no | same name | same name |
| `OTHER_QI` | low | no | `ACTIVITY`, `LIFE_EVENT`, `TEMPORAL`, `PERSONAL_DETAIL` | `OTHER_QUASI` |

- A VM or ketai `DATE` becomes `DOB` when the harness says `DOB` for the same string, or when `date of birth`, `dob`
  or `born` occurs in the 25 characters before one of its occurrences; else `YEAR` when it is `^\d{4}$`; else `DATE`.
  Harness categories are kept as they are. An unknown label becomes `OTHER_QI`.
- `direct` follows the paper's schema and the task's example ("names, date of birth"): names, DOB, dates, IDs and
  contacts. Ages and the quasi-identifiers are not direct.
- Category conflicts between the golds that list the same string: majority, ties go to harness, then ketai, then VM.
  27 conflicts on the eval split, listed in `merged_review.md` (mostly hospitals the harness calls `LOCATION` and the
  other two `ORGANIZATION`).
- `risk` follows the category (`common.CATEGORY_RISK`). The harness's 7 per-item risk overrides (`the 28th` as a
  medium-risk `DATE`, `works from home` as a low-risk `OCCUPATION`) live on in `pii_by_source.harness` only.

Concept types: VM's upper-case types and ketai's lower-case ones, `lab_or_test` → `TEST`, ketai's `anatomy` →
`ANATOMY`.

## Merge rules (`gold/merge_aci_gold.py`)

Per encounter and field, each gold's own list (usable, distinct per lowercased text, as the suites used it) goes in:

1. **Matching**: one item per lowercased string; strings with no whole-word occurrence in their field are dropped
   (none on the eval split). The surface form comes from the first gold in the order harness, ketai, VM.
2. **Nesting**: the longest span is canonical. A shorter item that is a whole-word substring of a longer one (by the
   scorer's own matching rule, `common.pattern`) is listed in the longer item's `contains`, and its golds vote for the
   longer item (`nested_votes`: the votes that come only from such a part). The shorter item stays an item of its own
   only when it also occurs outside every container: `standalone_occurrences` counts its occurrences that do not lie
   inside an occurrence of a container. Such a survivor lists its containers in `within` and inherits their votes. The
   others are absorbed (235 on the eval split, e.g. `Collins` inside `Martha Collins` in the note). The plan's rule
   `count(s) > Σ count(containers)` is the same test, done on spans so that nested containers are not counted twice.
3. **Votes**: `votes` is the number of golds behind an item (listing it, a part of it, or a container of a survivor);
   `sources` names them and `source_labels` gives the labels of the golds that list the string itself.
4. **Keep**: the harness's keep items plus ketai's non-PII ambiguous terms (`EPONYM` when the term is, or is inside, a
   concept; else `NOT_A_NAME`; ketai's comment becomes `why`). A gold never votes for a string it lists as not PII
   (its keep items, and ketai's built-in non-PII words: `dax`, `dragon`, `doctor`, `patient`, `nurse`, weekdays), not
   through nesting and not through inheritance. A string that is both PII and keep stays PII when at least two golds
   list it, or a part of it, themselves (inherited votes do not count); otherwise keep wins and the PII item is
   dropped before nesting. 8 conflicts went to PII and 31 to keep (mostly VM-only activities such as `golfing` that
   the harness keeps as ICD code evidence).
5. **Concepts**: VM ∪ ketai per lowercased string; `type` by vote (tie: ketai, then VM), `icd_relevant` when any gold
   says so, `icd_relevant_votes`, `sources`, `votes`. A concept equal to a direct PII item is dropped (none on the
   eval split).

Merged PII item:

```json
{"field": "dialogue", "text": "Martha Collins", "category": "PATIENT_NAME", "risk": "high", "direct": true,
 "sources": ["harness", "vm", "ketai"], "votes": 3, "nested_votes": 1, "source_labels": {"vm": "NAME_PATIENT", "ketai": "PATIENT_NAME"},
 "label": "PATIENT_NAME", "contains": ["collins", "martha"], "standalone_occurrences": 1}
```

`merged.jsonl` has one row per encounter: `{"encounter_id", "pii", "pii_by_source": {"harness", "vm", "ketai"}, "keep",
"concepts"}`. `aci_eval`'s `labels.jsonl` copies `pii` (every merged item), `pii_by_source`, `keep` and `concepts`
from it; `suite.json` records `gold_sources`, `pii_levels` and `merged_sha1`, and the merged file is part of
`source_sha1`. `aci_dev` keeps the harness gold alone (A+D tuned on it and there is no second gold for it).

### Counts (eval split)

| | Items | Occurrences |
|---|---:|---:|
| union | 786 | 1,017 |
| consensus | 538 | 725 |
| direct | 261 | 393 |

From the 1,052 exact strings: 235 nested strings are absorbed and 31 lose to keep, which leaves 786. Nesting turns
span disagreement into agreement: 456 items have three votes (289 exactly), 82 have two, 248 have one. Every direct
item has at least two votes, so on this split consensus is exactly the items with two or three votes. The 248
single-vote items are 235 `OTHER_QI` and 13 `OCCUPATION`, 244 of them VM's: the VM guidelines ask for hobbies, life
events and other rare personal facts that the other two golds mostly leave out. Keep: 267 items (150 harness, 73 ketai, 44 both).
Concepts: 6,607 (2,937 in both golds, 2,273 VM only, 1,397 ketai only), 3,860 ICD-relevant.

## Levels in the score (`metrics.privacy`)

| Level | Items | Use |
|---|---|---|
| `consensus` | direct, or at least two votes | **headline**: `removed`, `removed_high_risk`, `breakdown` and `counts` are on this level |
| `direct` | names, DOB, dates, IDs, contacts | a gate |
| `union` | every merged item | pessimistic bound |

`privacy.levels.<level>` = `{removed, removed_high_risk, entity_removed, items, occurrences}`. `entity_removed` is the
share of items fully removed (no occurrence and no distinctive token left), the paper's strict entity-level matching;
`privacy.entity_removed` is the consensus one. `privacy.by_source.<gold>` scores each gold's own list
(`pii_by_source`), so `by_source.harness` is the old `aci_eval` privacy and `by_source.vm` the old VM-gold privacy
(`privacy_alt`, now gone). Breakdown keys: `level:<level>` and `entity:<level>`, plus `votes:<n>` and `src:<gold>` over
the union items (`src:vm`: the merged items VM supports). Leaks list every union item with its `votes` and `direct`.
An item without vote metadata belongs to every level, so on the other suites the levels coincide and `by_source` is
empty.

Re-scored `aci_eval` (runs unchanged):

| model | consensus | high-risk | entity | direct | union | union entity | harness | VM | ketai |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ad_night` | 0.7738 | 0.9404 | 0.7472 | 0.9262 | 0.5615 | 0.5165 | 0.8388 | 0.5398 | 0.8211 |
| `ad_night_submitted` | 0.7738 | 0.9404 | 0.7472 | 0.9262 | 0.5615 | 0.5165 | 0.8388 | 0.5398 | 0.8211 |
| `ad_night_nocodes` | 0.7793 | 0.9404 | 0.7509 | 0.9262 | 0.5664 | 0.5204 | 0.8428 | 0.5452 | 0.8257 |
| `ad_rules` | 0.7586 | 0.9384 | 0.7286 | 0.9237 | 0.5506 | 0.5038 | 0.8279 | 0.5316 | 0.8058 |
| `anonymous` | 0.9310 | 0.9821 | 0.9126 | 0.9822 | 0.8368 | 0.8041 | 0.9444 | 0.8391 | 0.9465 |
| `anonymous_rules` | 0.6097 | 0.8310 | 0.5558 | 0.8041 | 0.4562 | 0.4059 | 0.6355 | 0.4458 | 0.6651 |
| `identity` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

For every model the `harness` and `VM` columns equal the old `aci_eval` privacy and VM-gold privacy to the fourth
decimal (`ad_night` 0.8388 and 0.5398). 16 keep strings sit inside a longer PII item (`hay` inside
`unloading a lot of hay`; the audit lists them), so removing exactly the gold also removes them.

## Audit (`gold/audit_gold.py`)

Joins ketai's `audit_gold.py` and VM's `check_annotations.py`; writes `audit.md` and `audit.json` next to the merged
gold and exits 1 when `merged.jsonl` is missing. Current eval split:

- verbatim: every PII and keep string of the three golds occurs as a whole word in its field; one VM concept (`SPO2`)
  occurs only as part of a longer token.
- contradictions: 18 PII items are also concepts (activities such as `pilates` that ketai lists as concepts for ICD
  external-cause codes), 16 keep items sit inside a PII item; no PII is also keep, none is made only of the
  encounter's ICD-10 description words.
- regex candidates (ages, dates, title + name, e-mails, phones, digit runs of 6+) that no union or keep item covers:
  none.
- single-vote items, the vote histogram per category, and the diff against the previous `merged.jsonl`
  (`merged.prev.jsonl`, kept by the merge when the content changes).

## Caveats

- All three golds are LLM-assisted (see Sources); votes are not independent evidence.
- `direct` does not saturate: `ad_night` keeps bare month names (`march`, `june`) that VM and ketai annotate as
  `DATE` and the harness does not, so its direct removal is 0.926. A month alone identifies little; the level follows
  the plan's category rule and does not special-case it.
- Inheritance can promote a generic word: a survivor inherits the votes of its container even for occurrences
  elsewhere (`monday` from `next monday is actually my birthday`). `merged_review.md` lists the 18 survivors.
- A survivor's occurrences inside its container are counted again by the scorer (`martha` inside `Martha Collins`),
  as the harness gold always did. `standalone_occurrences` is metadata for the merge, not used by the scorer.
- Rebuilding `aci_eval` changes every published ACI privacy number: the headline is now the consensus level. Use
  `by_source.harness` for the A+D cross-check (0.839) and `by_source.vm` for the old VM-gold number.
- The organizers' hidden gold is unknown; the three levels bracket it.
