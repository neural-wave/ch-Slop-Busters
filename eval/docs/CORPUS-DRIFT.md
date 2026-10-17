# Corpus drift

`eval/bonus/drift.py` measures how far the text of each suite is from the ACI train corpus, the data every team
developed on. The robustness bonus asks for the "complexity of additional data" next to the usual metrics (task brief,
p. 3). The drift rank is that complexity: rank 1 is the suite closest to ACI train.

```bash
python3 -I eval drift                     # every suite in data/eval/suites/ (about 8 s for 13 suites)
python3 -I eval drift --suites robust_medconv aci_eval
```

The metrics come from Cornelius & Rinaldi 2026, "Measuring the gap: correlating synthetic-to-real drift with PHI
de-identification" (`s44342-026-00072-9.pdf` in the container root). The metric families are listed on p. 4 (§4.2) and
defined on pp. 9–10 (appendix "Drift metrics details"). The correlations with ΔF1 are in §5.2 (p. 6) and Fig. 4
(p. 7) for the main analysis, and in Fig. 5 (p. 12) for the ablation.

## Output: `data/eval/drift.json`

```
reference        {"name": "aci_train", "records": 140, "tokens", "values": {<stat>: the reference's own value},
                  "code_sha1": sha1 of drift.py}
suites           {<suite>: {"records", "input_sha1": sha1 of its encounters.jsonl, "tokens", "types", "jsd_unigram",
                  "jsd_bigram", "vocab_jaccard", "<stat>_diff": |suite - reference|, "<stat>": the suite's own value}}
                 (suites ordered by rank)
complexity_rank  {<suite>: 1 = smallest jsd_unigram (closest to ACI train); ties by name}
```

- The file always covers every suite in `data/eval/suites/`, so the rank covers all of them, including the `robust_*`
  and Kaggle suites.
- `--suites` recomputes the named suites (`all` passes its `--suites`). The other suites keep their previous entry as
  long as the reference, `drift.py` (`code_sha1`) and their `encounters.jsonl` (`input_sha1`) are unchanged. Otherwise
  they are recomputed too, so a suite rebuilt after the last drift run never keeps a stale entry or rank. A suite that
  is no longer built is dropped.
- The output is deterministic: values are rounded to 6 decimals, and sums use `math.fsum`, so the result does not
  depend on hash order.
- The paper reports m_A, m_B and |m_A − m_B| "so departures are directly auditable" (p. 9). The file keeps all three:
  `reference.values`, `<stat>` and `<stat>_diff`.

## Corpora and tokenization

| What | Rule |
|---|---|
| Reference | The 140 ACI-Bench train encounters (`data/converted/aci_bench`, labels `meta.split == "train"`), dialogue and note: 229,443 word tokens |
| Suite | The dialogue and note of every record in its `encounters.jsonl`. Each field is a separate text, and n-grams never cross a text |
| Word tokens | `common.WORD` (runs of letters or digits, with one inner apostrophe), case-folded. Used by the n-gram, vocabulary, diversity, sentence-length and readability metrics |
| Whitespace tokens | `str.split()`, case kept. Used by the token-class ratios, all-caps, special characters and line-break density |
| Line breaks | `\n`, `\r\n`, `\r`, and the literal two-character escape `\n` that some sources leave in the text (293 of them in `pmc_patients`) |
| Sentences | A line is split after `.`, `!` or `?` followed by whitespace. There is no split after an abbreviation (`Dr.`, `Mr.`, `e.g.`, …), an initial (`J.`) or a leading ordinal (`1. Hypertension`). Only sentences with at least one word token count |
| Corpus statistics | Every statistic pools all texts of a corpus. MTLD runs over the concatenated token stream, in file order |

## Metrics (implemented)

The last column gives the paper's correlation r between the metric and ΔF1. ΔF1 is the gap between the best
real-trained F1 and the mean synthetic-trained F1 of PHI taggers, so a higher drift means a larger drop. The first
value comes from the main analysis (Fig. 4, p. 7) and the second from the ablation (Fig. 5, p. 12). A dash means the
paper does not report one.

| Key in `drift.json` | Family | Definition | A high value means | r with ΔF1 (main / ablation) |
|---|---|---|---|---:|
| `jsd_unigram` | lexical | Jensen-Shannon divergence ½KL(P‖M) + ½KL(Q‖M), M = ½(P + Q), of the unigram distributions. A pseudo-count ε = 1e-9 is added to every type of the union vocabulary ("small ε-smoothing", p. 9). Log base 2, so the range is [0, 1]: 0 for identical corpora, 1 for disjoint ones (ln 2 in nats) | different word choice. **Ranks the suites** | 0.54 / 0.71 |
| `jsd_bigram` | lexical | The same on bigrams (inside a text) | different local collocations | 0.54 / 0.69 |
| `vocab_jaccard` | lexical | \|V_ref ∩ V_suite\| / \|V_ref ∪ V_suite\| of the case-folded types (p. 9). A similarity, so **low** means drift | a closer vocabulary | −0.47 / −0.78 |
| `entropy_diff` | lexical diversity | \|H_ref − H_suite\|, with H = −Σ p(w) log₂ p(w) of the unigram distribution, in bits (p. 9) | a broader or narrower vocabulary use | 0.48 / 0.37 |
| `mtld_diff` | lexical diversity | MTLD (McCarthy & Jarvis 2010). A pass counts factors, the stretches after which the type-token ratio falls to 0.72 or below, plus a partial factor (1 − TTR)/(1 − 0.72) for the rest. MTLD is tokens / factors, averaged over a forward and a backward pass. A larger MTLD means a more diverse text (p. 9) | a different lexical diversity | 0.53 / 0.68 |
| `yule_k_diff` | lexical diversity | Yule's K = 10⁴ (Σᵢ i² Vᵢ − N) / N², where Vᵢ is the number of types seen i times. A smaller K means a more diverse text (p. 9) | different repetitiveness | 0.48 / 0.31 |
| `distinct_1_diff`, `distinct_2_diff` | lexical diversity | Distinct-n = unique n-grams / all n-grams, for n = 1 and 2 (p. 10) | different within-corpus repetition | 0.31, 0.27 / 0.37, 0.40 |
| `self_repetition_4_diff` | lexical diversity | The share of 4-gram occurrences whose 4-gram occurs more than once in the corpus (p. 10) | more or less template reuse | 0.27 / 0.45 |
| `sentence_length_mean_diff`, `sentence_length_std_diff` | style | \|μ_ref − μ_suite\| and \|σ_ref − σ_suite\| of word tokens per sentence, with the population σ (p. 10) | terser or more verbose text | 0.23, 0.26 / 0.21, 0.04 |
| `punct_ratio_diff` | style | The share of whitespace tokens that contain a Unicode punctuation character (category P*) (p. 10) | a different punctuation density | 0.42 / 0.36 |
| `digit_ratio_diff` | style | The share of whitespace tokens that contain a digit. The paper calls it "digit-containing tokens" (p. 10) and "Number Ratio" in Figs. 4 and 5 | a different numeric density (vitals, dates, IDs) | 0.45 / 0.60 |
| `caps_ratio_diff` | style | The share of whitespace tokens whose first letter or digit is an uppercase letter (p. 10) | cased text against lowercased transcripts, proper-noun density | 0.22 / 0.17 |
| `flesch_kincaid_diff` | readability | The difference of the FK grade = 0.39 · words/sentences + 11.8 · syllables/words − 15.59. Syllables come from a heuristic: vowel groups (a e i o u y and accented vowels), minus a silent final e (only after a consonant, so not in `agree`; not in consonant + `le`, as in `table`; and not when it is the word's only vowel group, as in `the`), at least 1, and a number counts as 1. The formula is English-only (p. 10) | a different reading level | 0.51 / 0.50 |
| `line_breaks_per_100_diff` | structural | 100 × line breaks / whitespace tokens (p. 10) | different paragraphing and wrapping | – |
| `key_value_ratio_diff` | structural | The share of sentences that start with a key of 1–4 words (the first character a letter) followed by `:` or `=`, such as `Name: John`, `A=B` or `CV: Mild edema`. A `Doctor:` speaker prefix counts too (p. 10) | form-like or log-like text | 0.27 / 0.20 |
| `bullet_ratio_diff` | structural | The share of sentences whose first whitespace token is a bullet symbol (`- * + – — • ▪ ►` …) or an ordinal with a trailing period (`1.`), as on p. 10. Extension beyond the paper: an ordinal closed by a parenthesis (`2)`, `a)`, `(1)`, `(a)`) and a bullet glyph glued to its word (`•Fever`) count too. On the current suites and the reference this changes no value | list-oriented text | 0.48 / 0.08 |
| `allcaps_ratio_diff` | structural | The share of whitespace tokens with at least 2 letters, all of them uppercase. Single-letter tokens never count (p. 10) | headings, acronyms, shouting | 0.26 / 0.43 |
| `special_char_ratio_diff` | structural | The share of whitespace tokens that contain one of `@ # $ % ^ & * _ ~ \| \ < > { } = +` or a backtick. The paper says "characters such as @#$%^&" (p. 10) | code-like, templated or placeholder text | 0.15 / 0.00 |

Each entry also carries `records`, `tokens`, `types` and the suite's own value of every statistic, which gives the
direction of a difference.

## Metrics not implemented (TODO)

These metrics need a neural model. The harness is standard library only, so each one would run as a worker script
with a model venv's interpreter (for example `models/anonymous-deps/.venv`, which has torch and transformers), the way
the anonymizers do.

| Metric | Family | Definition (p. 10) | Model it needs | r with ΔF1 (main / ablation) |
|---|---|---|---|---:|
| Perplexity diff | lexical (fluency) | PPL = exp(−1/N Σ log p(wᵢ)) under a fixed LM. Report the difference | GPT-2 (124M, `gpt2`), as in the paper. Runs on CPU | 0.52 / 0.64 |
| MMD | semantic | √MMD², biased estimator with an RBF kernel, bandwidth from the median heuristic, over sentence embeddings | a sentence encoder. The paper only says "a compact transformer encoder" (p. 4). Candidates: `sentence-transformers/all-MiniLM-L6-v2` (22M, CPU), or `bge-m3` through Ollama (the encoder of the A+D ICD retriever) | 0.49 / 0.63 |
| Fréchet distance | semantic | ‖μ_r − μ_s‖² + Tr(Σ_r + Σ_s − 2(Σ_r Σ_s)^½) of Gaussian fits to the embeddings | the same encoder, plus a matrix square root (numpy or scipy in the worker) | 0.48 / 0.64 |
| Embedding-mean distance | semantic | ‖μ_r − μ_s‖₂ | the same encoder | 0.50 / 0.63 |
| Cluster-coverage JSD | semantic | k-means on the reference embeddings, suite embeddings assigned to the centroids, then the JSD of the two cluster histograms | the same encoder, plus k-means (scikit-learn in the worker) | 0.50 / 0.72 |

## Reading the numbers

- **Rank.** `complexity_rank` orders the suites by `jsd_unigram`, the strongest single predictor of the main analysis
  ("JSD Unigram and JSD Bigram (both r = 0.54)", p. 6). The other metrics explain *how* a suite differs: casing,
  lists, forms, sentence length, numbers.
- **Floor.** `aci_dev` holds 40 of the 140 reference encounters, so it is not an independent suite. It shows the
  in-distribution floor: JSD unigram ≈ 0.03, bigram ≈ 0.18, Jaccard ≈ 0.54. `aci_eval` comes next (≈ 0.04). Every
  other suite should be read against these two.
- **Size.** JSD and Jaccard depend on corpus size. A small sample from the same distribution has fewer types, so its
  Jaccard is lower and its JSD higher. Compare suites of similar size, and be careful with `sadiqguru` (4 records,
  about 6k tokens) and `devp_discharge` (about 5k tokens).
- **Language.** `imerit_multi` is German, French, Spanish and Dutch. The lexical metrics put it last, as they should,
  but its Flesch-Kincaid value (an English formula with an English syllable heuristic) is only a rough signal.
- **Direction.** The `_diff` values are unsigned. The suite's own `<stat>` and `reference.values` give the direction:
  for example, the ACI dialogues are lowercased ASR text, so a cased suite differs in `caps_ratio`.
- **Transfer.** The paper's correlations come from PHI taggers (BioClinicalBERT, DeBERTa) trained on one corpus and
  tested on another, across four PHI types (p. 4, p. 6). They say which kinds of drift tend to go with lower F1, not
  how much a given anonymizer will degrade. The text calls r a Pearson correlation (p. 6). The Fig. 4 caption says
  Spearman (p. 7). Line-break density is defined on p. 10, but it is not among the 24 metrics of Figs. 4 and 5.
