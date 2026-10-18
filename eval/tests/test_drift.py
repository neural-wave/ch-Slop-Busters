"""bonus.drift: divergences vanish on identical corpora and saturate on disjoint ones; small cases with known values for
the diversity, readability and structural statistics; drift.json on a toy reference and toy suites."""

from __future__ import annotations

import json
import math
from collections import Counter

import pytest

from eval.bonus import drift as D

FORM = ["PATIENT INFORMATION\nName: John Smith\n- Fever.\n1. Cough.\nBP=120/80 @home"]
CORPUS = [
    "[doctor] hi , andrew . how are you ?\n[patient] i'm fine , thanks .\n[doctor] any chest pain ?",
    "CHIEF COMPLAINT\n\nChest pain.\n\nASSESSMENT AND PLAN\n1. Hypertension.\n- Medical Treatment: Lisinopril 10 mg daily.\nDr. Smith saw him.",
]


def test_identical_corpora_have_zero_drift():
    m = D.compare(D.profile(CORPUS), D.profile(list(CORPUS)))
    assert m["jsd_unigram"] == 0 and m["jsd_bigram"] == 0 and m["vocab_jaccard"] == 1
    diffs = {k: v for k, v in m.items() if k.endswith("_diff")}
    assert len(diffs) == len(D.STATS) and set(diffs.values()) == {0}


def _letters(i: int) -> str:
    return "zq" + "".join(chr(97 + int(d)) for d in str(i))


def test_disjoint_vocabulary():
    """Every word type renamed to an unseen one, so no type is shared: Jaccard 0, and JSD in bits saturates at 1
    (ln 2 in nats, up to the eps smoothing)."""
    rename: dict[str, str] = {}
    other = [D.WORD.sub(lambda m: rename.setdefault(m.group(0).lower(), _letters(len(rename))), t) for t in CORPUS]
    m = D.compare(D.profile(CORPUS), D.profile(other))
    assert m["vocab_jaccard"] == 0 and m["types"] == len(rename)
    assert m["jsd_unigram"] == pytest.approx(1, abs=1e-6) and m["jsd_bigram"] == pytest.approx(1, abs=1e-6)
    assert m["line_breaks_per_100_diff"] == 0 and m["tokens"] == D.profile(CORPUS)["tokens"]  # one word for one word


def test_jsd_known_values():
    p, q = Counter("ab"), Counter("bc")
    assert D.jsd(p, p) == 0
    assert D.jsd(p, q) == pytest.approx(0.5, abs=1e-6) and D.jsd(q, p) == pytest.approx(D.jsd(p, q))  # half the mass shared
    assert D.jsd(Counter("ab"), Counter("cd")) == pytest.approx(1, abs=1e-6)  # bits: ln 2 nats
    assert D.jsd(Counter(), p) is None
    assert D.jaccard("ab", "bc") == pytest.approx(1 / 3) and D.jaccard("ab", "cd") == 0


def test_mtld_known_values():
    assert D.mtld("a b a b a b".split()) == 3.0  # TTR 2/3 <= 0.72 after every 3 tokens: 2 factors, both directions
    assert D.mtld("a b c a".split()) == pytest.approx(4 / (0.25 / 0.28))  # one partial factor (1 - 0.75) / (1 - 0.72): 4.48
    assert D.mtld("a b c".split()) == 3.0  # TTR never falls: one factor, the length
    assert D.mtld([]) is None


def test_yule_k_entropy_distinct_repetition():
    assert D.yule_k(Counter("aab")) == pytest.approx(1e4 * (5 - 3) / 9)  # 2222.22
    assert D.yule_k(Counter("abc")) == 0 and D.yule_k(Counter()) is None
    assert D.entropy(Counter("ab")) == 1 and D.entropy(Counter("aaaa")) == 0
    toks = "a b a b".split()
    assert D.distinct(Counter(toks)) == 0.5 and D.distinct(Counter(zip(toks, toks[1:]))) == pytest.approx(2 / 3)
    toks = "a b c d a b c d".split()
    assert D.self_repetition(Counter(zip(toks, toks[1:], toks[2:], toks[3:]))) == pytest.approx(2 / 5)  # abcd twice of 5 4-grams


def test_profile_counts_ngrams_within_texts():
    p = D.profile(["a b a b", "c d c d"])
    assert p["tokens"] == 8 and p["types"] == 4 and p["values"]["distinct_1"] == 0.5
    assert p["values"]["distinct_2"] == pytest.approx(4 / 6) and p["bigrams"][("b", "c")] == 0  # no bigram across the two texts
    assert p["values"]["self_repetition_4"] == 0  # abab and cdcd once each


@pytest.mark.parametrize("word,n", [("the", 1), ("make", 1), ("table", 2), ("people", 2), ("agree", 2), ("while", 1), ("syllable", 3),
                                    ("rhythm", 1), ("hypertension", 4), ("120", 1)])  # fmt: skip
def test_syllable_heuristic(word, n):
    assert D.syllables(word) == n


def test_flesch_kincaid_grade():
    v = D.profile(["The cat sat. The dog ran."])["values"]
    assert v["flesch_kincaid"] == pytest.approx(0.39 * 3 + 11.8 * 1 - 15.59)  # 6 words, 2 sentences, 6 syllables: -2.62
    assert D.flesch_kincaid(0, 0, 0) is None


def test_sentences():
    assert D.sentences("Dr. Smith saw J. Doe today. 1. Rest. 2. Fluids!") == ["Dr. Smith saw J. Doe today.", "1. Rest.", "2. Fluids!"]
    assert D.sentences("hi , andrew . how are you ?") == ["hi , andrew .", "how are you ?"]
    assert D.sentences("BP 120/80. Pulse 2.5 mg") == ["BP 120/80.", "Pulse 2.5 mg"]


def test_structural_and_token_class_ratios():
    """5 lines, 4 breaks, 11 whitespace tokens, 5 sentences of 2, 3, 1, 2, 4 words."""
    v = D.profile(FORM)["values"]
    assert v["line_breaks_per_100"] == pytest.approx(400 / 11)
    assert v["key_value_ratio"] == pytest.approx(2 / 5)  # `Name: John Smith`, `BP=120/80`
    assert v["bullet_ratio"] == pytest.approx(2 / 5)  # `- Fever.`, `1. Cough.`
    assert v["allcaps_ratio"] == pytest.approx(3 / 11)  # PATIENT, INFORMATION, BP=120/80
    assert v["caps_ratio"] == pytest.approx(8 / 11)
    assert v["punct_ratio"] == pytest.approx(7 / 11)  # `=` is a math symbol, `/` and `@` punctuation
    assert v["digit_ratio"] == pytest.approx(2 / 11) and v["special_char_ratio"] == pytest.approx(2 / 11)
    assert v["sentence_length_mean"] == pytest.approx(2.4) and v["sentence_length_std"] == pytest.approx(math.sqrt(1.04))


@pytest.mark.parametrize("text,bullet", [("- Rest", 1), ("1. Rest", 1), ("2) Rest", 1), ("a) Rest", 1), ("(1) Rest", 1), ("•Rest", 1),
                                         ("Rest", 0), ("a. Rest", 0), ("2.5 mg", 0)])  # fmt: skip
def test_bullet_openings(text, bullet):
    """The paper's bullet symbol or ordinal with a trailing period, extended to `2)`, `a)`, `(1)` and a glued glyph."""
    assert D.profile([text])["values"]["bullet_ratio"] == bullet


def test_literal_newline_escape_is_a_line_break():
    assert D.profile(["First line.\\nSecond line."])["values"] == D.profile(["First line.\nSecond line."])["values"]


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _enc(i, dialogue, note):
    return {"encounter_id": f"e{i}", "dataset": "toy", "dialogue": dialogue, "note": note, "icd10_codes": []}


@pytest.fixture
def toy(tmp_path, monkeypatch):
    train = [_enc(0, CORPUS[0], CORPUS[1]), _enc(1, "[doctor] any fever ?\n[patient] no .", "Fever: none.")]
    _jsonl(tmp_path / "converted" / "aci_bench" / "encounters.jsonl", [*train, _enc(2, "eval only words", "unseen")])
    _jsonl(tmp_path / "converted" / "aci_bench" / "labels.jsonl", [{"meta": {"split": s}} for s in ("train", "train", "eval")])
    _jsonl(tmp_path / "suites" / "same" / "encounters.jsonl", train)
    _jsonl(tmp_path / "suites" / "half" / "encounters.jsonl", [train[0], _enc(3, "", "Completely different vocabulary here.")])
    _jsonl(tmp_path / "suites" / "other" / "encounters.jsonl", [_enc(4, "", "Zebra quokka marmot.")])
    monkeypatch.setattr(D, "CONVERTED", tmp_path / "converted")
    monkeypatch.setattr(D, "SUITES_DIR", tmp_path / "suites")
    monkeypatch.setattr(D, "DRIFT_FILE", tmp_path / "drift.json")
    return tmp_path


def test_drift_json_contract(toy):
    assert D.main([]) == 0
    doc = json.loads((toy / "drift.json").read_text(encoding="utf-8"))
    assert list(doc) == ["reference", "suites", "complexity_rank"]
    assert {k: doc["reference"][k] for k in ("name", "records")} == {"name": "aci_train", "records": 2}
    assert doc["complexity_rank"] == {"same": 1, "half": 2, "other": 3} and list(doc["suites"]) == ["same", "half", "other"]
    same = doc["suites"]["same"]
    assert same["records"] == 2 and same["tokens"] == doc["reference"]["tokens"] and same["jsd_unigram"] == 0 and same["vocab_jaccard"] == 1
    assert all(v == 0 for k, v in same.items() if k.endswith("_diff"))
    assert doc["suites"]["other"]["vocab_jaccard"] == 0 and doc["suites"]["other"]["jsd_unigram"] == pytest.approx(1, abs=1e-5)
    first = (toy / "drift.json").read_bytes()
    assert D.main(["--suites", "same", "half", "other"]) == 0 and (toy / "drift.json").read_bytes() == first  # deterministic


def _mark(toy, *suites):
    """Overwrite jsd_bigram of `suites` in drift.json with a marker that survives only if the entry is kept."""
    doc = json.loads((toy / "drift.json").read_text(encoding="utf-8"))
    for s in suites:
        doc["suites"][s]["jsd_bigram"] = -1
    (toy / "drift.json").write_text(json.dumps(doc), encoding="utf-8")


def _bigram(toy, suite):
    return json.loads((toy / "drift.json").read_text(encoding="utf-8"))["suites"][suite]["jsd_bigram"]


def test_unselected_suites_kept_unless_changed(toy, capsys, monkeypatch):
    D.main([])
    _mark(toy, "same", "half", "other")
    _jsonl(toy / "suites" / "other" / "encounters.jsonl", [_enc(4, "", "Chest pain."), _enc(5, "", "Fever.")])
    assert D.main(["--suites", "same", "missing"]) == 0
    doc = json.loads((toy / "drift.json").read_text(encoding="utf-8"))
    assert doc["suites"]["half"]["jsd_bigram"] == -1  # not selected and unchanged: kept from the previous file
    assert doc["suites"]["same"]["jsd_bigram"] == 0  # selected: recomputed
    assert doc["suites"]["other"]["records"] == 2 and doc["suites"]["other"]["jsd_bigram"] > 0  # not selected, but its input changed
    assert "suite missing is not built" in capsys.readouterr().err
    monkeypatch.setattr(D, "CODE_SHA1", "changed")
    assert D.main(["--suites", "same"]) == 0 and _bigram(toy, "half") > 0  # the drift code changed: everything recomputed
    (toy / "suites" / "half" / "encounters.jsonl").unlink()
    assert D.main(["--suites", "same"]) == 0
    doc = json.loads((toy / "drift.json").read_text(encoding="utf-8"))
    assert set(doc["suites"]) == set(doc["complexity_rank"]) == {"same", "other"}  # a suite no longer built is dropped
    assert doc["suites"]["other"]["input_sha1"] == D.sha1_file(toy / "suites" / "other" / "encounters.jsonl")


def test_missing_reference_or_suites(toy, capsys):
    (toy / "converted" / "aci_bench" / "labels.jsonl").unlink()
    assert D.main([]) == 1 and "reference corpus missing" in capsys.readouterr().err
    for p in (toy / "suites").glob("*/encounters.jsonl"):
        p.unlink()
    assert D.main([]) == 1 and "no suite" in capsys.readouterr().err
