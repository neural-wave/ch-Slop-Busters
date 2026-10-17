"""Report, tradeoff, CSV exports: a tiny synthetic data/eval tree with known answers, built in tmp_path."""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET

import pytest

from eval import report as R

MODELS = """
[alpha]
cmd = ["x"]
[beta]
cmd = ["x"]
[gamma]
cmd = ["x"]
[alpha_nocodes]
cmd = ["x"]
blind_codes = true
[broken]
cmd = ["x"]
[identity]
cmd = ["x"]
[later]
cmd = ["x"]
enabled = false
"""

CODERS = """
[llm_note]
kind = "ollama"
primary = true
[knn]
kind = "knn"
"""

SECRET = "Zebulon Quixote"  # a leaked string in the score files: must never reach an output
HEADER = "model,suite,category,metric,value,n,ci_low,ci_high,coder,level,notes"


def arm(f1: float) -> dict:
    lv = {"precision": f1, "recall": f1, "f1": f1, "mean_f1": f1, "ci95": [round(f1 - 0.05, 4), round(f1 + 0.05, 4)]}
    return {"code": lv, "category": dict(lv)}


def coder(orig: float, anon: float, agree: float, failed: int = 0) -> dict:
    return {"coder": "c", "n_records": 3, "n_codes": 5, "original": arm(orig), "anonymized": arm(anon), "agreement": arm(agree),
            "f1_delta": {"value": round(anon - orig, 4), "ci95": [-0.1, 0.02]}, "f1_delta_category": {"value": round(anon - orig, 4), "ci95": [-0.1, 0.02]},
            "errors": {"original": 0, "anonymized": failed}, "invalid_codes": {"original": 0, "anonymized": 0},
            "valid_for_comparison": not failed, "changed_predictions": [{"encounter_id": "e1", "gold": [SECRET]}]}  # fmt: skip


def run(ms: float, ram: float, vram: float) -> dict:
    return {"exit_code": 0, "wall_s": round(3 * ms / 1000, 4), "n_records": 3, "ms_per_encounter": ms, "peak_ram_mb": ram,
            "peak_vram_mb": vram, "within_vram_budget": vram <= 4096}  # fmt: skip


def score(p: float, icd: dict | None = None, rn: dict | None = None, explained: float | None = 1.0) -> dict:
    lv = lambda r: {"removed": r, "removed_high_risk": r, "entity_removed": round(r * 0.9, 4), "items": 10, "occurrences": 12}  # noqa: E731
    sc = {
        "validity": {"valid_rate": 1.0, "records_changed": 1.0, "errors": []},
        "privacy": {"removed": p, "removed_high_risk": p, "entity_removed": round(p * 0.9, 4),
                    "levels": {"union": lv(round(p * 0.8, 4)), "consensus": lv(p), "direct": lv(round(min(1.0, p + 0.05), 4))},
                    "by_source": {"harness": lv(round(min(1.0, p + 0.0123), 4)), "vm": lv(round(p * 0.7, 4)), "ketai": lv(p)},
                    "breakdown": {"ALL": p, "cat:PATIENT_NAME": 1.0, "cat:OTHER_QI": round(p / 2, 4), "field:dialogue": p, "field:note": p,
                                  "lang:en": p, "risk:high": p, "votes:1": 0.25, "votes:3": 0.875, "src:harness": p, "level:union": 0.1},
                    "counts": {"ALL": [9, 12], "cat:PATIENT_NAME": [4, 4], "votes:1": [1, 4]},
                    "leaks": [{"text": SECRET, "category": "PATIENT_NAME"}]},
        "keep": {"kept": 0.98, "breakdown": {"ALL": 0.98, "category:EPONYM": 1.0}, "counts": {"ALL": [49, 50]}, "top_lost": [SECRET]},
        "concepts": {"kept": 0.97, "breakdown": {"ALL": 0.97, "icd_relevant": 0.99, "type:DIAGNOSIS": 1.0}, "top_lost": [SECRET]},
        "edits": {"non_pii_tokens_kept": 0.999, "pii_tokens_changed": 0.5, "edit_precision": 0.9, "tokens_changed": 40},
        "terms": {"kept": 0.995, "n_terms": 1000, "top_lost": [SECRET]},
        "placeholders_per_1k_words": 4.2,
        "explain": {"explained": explained, "changed_tokens": 40, "covered_tokens": 40, "decisions": 30, "with_reason": 30, "unaligned": 0, "source": "sidecar"},
        "icd": icd or {},
    }  # fmt: skip
    if rn:
        sc["run"] = rn
    return sc


def attack(a2: float, a3: float, a4_error: bool = False, a4: float | None = None) -> dict:
    arm_ = lambda r: {"recovered": r, "hits": 1, "total": 10, "by_category": {"PATIENT_NAME": r}}  # noqa: E731
    at = {"A2": {"name": "roster dictionary", "anonymized": arm_(a2), "control": arm_(1.0), "found": [{"token": SECRET}]},
          "A3": {"name": "rare words", "anonymized": arm_(a3), "control": arm_(1.0)},
          "A5": {"name": "record linkage", "anonymized": arm_(1.0), "control": arm_(1.0)}}  # fmt: skip
    if a4_error:
        at["A4"] = {"name": "LLM inference", "anonymized": None, "control": None, "error": "Ollama unreachable"}
    elif a4 is not None:
        at["A6"] = {"name": "LLM agent adversary", "anonymized": arm_(a4), "control": arm_(0.8), "llm": "claude-x"}
    return {"model": "m", "suite": "aci_eval", "attacks": at, "max_recovered": max(a2, a3)}


def write(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj, encoding="utf-8")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """alpha (0.9 / F1 0.38), beta (0.8 / 0.396), gamma (0.7 / 0.36): alpha and beta Pareto-optimal, gamma dominated by alpha."""
    out = tmp_path / "data" / "eval"
    write(tmp_path / "models.toml", MODELS)
    write(tmp_path / "coders.toml", CODERS)
    monkeypatch.setattr(R, "OUT", out)
    monkeypatch.setattr(R, "MODELS_TOML", tmp_path / "models.toml")
    monkeypatch.setattr(R, "CODERS_TOML", tmp_path / "coders.toml")
    sc = out / "scores"
    write(sc / "alpha" / "aci_eval.json", score(0.9, {"llm_note": coder(0.4, 0.38, 0.9), "tree": coder(0.7, 0.63, 0.95), "knn": coder(0.3, 0.3, 1.0)},
                                                run(512.25, 1234.5, 1500.0)))  # fmt: skip
    write(sc / "beta" / "aci_eval.json", score(0.8, {"llm_note": coder(0.4, 0.396, 0.97), "knn": {"error": "knn index missing"}}, run(90.0, 80.0, 5000.0)))
    write(sc / "gamma" / "aci_eval.json", score(0.7, {"llm_note": coder(0.4, 0.36, 0.8, failed=2)}, run(30.0, 50.0, 0.0), explained=0.5))
    write(sc / "alpha_nocodes" / "aci_eval.json", score(0.91, {"knn": coder(0.3, 0.29, 0.99)}, explained=None))
    write(sc / "broken" / "aci_eval.json", {"model": "broken", "suite": "aci_eval", "error": "run failed (exit 1)"})
    write(sc / "identity" / "aci_eval.json", score(0.0, {"llm_note": coder(0.4, 0.4, 1.0)}, run(5.0, 30.0, 0.0), explained=None))
    write(sc / "alpha" / "robust_synthetic.json", score(0.6))
    write(out / "runs" / "alpha_nocodes" / "aci_eval" / "run.json", run(700.0, 2000.0, 1000.0))  # a run.json without a run block in the score
    write(out / "runs" / "beta" / "pmc_patients" / "run.json", {**run(10.0, 10.0, 0.0), "exit_code": None})  # still running: ignored
    write(out / "attacks" / "alpha" / "aci_eval.json", attack(0.1, 0.2, a4_error=True))
    write(out / "attacks" / "beta" / "aci_eval.json", attack(0.3, 0.05, a4=0.2))
    write(out / "attacks" / "gamma" / "aci_eval.json", attack(0.3, 0.05, a4=0.1))
    write(out / "attacks" / "identity" / "aci_eval.json", attack(1.0, 1.0, a4=0.8))
    write(out / "suites" / "aci_eval" / "suite.json", {"name": "aci_eval", "role": "main", "about": "toy eval", "records": 3})
    write(out / "suites" / "robust_synthetic" / "suite.json", {"name": "robust_synthetic", "role": "robust", "about": "toy robust", "records": 2})
    write(out / "drift.json", {"reference": {"name": "aci_train", "records": 140, "tokens": 1000, "values": {"mtld": 60.0}},
                               "suites": {"aci_eval": {"records": 3, "tokens": 100, "jsd_unigram": 0.04, "jsd_bigram": 0.3, "vocab_jaccard": 0.45},
                                          "robust_synthetic": {"records": 2, "tokens": 50, "jsd_unigram": 0.5, "jsd_bigram": 0.9, "vocab_jaccard": 0.2}},
                               "complexity_rank": {"aci_eval": 1, "robust_synthetic": 2}})  # fmt: skip
    write(out / "gold" / "aci_eval" / "merged_stats.json", {"pii": {"levels": {"union": 20, "consensus": 12, "direct": 6},
                                                                    "occurrences": {"union": 30, "consensus": 18, "direct": 9}, "votes": {"1": 8, "2": 2, "3": 10}}})  # fmt: skip
    assert R.main([]) == 0
    return out


def md_table(md: str, heading: str) -> tuple[list[str], list[list[str]]]:
    lines = md.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("## ") and heading in ln)
    rows = []
    for ln in lines[i + 1 :]:
        if ln.startswith("#"):
            break
        if ln.startswith("|"):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows[0], rows[2:]


def read_csv(path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def model_of(cell: str) -> str:
    return re.search(r"`([^`]+)`", cell).group(1)


def test_every_headline_cell_is_a_csv_row(tree):
    header, rows = md_table((tree / "report.md").read_text(encoding="utf-8"), "Headline")
    assert len(header) == 1 + len(R.HEADLINE) and header[1:] == [c.header for c in R.HEADLINE] and header[7] == R.H("icd_f1_anon")
    csv_rows = read_csv(tree / "results.csv")
    index = {(r["model"], r["suite"], r["category"], r["metric"], r["coder"], r["level"]): r for r in csv_rows}
    checked = 0
    for row in rows:
        m = model_of(row[0])
        for col, cell in zip(R.HEADLINE, row[1:], strict=True):
            if cell in (R.DASH, "failed", "error"):
                continue
            r = index[(m, "aci_eval", col.category, col.metric, "llm_note" if col.coder == "*" else col.coder, col.level)]
            v = float(r["value"])
            expected = ("yes" if v else "**no**") if col.digits == "bool" else R.fmt(v, col.digits)
            assert cell == expected, (m, col.key, cell, r["value"])
            checked += 1
    assert checked >= 60
    # spot values: full precision in the CSV, 3 decimals (thousands separator) in the markdown
    alpha = dict(zip([c.key for c in R.HEADLINE], next(r for r in rows if model_of(r[0]) == "alpha")[1:], strict=True))
    assert alpha["privacy"] == "0.900" and alpha["privacy_harness"] == "0.912" and alpha["ms_per_encounter"] == "512.2"
    assert alpha["peak_ram_mb"] == "1,234" and alpha["vram_ok"] == "yes" and alpha["attack_max"] == "0.200"
    assert index[("alpha", "aci_eval", "privacy", "removed", "", "gold:harness")]["value"] == "0.9123"
    beta = next(r for r in rows if model_of(r[0]) == "beta")
    assert beta[[c.key for c in R.HEADLINE].index("vram_ok") + 1] == "**no**"
    wide = {r["model"]: r for r in read_csv(tree / "results_headline.csv")}
    assert float(wide["alpha"]["icd_f1_delta"]) == -0.02 and wide["beta"]["vram_ok"] == "0" and wide["identity"]["reference"] == "1"
    assert alpha["icd_f1_delta"] == "-0.020" and alpha["icd_f1_anon"] == "0.380"  # a change is signed; no ratio anywhere
    assert "retention" not in (tree / "results.csv").read_text(encoding="utf-8-sig")
    assert alpha["tree_f1_anon"] == "0.630" and alpha["tree_f1_delta"] == "-0.070"  # the tree at its category level
    assert alpha["recovered_llm"] == R.DASH and beta[[c.key for c in R.HEADLINE].index("recovered_llm") + 1] == "0.200"


def test_pareto_flags(tree):
    t = json.loads((tree / "report.json").read_text(encoding="utf-8"))["tradeoff"]
    assert not t["fallback"] and "llm_note" in t["y"]
    flags = {m: p["pareto"] for m, p in t["points"].items()}
    assert flags["alpha"] is True and flags["beta"] is True and flags["gamma"] is False
    assert flags["alpha_nocodes"] is None and flags["broken"] is None  # no LLM-judge F1 / no score
    assert t["points"]["alpha"]["y"] == 0.38  # plain F1 on the anonymized text
    assert t["points"]["identity"]["reference"] and flags["identity"] is None  # the reference is never in the comparison
    _, rows = md_table((tree / "report.md").read_text(encoding="utf-8"), "Privacy vs ICD utility")
    order = [model_of(r[0]) for r in rows]
    assert order[:3] == ["alpha_nocodes", "alpha", "beta"] and order[-1] == "identity"  # by privacy, the reference last
    pareto = {model_of(r[0]): r[-1] for r in rows}
    assert pareto["alpha"] == "**yes**" and pareto["gamma"] == "no" and pareto["identity"] == "ref" and pareto["alpha_nocodes"] == R.DASH
    ta = json.loads((tree / "report.json").read_text(encoding="utf-8"))["tradeoff_adversarial"]
    assert ta["points"]["beta"]["x"] == 0.8 and ta["points"]["gamma"]["x"] == 0.9 and ta["points"]["alpha"]["x"] is None  # 1 − A4 recovered
    assert ta["points"]["gamma"]["pareto"] and ta["points"]["beta"]["pareto"] and ta["points"]["alpha"]["pareto"] is None
    assert any("LLM adversary" in (x.text or "") for x in ET.parse(tree / "tradeoff_adversarial.svg").getroot().iter("{http://www.w3.org/2000/svg}text"))


def test_pareto_function():
    assert R.pareto({"a": (0.9, 0.95), "b": (0.8, 0.99), "c": (0.7, 0.9)}) == {"a": True, "b": True, "c": False}
    assert R.pareto({"a": (0.9, 0.9), "b": (0.9, 0.9)}) == {"a": True, "b": True}  # a tie dominates nobody
    assert R.pareto({"a": (0.9, 0.9), "b": (0.9, 0.8), "c": (None, 1.0)}) == {"a": True, "b": False, "c": None}


def test_missing_data_and_errors(tree):
    md = (tree / "report.md").read_text(encoding="utf-8")
    _, rows = md_table(md, "Headline")
    cells = {model_of(r[0]): dict(zip([c.key for c in R.HEADLINE], r[1:], strict=True)) for r in rows}
    assert cells["alpha_nocodes"]["icd_f1_anon"] == R.DASH and cells["alpha_nocodes"]["attack_max"] == R.DASH
    assert cells["alpha_nocodes"]["ms_per_encounter"] == "700.0"  # from runs/*/*/run.json
    assert cells["alpha_nocodes"]["explained"] == R.DASH
    assert cells["broken"]["privacy"] == "failed" and cells["broken"]["concepts"] == "failed"
    assert "later" not in cells and "`later`" in md  # disabled, no data: listed as pending
    _, icd_rows = md_table(md, "F1-Score ICD-10 code prediction, every coder")
    by = {(model_of(r[0]), model_of(r[1])): r for r in icd_rows}
    assert by[("beta", "knn")][2] == "error" and "knn index missing" in md
    assert by[("gamma", "llm_note")][1] == "`llm_note` !" and by[("gamma", "llm_note")][-1] == "2"  # not valid for comparison
    assert by[("alpha", "llm_note")][4] == "-0.020 [-0.100, +0.020]" and by[("alpha", "llm_note")][7] == "-0.020 [-0.100, +0.020]"  # code, category
    _, at_rows = md_table(md, "(adversarial attacks, bonus iv.a)")
    at = {model_of(r[0]): r for r in at_rows}
    assert at["alpha"][3] == "error" and at["beta"][3] == R.DASH and at["alpha"][1] == "0.100 / 1.000"  # A2 A3 A4 A5 A6 max
    assert at["beta"][5] == "0.200 / 0.800" and at["alpha"][5] == R.DASH  # the agent adversary
    _, nc_rows = md_table(md, "Without ICD-10 codes")
    assert [R.H("privacy"), "0.900", "0.910", "+0.010"] in nc_rows  # rows = metrics; columns = with codes, without, Δ per pair
    _, rb = md_table(md, "Robustness")
    assert rb[0][:2] == ["`alpha`", "0.600"]
    csv_rows = read_csv(tree / "results.csv")
    assert any(r["metric"] == "error" and r["coder"] == "knn" and r["model"] == "beta" and r["value"] == "" for r in csv_rows)
    for name in ("report.md", "report.json", "report_discord.md", "results.csv", "results_headline.csv", "tradeoff.svg", "tradeoff_adversarial.svg"):
        assert SECRET not in (tree / name).read_text(encoding="utf-8-sig"), name


def test_empty_and_partial_tree(tmp_path, monkeypatch):
    write(tmp_path / "models.toml", MODELS)
    monkeypatch.setattr(R, "OUT", tmp_path / "eval")
    monkeypatch.setattr(R, "MODELS_TOML", tmp_path / "models.toml")
    monkeypatch.setattr(R, "CODERS_TOML", tmp_path / "missing.toml")
    assert R.main([]) == 0  # nothing at all: every cell "–"
    _, rows = md_table((tmp_path / "eval" / "report.md").read_text(encoding="utf-8"), "Headline")
    assert rows and all(c == R.DASH for r in rows for c in r[1:])
    write(tmp_path / "eval" / "scores" / "alpha" / "aci_eval.json", "{not json")  # a file being written
    write(tmp_path / "eval" / "scores" / "beta" / "aci_eval.json", score(0.5))
    assert R.main([]) == 0
    t = json.loads((tmp_path / "eval" / "report.json").read_text(encoding="utf-8"))["tradeoff"]
    assert t["fallback"] and "ICD concepts kept" in t["y"] and t["points"]["beta"]["y"] == 0.99
    assert "fallback" in ET.parse(tmp_path / "eval" / "tradeoff.svg").getroot().find("{http://www.w3.org/2000/svg}text[@transform]").text


def test_csv_bom_and_header(tree):
    raw = (tree / "results.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and raw[3:].decode("utf-8").splitlines()[0] == HEADER
    head = (tree / "results_headline.csv").read_bytes()
    assert head.startswith(b"\xef\xbb\xbf")
    assert head[3:].decode("utf-8").splitlines()[0] == ",".join(["model", "reference"] + [c.key for c in R.HEADLINE] + ["pareto"])
    rows = read_csv(tree / "results.csv")
    assert {r["category"] for r in rows} <= {"privacy", "privacy_by_category", "privacy_by_field", "privacy_by_language", "keep", "concepts", "edits",
                                            "terms", "icd", "attack", "robustness", "drift", "efficiency", "explain", "validity"}  # fmt: skip
    f1 = next(r for r in rows if r["model"] == "alpha" and r["coder"] == "llm_note" and r["metric"] == "anonymized.code.f1")
    assert (f1["value"], f1["ci_low"], f1["ci_high"], f1["n"]) == ("0.38", "0.33", "0.43", "3")
    exports = json.loads((tree / "report.json").read_text(encoding="utf-8"))["exports"]
    assert sorted(exports) == ["report_discord_md", "results_csv", "results_headline_csv", "tradeoff_adversarial_svg", "tradeoff_svg"]
    assert all((tree / p.rsplit("/", 1)[-1]).exists() for p in exports.values())


def test_svg_is_well_formed(tree):
    root = ET.parse(tree / "tradeoff.svg").getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{ns}svg" and root.find(f"{ns}rect").get("fill")  # explicit background
    labels = {t.text for t in root.iter(f"{ns}text")}
    assert {"alpha", "beta", "gamma", "identity"} <= labels
    assert any("alpha_nocodes" in (t.text or "") and "not plotted" in t.text for t in root.iter(f"{ns}text"))
    assert len(root.findall(f"{ns}rect[@class='refm']")) == 1 + 1  # identity and the legend


def test_discord(tree):
    text = (tree / "report_discord.md").read_text(encoding="utf-8")
    assert text.count("\n|") >= 8 and "`alpha`, `beta`" in text and text.count("\n- ") >= 3


def test_glossary_and_sources(tree):
    md = (tree / "report.md").read_text(encoding="utf-8")
    assert md.index("# ACI (`aci_eval`)") < md.index("# Glossary") < md.index("# Sources")
    _, rows = md_table(md.replace("# Glossary", "## Glossary"), "Glossary")
    assert [r[0] for r in rows] == [c.header for c in R.COLUMNS.values()] and all(r[1] for r in rows)
    assert {r[2] for r in rows} >= {"i.a", "ii.a", "ii.b", "iv.a.ii"}
    _, src = md_table(md, "Datasets")
    assert {"`aci_eval`", "`robust_synthetic`"} <= {r[0] for r in src}
    _, tools = md_table(md, "Coders, adversary and detectors")
    assert ["`claude-x` (A6)"] == [r[0] for r in tools if "A6" in r[0]]
