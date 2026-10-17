"""Second ICD-10 utility proxy: a local LLM selects codes among retrieved candidates.

Retrieval (bge-m3 embeddings, see `utility.icd_proxy`) proposes the 15 most similar codes of the label
space for each version of the text separately; the LLM then picks the codes supported by that text.
Scored like the organizers' metric (F1 against the gold codes) for original and anonymized text,
plus anonymized-vs-original agreement. Development-time only; never part of the pipeline.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from .utility import EmbedCache, chunk_text, code_text

PROMPT = """You are a certified medical coder. Read the clinical encounter and select the ICD-10-CM codes from the
candidate list that are supported by the text (diagnoses addressed in this encounter, including external-cause /
activity codes when the injury circumstances are described). Answer JSON only: {{"codes": ["..."]}}

CANDIDATES:
{cands}

ENCOUNTER:
{text}
"""


def _ask(model: str, prompt: str) -> list[str]:
    body = {"model": model, "prompt": prompt, "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 150, "seed": 0}}  # fmt: skip
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        try:
            ans = json.loads(json.loads(resp.read()).get("response", "{}"))
        except json.JSONDecodeError:
            return []
    codes = ans.get("codes", []) if isinstance(ans, dict) else []
    return [str(c).strip().upper() for c in codes if isinstance(c, str)]


def icd_llm_proxy(
    original, anonymized, label_space: dict[str, str], cache_path: Path, model: str | None = None, k: int = 15
) -> dict:
    model = model or "qwen3.5:2b-q4_K_M"
    ec = EmbedCache(cache_path, "bge-m3")
    codes = sorted(label_space)
    cmat = np.array(ec.embed([code_text(c, label_space[c]) for c in codes]), dtype=np.float32)
    by_id = {r["encounter_id"]: r for r in anonymized}

    def predict(rec: dict) -> list[str]:
        vmat = np.array(ec.embed(chunk_text(rec)), dtype=np.float32)
        sims = cmat @ vmat.T
        scores = np.sort(sims, axis=1)[:, -2:].mean(axis=1) if sims.shape[1] > 1 else sims[:, 0]
        cands = [codes[i] for i in np.argsort(-scores)[:k]]
        text = (rec.get("note", "")[:5000] + "\n\nDIALOGUE (excerpt):\n" + rec.get("dialogue", "")[:4000]).strip()
        listing = "\n".join(f"{c}: {label_space[c]}" for c in cands)
        picked = _ask(model, PROMPT.format(cands=listing, text=text))
        norm = {re.sub(r"[^A-Z0-9]", "", c): c for c in cands}
        return sorted({norm[re.sub(r"[^A-Z0-9]", "", p)] for p in picked if re.sub(r"[^A-Z0-9]", "", p) in norm})

    tp_o = tp_a = agree = n_o = n_a = n_g = 0
    rows = []
    for r in original:
        a = by_id.get(r["encounter_id"])
        if a is None:
            continue
        gold = {c["code"] for c in r.get("icd10_codes") or []}
        po, pa = set(predict(r)), set(predict(a))
        tp_o += len(po & gold)
        tp_a += len(pa & gold)
        agree += len(po & pa)
        n_o, n_a, n_g = n_o + len(po), n_a + len(pa), n_g + len(gold)
        rows.append({"encounter_id": r["encounter_id"], "gold": sorted(gold), "orig": sorted(po), "anon": sorted(pa)})

    def f1(t, p, g):
        return round(2 * t / (p + g), 4) if p + g else None

    return {"icd_f1_original": f1(tp_o, n_o, n_g), "icd_f1_anonymized": f1(tp_a, n_a, n_g),
            "icd_agreement_f1": f1(agree, n_a, n_o),
            "model": model, "candidates_k": k, "rows": rows}  # fmt: skip
