"""Utility metrics: how much ICD-10-relevant information survives anonymization.

1. Medical concept recall (lexicon extractor, independent of the per-encounter shield): ICD-10-CM
   index main terms + drug names + lab/vital names + anatomy, matched in original and anonymized
   text; recall = surviving occurrences / original occurrences.
2. Code-evidence retention: words of each assigned code's description and index synonyms that occur
   in the original text and still occur after anonymization (decisive in the without-codes mode).
3. ICD-10 prediction proxy (local, development-time): zero-shot retrieval with a local embedding
   model (bge-m3 via Ollama) over the label space; top-k codes compared with the gold codes for the
   original and the anonymized text, plus anonymized-vs-original agreement, which isolates the
   damage done by anonymization from the predictor's own quality.
   Optional `llm` mode lets a local LLM pick codes among the retrieved candidates.

The gold codes are used only to score; the predictors never see them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import urllib.request
from functools import cache
from pathlib import Path

from .. import lexicons as L
from ..shield import code_evidence_terms

ANATOMY = frozenset(
    """head neck chest abdomen back spine shoulder shoulders arm arms elbow elbows wrist wrists hand hands finger fingers
    thumb hip hips knee knees leg legs ankle ankles foot feet toe toes heel eye eyes ear ears nose throat mouth tongue
    teeth tooth jaw skin scalp breast breasts heart lung lungs liver kidney kidneys bladder prostate uterus ovary
    ovaries cervix colon bowel stomach esophagus pancreas thyroid brain nerve nerves muscle muscles tendon tendons
    ligament bone bones joint joints groin pelvis buttock thigh calf shin forearm sinus sinuses lymph node nodes
    gallbladder spleen rectum vagina penis testicle scrotum lumbar cervical thoracic sacral meniscus acl rotator cuff
    achilles patella femur tibia fibula humerus radius ulna clavicle scapula rib ribs sternum vertebra disc""".split()
)
LAB_TERMS = frozenset(
    """a1c hemoglobin a1c hba1c creatinine egfr bnp ldl hdl cholesterol triglycerides glucose potassium sodium tsh psa
    wbc platelets ejection fraction inr troponin vitamin d b12 ferritin calcium magnesium bilirubin albumin bun
    lipase hematocrit blood pressure heart rate oxygen saturation temperature bmi weight ekg ecg echocardiogram
    x-ray mri ct scan ultrasound biopsy colonoscopy endoscopy mammogram urinalysis cbc cmp lipid panel""".split()
)
STOP_CONCEPTS = frozenset(
    """history status other unspecified condition disease disorder problem care examination exam test tests result
    results visit follow-up follow up encounter patient person use family personal admission complication
    complications presence absence change changes""".split()
)


@cache
def concept_lexicon() -> frozenset[str]:
    terms: set[str] = set()
    for t in L.icd_main_terms():
        for part in re.split(r",\s*", t):
            part = re.sub(r"\s*\(.*?\)", "", part).strip().lower()
            if 4 <= len(part) <= 40 and part not in STOP_CONCEPTS and len(part.split()) <= 4:
                terms.add(part)
    for d in L.drug_names():
        if len(d) >= 4 and len(d.split()) <= 3 and not L.is_common_word(d, 300):
            terms.add(d)
    terms |= ANATOMY | LAB_TERMS
    # retailers/labelers (kroger, walmart), places, person names and months are not medical concepts
    noise = set(L.places()) | set(L.first_names()) | L.EMPLOYERS | L.BRANDS | {
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
    return frozenset(t for t in terms if t not in noise or t in ANATOMY or t in LAB_TERMS)


def extract_concepts(text: str, skip: list[tuple[int, int]] | None = None) -> dict[str, int]:
    """Multiset of lexicon concepts (longest match first, up to 4 words), ignoring spans in `skip`."""
    lex = concept_lexicon()
    t = text.lower()
    words = [(m.start(), m.end()) for m in re.finditer(r"[a-z0-9][a-z0-9'\-]*", t)]
    out: dict[str, int] = {}
    i = 0
    while i < len(words):
        for n in range(min(4, len(words) - i), 0, -1):
            s0, e0 = words[i][0], words[i + n - 1][1]
            ph = re.sub(r"\s+", " ", t[s0:e0])
            if ph in lex:
                if not skip or not any(a < e0 and s0 < b for a, b in skip):
                    out[ph] = out.get(ph, 0) + 1
                i += n
                break
        else:
            i += 1
    return out


def concept_recall(original: list[dict], anonymized: list[dict], fields=("dialogue", "note"), gold: dict | None = None) -> dict:
    """Share of original concept occurrences still present. Occurrences inside gold PII spans are excluded:
    removing `Mercy General Hospital` or `Kroger` is privacy, not a lost medical concept."""
    from .privacy import pattern

    by_id = {r["encounter_id"]: r for r in anonymized}
    kept = total = 0
    lost: dict[str, int] = {}
    per_field = {f: [0, 0] for f in fields}
    for r in original:
        a = by_id.get(r["encounter_id"])
        if a is None:
            continue
        for f in fields:
            if not isinstance(r.get(f), str):
                continue
            skip = None
            if gold and r["encounter_id"] in gold:
                skip = [(m.start(), m.end()) for it in gold[r["encounter_id"]]["pii"] if it["field"] == f for m in pattern(it["text"]).finditer(r[f])]
            co, ca = extract_concepts(r[f], skip), extract_concepts(a.get(f) or "")
            for c, n in co.items():
                k = min(n, ca.get(c, 0))
                kept += k
                total += n
                per_field[f][0] += k
                per_field[f][1] += n
                if k < n:
                    lost[c] = lost.get(c, 0) + n - k
    return {
        "concept_recall": round(kept / total, 4) if total else None,
        "by_field": {f: round(a / b, 4) if b else None for f, (a, b) in per_field.items()},
        "n_concepts": total,
        "top_lost": sorted(lost.items(), key=lambda x: -x[1])[:25],
    }


def evidence_retention(original: list[dict], anonymized: list[dict]) -> dict:
    by_id = {r["encounter_id"]: r for r in anonymized}
    kept = total = 0
    ext_kept = ext_total = 0
    for r in original:
        a = by_id.get(r["encounter_id"])
        if a is None:
            continue
        terms = code_evidence_terms(r.get("icd10_codes") or [])
        o_text = ((r.get("dialogue") or "") + "\n" + (r.get("note") or "")).lower()
        a_text = ((a.get("dialogue") or "") + "\n" + (a.get("note") or "")).lower()
        for w, code in terms.items():
            rx = re.compile(rf"(?<![a-z]){re.escape(w)}")
            n0 = len(rx.findall(o_text))
            if not n0:
                continue
            n1 = min(n0, len(rx.findall(a_text)))
            kept += n1
            total += n0
            if code[:1].upper() in "VWXY":
                ext_kept += n1
                ext_total += n0
    return {
        "evidence_retention": round(kept / total, 4) if total else None,
        "external_cause_evidence_retention": round(ext_kept / ext_total, 4) if ext_total else None,
        "n_evidence_occurrences": total,
    }


# --------------------------------------------------------------------------- embedding ICD proxy

OLLAMA = "http://localhost:11434"


class EmbedCache:
    def __init__(self, path: Path, model: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists emb (k text primary key, v text)")
        self.model = model

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}\x00{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        have = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = "select k, v from emb where k in (%s)" % ",".join("?" * len(chunk))
            have.update({k: json.loads(v) for k, v in self.db.execute(q, chunk)})
        missing = [(k, t) for k, t in zip(keys, texts) if k not in have]
        for i in range(0, len(missing), 32):
            batch = missing[i : i + 32]
            vecs = _ollama_embed(self.model, [t for _, t in batch])
            for (k, _), v in zip(batch, vecs):
                have[k] = v
                self.db.execute("insert or replace into emb values (?, ?)", (k, json.dumps([round(x, 5) for x in v])))
            self.db.commit()
        return [have[k] for k in keys]


def _ollama_embed(model: str, texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(f"{OLLAMA}/api/embed", data=json.dumps({"model": model, "input": texts}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        vecs = json.loads(resp.read())["embeddings"]
    return [_norm(v) for v in vecs]


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def chunk_text(rec: dict, max_chars: int = 700) -> list[str]:
    chunks: list[str] = []
    for f in ("note", "dialogue"):
        text = rec.get(f) or ""
        parts = [p.strip() for p in re.split(r"\n\s*\n" if f == "note" else r"\n", text) if p.strip()]
        cur = ""
        for p in parts:
            if len(cur) + len(p) > max_chars and cur:
                chunks.append(cur)
                cur = ""
            cur = f"{cur}\n{p}" if cur else p
        if cur:
            chunks.append(cur)
    return [c for c in chunks if re.search(r"[a-z]", c.lower())] or ["empty"]


def code_text(code: str, desc: str) -> str:
    base = code.split(".")[0] if code[:1] in "VWXY" else code
    syn = [p.split(" > ")[-1] for p in L.icd_index_paths().get(base, [])[:4]]
    return f"{desc}" + (f" ({'; '.join(syn)})" if syn else "")


def icd_proxy(original: list[dict], anonymized: list[dict], label_space: dict[str, str], cache_path: Path, model: str = "bge-m3", top: str = "oracle") -> dict:
    """Zero-shot code retrieval; F1 vs gold for both versions and anonymized-vs-original agreement."""
    import numpy as np

    ec = EmbedCache(cache_path, model)
    codes = sorted(label_space)
    cmat = np.array(ec.embed([code_text(c, label_space[c]) for c in codes]), dtype=np.float32)
    by_id = {r["encounter_id"]: r for r in anonymized}

    def predict(rec: dict, k: int) -> list[str]:
        vmat = np.array(ec.embed(chunk_text(rec)), dtype=np.float32)
        sims = cmat @ vmat.T  # codes x chunks
        if sims.shape[1] > 1:
            top2 = np.sort(sims, axis=1)[:, -2:]
            scores = top2.mean(axis=1)
        else:
            scores = sims[:, 0]
        return [codes[i] for i in np.argsort(-scores)[:k]]

    tp = {"orig": 0, "anon": 0, "agree": 0}
    n_pred = n_gold = n_orig_pred = 0
    rows = []
    for r in original:
        a = by_id.get(r["encounter_id"])
        if a is None:
            continue
        gold = {c["code"] for c in r.get("icd10_codes") or []}
        k = max(1, len(gold)) if top == "oracle" else int(top)
        po, pa = set(predict(r, k)), set(predict(a, k))
        tp["orig"] += len(po & gold)
        tp["anon"] += len(pa & gold)
        tp["agree"] += len(pa & po)
        n_pred += len(pa)
        n_orig_pred += len(po)
        n_gold += len(gold)
        rows.append({"encounter_id": r["encounter_id"], "gold": sorted(gold), "orig": sorted(po), "anon": sorted(pa)})

    def f1(t, p, g):
        return round(2 * t / (p + g), 4) if p + g else None

    return {
        "icd_f1_original": f1(tp["orig"], n_orig_pred, n_gold),
        "icd_f1_anonymized": f1(tp["anon"], n_pred, n_gold),
        "icd_agreement_f1": f1(tp["agree"], n_pred, n_orig_pred),
        "k": top,
        "label_space": len(codes),
        "model": model,
        "rows": rows,
    }
