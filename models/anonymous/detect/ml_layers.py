"""Layers 2+3: i2b2-trained de-identification transformer and zero-shot GLiNER span model."""
from __future__ import annotations

import functools
import os

from common import NEVER_PHI, Span

DEID_MODEL = os.environ.get("NW_DEID_MODEL", "obi/deid_roberta_i2b2")
GLINER_MODEL = os.environ.get("NW_GLINER_MODEL", "urchade/gliner_multi_pii-v1")
DEVICE = os.environ.get("NW_DEVICE", "cpu")

DEID_MAP = {"AGE": "AGE", "DATE": "DATE", "EMAIL": "CONTACT", "PHONE": "CONTACT", "ID": "ID", "HOSP": "ORGANIZATION",
            "PATORG": "ORGANIZATION", "LOC": "LOCATION", "PATIENT": "NAME_PATIENT", "STAFF": "NAME_CLINICIAN",
            "OTHERPHI": "PERSONAL_DETAIL"}
GLINER_LABELS = {
    "person name": "NAME_OTHER", "organization": "ORGANIZATION", "company": "ORGANIZATION", "hospital": "ORGANIZATION",
    "school": "ORGANIZATION", "sports team": "ORGANIZATION", "city": "LOCATION", "country": "LOCATION",
    "state": "LOCATION", "address": "LOCATION", "landmark": "LOCATION", "occupation": "OCCUPATION",
    "job title": "OCCUPATION", "workplace": "OCCUPATION", "hobby": "ACTIVITY", "sport": "ACTIVITY",
    "leisure activity": "ACTIVITY", "trip or vacation": "LIFE_EVENT", "life event": "LIFE_EVENT",
    "holiday": "TEMPORAL", "nationality": "PERSONAL_DETAIL", "religion": "PERSONAL_DETAIL", "pet": "PERSONAL_DETAIL",
}


def chunks(text: str, max_chars: int):
    """Split at line boundaries (dialogue turns / note lines) into <= max_chars pieces, with offsets."""
    out, start, cur = [], 0, 0
    for line in text.splitlines(keepends=True):
        if cur - start + len(line) > max_chars and cur > start:
            out.append((start, text[start:cur]))
            start = cur
        while len(line) > max_chars:  # very long single line: hard split on spaces
            cut = line.rfind(" ", 0, max_chars) or max_chars
            out.append((cur, text[cur:cur + cut]))
            cur += cut
            start = cur
            line = line[cut:]
        cur += len(line)
    if cur > start:
        out.append((start, text[start:cur]))
    return out


@functools.lru_cache(maxsize=1)
def _deid():
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DEID_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(DEID_MODEL).to(DEVICE).eval()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    return tok, model


def deid_detect(field: str, text: str, min_conf: float = 0.5) -> list[Span]:
    return _deid_with_offsets(field, text, min_conf)


def _deid_with_offsets(field, text, min_conf):
    import torch
    tok, model = _deid()
    id2label = model.config.id2label
    out = []
    for off, piece in chunks(text, 1500):
        enc = tok(piece, return_offsets_mapping=True, truncation=True, max_length=512, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            probs = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits[0].softmax(-1).cpu()
        conf, ids = probs.max(-1)
        ents, cur = [], None
        for (s, e), i, c in zip(offsets, ids.tolist(), conf.tolist()):
            if s == e:
                continue
            lab = id2label[i]
            typ = None if lab == "O" else lab.split("-", 1)[1]
            if cur and typ == cur[2] and piece[cur[1]:s].strip() == "":
                cur = [cur[0], e, typ, min(cur[3], c)]
                continue
            if cur:
                ents.append(cur)
            cur = [s, e, typ, c] if typ else None
        if cur:
            ents.append(cur)
        for s, e, typ, c in ents:
            frag = piece[s:e]
            if c < min_conf or frag.strip().lower() in NEVER_PHI or len(frag.strip()) < 2:
                continue
            out.append(Span(field, s + off, e + off, frag, DEID_MAP.get(typ, "PERSONAL_DETAIL"), "ner", round(c, 3),
                            f"i2b2 de-id transformer tagged {typ} (p={c:.2f})"))
    return out


@functools.lru_cache(maxsize=1)
def _gliner():
    from gliner import GLiNER
    return GLiNER.from_pretrained(GLINER_MODEL).to(DEVICE)


def gliner_detect(field: str, text: str, threshold: float = 0.45, labels=None) -> list[Span]:
    model = _gliner()
    labels = labels or list(GLINER_LABELS)
    out = []
    for off, piece in chunks(text, 700):
        for ent in model.predict_entities(piece, labels, threshold=threshold):
            frag = ent["text"]
            if frag.strip().lower() in NEVER_PHI or len(frag.strip()) < 2:
                continue
            typ = GLINER_LABELS.get(ent["label"], "PERSONAL_DETAIL")
            out.append(Span(field, ent["start"] + off, ent["end"] + off, frag, typ, "gliner", round(ent["score"], 3),
                            f"GLiNER zero-shot label '{ent['label']}' (p={ent['score']:.2f})"))
    return out
