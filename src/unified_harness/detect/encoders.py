"""Optional local model detectors for the residue the rules miss.

- `deid`  : obi/deid_roberta_i2b2 (RoBERTa-large fine-tuned on i2b2 2014 de-identification), CPU.
- `gliner`: knowledgator/gliner-pii-base-v1.0 (zero-shot PII span model), CPU.
- `llm`   : Qwen 3.5 2B Q4_K_M through a local Ollama server, as a second-opinion name/place finder.

All run on-prem. Transformers run on CPU by default (0 VRAM); the LLM is the only GPU user and is
bounded (`num_ctx`) so that the total stays far below the 4 GB VRAM budget (checked at load time).
Every model span passes the same gate as the weak rules: person names must look like names (not a
drug, eponym, scribe product or ordinary word) and then become identity-graph anchors, so a single
model hit is propagated through both fields like any other name.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from .. import lexicons as L
from ..config import Config
from ..spans import Span
from ..text.normalize import Doc
from .rules import ROLE_CAT

# Inference is on-prem: models must come from the local cache, never from the network (forced, not
# defaulted, and every loader also passes local_files_only=True). Local calls bypass any HTTP proxy.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

VRAM_BUDGET_MB = 4096
# Measured peaks on Apple M4 Max (fp16, cache released after every call): obi/deid_roberta_i2b2 1033 MB,
# GLiNER-PII base 1060 MB (batch 2), Qwen 3.5 2B Q4_K_M at num_ctx 4096 1555 MB -> 3.65 GB for all three.
# Used to decide placement before loading; the pipeline also enforces the measured total at runtime.
VRAM_ESTIMATE_MB = {"deid": 1100, "gliner": 1100, "llm": 1600}
SAFETY_MB = 150


def place(detectors: tuple[str, ...], device: str) -> dict[str, str]:
    """Device per transformer detector. `auto`: GPU while the summed estimate fits the 4 GB budget."""
    if device != "auto":
        return {d: device for d in detectors}
    gpu = _gpu()
    used = VRAM_ESTIMATE_MB["llm"] if "llm" in detectors else 0
    out = {}
    for d in ("deid", "gliner"):
        if d not in detectors:
            continue
        if gpu and used + VRAM_ESTIMATE_MB[d] <= VRAM_BUDGET_MB - SAFETY_MB:
            out[d], used = gpu, used + VRAM_ESTIMATE_MB[d]
        else:
            out[d] = "cpu"
    return out


_PEAK_MB = [0.0]


def note_vram() -> float:
    """Sample torch GPU memory now and keep the running peak (call after loads and forward passes)."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return _PEAK_MB[0]
    cur = 0.0
    if torch.cuda.is_available():
        cur = torch.cuda.max_memory_reserved() / 2**20
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            cur = torch.mps.driver_allocated_memory() / 2**20
        except Exception:  # noqa: BLE001
            cur = 0.0
    _PEAK_MB[0] = max(_PEAK_MB[0], cur)
    return _PEAK_MB[0]


def peak_vram_mb() -> float:
    return round(_PEAK_MB[0], 1)


def release_gpu_cache() -> None:
    """Give cached allocator blocks back after each call so the measured peak stays bounded."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _gpu() -> str | None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return None


def _expand(text: str, s: int, e: int) -> tuple[int, int]:
    """Grow a subword-level span to whole words (`ros` -> `rosalind`)."""
    while s > 0 and (text[s - 1].isalnum() or text[s - 1] in "'-") and text[s - 1] != " ":
        s -= 1
    while e < len(text) and (text[e].isalnum() or text[e] == "-"):
        e += 1
    return s, e


def _person_tokens(doc: Doc, s: int, e: int, conf: float, source: str, role: str = "person") -> list[Span]:
    """Split a model PERSON span into gated name-token anchors."""
    out = []
    s, e = _expand(doc.view, s, e)
    for m in re.finditer(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]", doc.view[s:e]):
        ts, te = s + m.start(), s + m.end()
        w = m.group(0).lower().removesuffix("'s")
        if w in L.NAME_STOP or w in L.TITLES or w in L.SCRIBE_PRODUCTS or w in L.RELATIONS or len(w) < 2:
            continue
        if w in L.drug_tokens() or (w in L.MEDICAL_EPONYMS and re.match(L.EPONYM_NEXT, doc.lower[te : te + 25])):
            continue
        cased = doc.field == "note" and doc.view[ts].isupper()
        common = L.is_common_word(w)
        known = w in L.first_names() or w in L.surnames()
        if common and not (cased and known):
            continue
        if not known and (w in L.medical_vocab() or L.lower_uses(w) >= 5):
            continue
        out.append(
            Span(
                doc.field,
                ts,
                te,
                ROLE_CAT.get(role, "PERSON_NAME"),
                source,
                conf,
                f"{source.split(':')[0]} model person span",
                meta={"role": role, "anchor": True},
            )
        )
    return out


CARE_WORDS = (
    frozenset(
        """orthopedics orthopaedics ophthalmology cardiology neurology oncology dermatology urology gastroenterology
    endocrinology rheumatology pulmonology nephrology radiology pediatrics podiatry psychiatry psychology physical therapy
    clinic hospital pharmacy er emergency icu lab labs imaging urgent care culdesac cul-de-sac bars race tracks gym
    gyms office offices school schools work home church store stores""".split()
    )
    | L.STAFF_WORDS
)


def fuzzy_drug(w: str) -> bool:
    """`welixa` (typo of Wixela): one edit away from a drug name of the same length."""
    if len(w) < 6:
        return False
    for d in L.DRUG_EXTRA:
        if len(d) == len(w) and sum(a != b for a, b in zip(d, w, strict=True)) <= 2 and sorted(d) == sorted(w):
            return True
    return False


def _generic(doc: Doc, s: int, e: int, cat: str, conf: float, source: str) -> list[Span]:
    s, e = _expand(doc.view, s, e)
    txt = doc.lower[s:e].strip()
    if len(txt) < 2 or not re.search(r"[a-z0-9]", txt):
        return []
    if cat in ("LOCATION", "FACILITY", "ORGANIZATION"):
        if txt in L.NON_PERSONAL_ORGS or txt in L.GENERIC_PLACES or txt in L.SCRIBE_PRODUCTS:
            return []
        words = re.findall(r"[a-z][a-z'\-]+", txt)
        if any(w in L.drug_names() or w in L.SCRIBE_PRODUCTS for w in words) or all(
            w in L.GENERIC_PLACES or L.is_common_word(w, 30) for w in words
        ):
            return []
        if not words or all(
            L.is_common_word(w, 100) or w in L.medical_vocab() or w in CARE_WORDS or w in L.drug_tokens() for w in words
        ):
            return []  # "the hospital", "home", "ophthalmology", "cul-de-sac"
        if any(fuzzy_drug(w) for w in words):
            return []
    if cat == "DATE" and not (
        re.search(r"\d", txt) or re.search(L.MONTHS + r"\W+(?:\d|the|first|second|third|[a-z]+th)", txt)
    ):
        return []  # month names alone are relative timing, not dates
    if cat == "AGE":
        return []  # the age rules are exact and parse the value; model ages add only noise
    if cat == "ID" and len(re.sub(r"\D", "", txt)) < 4:
        return []  # `ca-125` is a tumour marker, not an identifier
    return [Span(doc.field, s, e, cat, source, conf, f"{source.split(':')[0]} model {cat.lower()} span")]


def _with_device(cfg: Config, device: str) -> Config:
    from dataclasses import replace

    return replace(cfg, device=device)


def local_model(key: str) -> str:
    """Directory of a pinned encoder from models.toml; fails when it is missing or not at the pinned revision."""
    from ..environment import check_hf_model, load_manifest, models_dir

    entry = load_manifest()["huggingface"][key]
    check = check_hf_model(entry, models_dir())
    if not check.ok:
        raise RuntimeError(f"{entry['repo']}: {check.detail}")
    return str(models_dir() / entry["dir"])


class DeidRoberta:
    LABELS = {"PATIENT": "patient", "STAFF": "clinician", "AGE": "AGE", "DATE": "DATE", "HOSP": "FACILITY", "LOC": "LOCATION",
              "PATORG": "ORGANIZATION", "ID": "ID", "PHONE": "PHONE", "EMAIL": "EMAIL", "OTHERPHI": "OTHER_QI"}  # fmt: skip

    def __init__(self, cfg: Config, name: str | None = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = cfg.device
        name = name or local_model("deid_roberta")
        self.tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True, local_files_only=True)
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        # loaded on the CPU in its final dtype, then moved: no fp32 copy ever sits on the GPU
        self.model = AutoModelForTokenClassification.from_pretrained(name, dtype=dtype, local_files_only=True).eval()
        self.model = self.model.to(self.device)
        note_vram()
        self.id2label = self.model.config.id2label
        if self.device == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    def to_cpu(self) -> None:
        self.model = self.model.to("cpu").float()  # move first, then upcast on the CPU (no GPU spike)
        self.device = "cpu"
        release_gpu_cache()

    def detect(self, doc: Doc) -> list[Span]:
        text = doc.view.replace("\x00", " ")
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        win, stride = 500, 400
        probs_tok: list = [None] * len(ids)
        starts = list(range(0, max(1, len(ids) - 100), stride)) or [0]
        batch = [ids[s : s + win] for s in starts]
        with self.torch.inference_mode():
            for bi in range(0, len(batch), 8):
                chunk = batch[bi : bi + 8]
                maxlen = max(len(c) for c in chunk) + 2
                inp = [
                    [self.tok.cls_token_id]
                    + c
                    + [self.tok.sep_token_id]
                    + [self.tok.pad_token_id] * (maxlen - len(c) - 2)
                    for c in chunk
                ]
                att = [[1] * (len(c) + 2) + [0] * (maxlen - len(c) - 2) for c in chunk]
                logits = self.model(
                    input_ids=self.torch.tensor(inp, device=self.device),
                    attention_mask=self.torch.tensor(att, device=self.device),
                ).logits
                p = logits.softmax(-1).cpu()
                if self.device != "cpu":
                    note_vram()
                for j, c in enumerate(chunk):
                    s0 = starts[bi + j]
                    for k in range(len(c)):
                        # keep the prediction from the window where the token is most central
                        if probs_tok[s0 + k] is None or (k >= 50 and s0 + k < s0 + win - 50):
                            probs_tok[s0 + k] = p[j, k + 1]
        spans: list[tuple[int, int, str, float]] = []
        cur = None
        for i, pr in enumerate(probs_tok):
            if pr is None:
                continue
            conf, lab = float(pr.max()), self.id2label[int(pr.argmax())]
            if lab == "O" or conf < 0.5:
                cur = None
                continue
            tag, typ = lab.split("-", 1)
            if tag in ("B", "U") or cur is None or cur[2] != typ:
                cur = [offs[i][0], offs[i][1], typ, conf]
                spans.append(cur)
            else:
                cur[1], cur[3] = offs[i][1], min(cur[3], conf)
            if tag in ("L", "U"):
                cur = None
        if self.device != "cpu":
            release_gpu_cache()
        merged: list[list] = []
        for sp in spans:  # B/I/L fragments of one entity separated by subword noise
            if merged and merged[-1][2] == sp[2] and sp[0] - merged[-1][1] <= 1:
                merged[-1][1], merged[-1][3] = sp[1], min(merged[-1][3], sp[3])
            else:
                merged.append(list(sp))
        out: list[Span] = []
        for s, e, typ, conf in merged:
            m = self.LABELS.get(typ)
            if m in ("patient", "clinician"):
                out += _person_tokens(doc, s, e, min(0.9, conf), "deid:" + typ.lower(), m)
            elif m:
                out += _generic(doc, s, e, m, min(0.85, conf), "deid:" + typ.lower())
        return out


class Gliner:
    # No "occupation"/"company": on this data they mostly tag care-team roles (`therapist`) and activities (`tennis`).
    LABELS = {"person": "person", "doctor": "clinician", "location": "LOCATION", "city": "LOCATION", "organization": "ORGANIZATION",
              "hospital": "FACILITY", "date": "DATE", "medical record number": "ID", "phone number": "PHONE",
              "email": "EMAIL", "street address": "ADDRESS"}  # fmt: skip

    def __init__(self, cfg: Config, name: str | None = None, threshold: float = 0.5):
        from gliner import GLiNER

        path = name or local_model("gliner_pii")
        self.model = GLiNER.from_pretrained(path, map_location="cpu", local_files_only=True)
        if cfg.device in ("mps", "cuda"):
            import torch

            self.model = self.model.to(dtype=torch.float16).to(device=cfg.device)  # fp32 weights never on the GPU
            note_vram()
        self.threshold = threshold
        self.device = cfg.device
        self.batch_size = 2 if cfg.device in ("mps", "cuda") else 8

    def to_cpu(self) -> None:
        self.model = self.model.to("cpu").float()
        self.device, self.batch_size = "cpu", 8
        release_gpu_cache()

    def detect(self, doc: Doc) -> list[Span]:
        text = doc.view.replace("\x00", " ")
        chunks, cur_s, pos = [], 0, 0
        for ln in text.split("\n"):
            end = pos + len(ln)
            if end - cur_s > 1200:
                chunks.append((cur_s, pos))
                cur_s = pos
            pos = end + 1
        chunks.append((cur_s, len(text)))
        out: list[Span] = []
        texts = [text[s:e] for s, e in chunks]
        preds = self.model.inference(texts, list(self.LABELS), threshold=self.threshold, batch_size=self.batch_size)
        if self.device != "cpu":
            note_vram()
            release_gpu_cache()
        for (s0, _), ents in zip(chunks, preds, strict=False):
            for ent in ents:
                s, e, m = s0 + ent["start"], s0 + ent["end"], self.LABELS[ent["label"]]
                conf = float(ent["score"])
                if m in ("person", "clinician"):
                    out += _person_tokens(doc, s, e, min(0.85, conf), "gliner:" + ent["label"], m)
                else:
                    out += _generic(doc, s, e, m, min(0.8, conf), "gliner:" + ent["label"].replace(" ", "_"))
        return out


LLM_PROMPT = """You find private information in a medical text (a lowercase speech transcript or a clinical note).
List, copied EXACTLY as written in the text, every:
- "names": first or last name of any real person (patient, doctor, nurse, relative, friend), including nicknames
- "places": city, state, country, street, named facility or landmark
- "orgs": employer, company, school, insurer, sports team
Do NOT list diseases, drugs, body parts, tests, eponyms (lyme, parkinson's), titles (dr, mr) or the words dax/dragon.
Answer with JSON only: {"names": [], "places": [], "orgs": []}

TEXT:
"""


class OllamaLLM:
    """Second-opinion finder; its suggestions are gated and must occur verbatim in the chunk."""

    def __init__(self, cfg: Config):
        self.model = cfg.llm_model
        self.num_ctx = cfg.llm_num_ctx
        from ..environment import ollama_url

        self.url = f"{ollama_url()}/api/generate"
        self._call("ok", 8)  # load once and check the VRAM budget
        from ..eval.resources import ollama_vram_mb

        used = ollama_vram_mb(self.model)
        if used > VRAM_BUDGET_MB:
            raise RuntimeError(f"{self.model} uses {used} MB VRAM > budget {VRAM_BUDGET_MB} MB; lower llm_num_ctx")
        self.vram_mb = used

    def _call(self, prompt: str, num_predict: int = 400) -> str:
        body = {
            "model": self.model, "prompt": prompt, "stream": False, "think": False, "format": "json", "keep_alive": "10m",
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": num_predict, "seed": 0},
        }  # fmt: skip
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with LOCAL_HTTP.open(req, timeout=300) as resp:  # never through a proxy: the prompt contains patient text
            return json.loads(resp.read()).get("response", "")

    def detect(self, doc: Doc) -> list[Span]:
        text = doc.view.replace("\x00", " ")
        out: list[Span] = []
        pos, n = 0, len(text)
        while pos < n:
            end = min(n, pos + 2400)
            if end < n:
                nl = text.rfind("\n", pos + 1200, end)
                end = nl if nl > 0 else end
            chunk = text[pos:end]
            try:
                ans = json.loads(self._call(LLM_PROMPT + chunk))
            except Exception:  # noqa: BLE001  (malformed JSON: the chunk keeps the rule decisions)
                ans = {}
            low = chunk.lower()
            for key, cat in (("names", "PERSON"), ("places", "LOCATION"), ("orgs", "ORGANIZATION")):
                for item in ans.get(key, []) if isinstance(ans.get(key), list) else []:
                    if not isinstance(item, str) or len(item.strip()) < 2:
                        continue
                    needle = item.strip().lower()
                    for m in re.finditer(r"(?<![a-z])" + re.escape(needle) + r"(?![a-z])", low):
                        s, e = pos + m.start(), pos + m.end()
                        if cat == "PERSON":
                            out += _person_tokens(doc, s, e, 0.7, "llm:name")
                        else:
                            out += _generic(doc, s, e, cat, 0.65, f"llm:{key}")
            pos = end
        return out


TARGETED_PROMPT_VERSION = "f2-v1"  # the prompt text; segment selection changes do not invalidate the cache
TARGETED_PROMPT = """You find details that could identify a patient in an excerpt of a medical encounter (a lowercase
speech transcript or a clinical note). List, copied EXACTLY as written in the excerpt, every:
- "names": first or last name of any real person (patient, doctor, relative, friend), including nicknames
- "places": city, town, state, country, street, neighborhood, named facility or landmark
- "orgs": employer, company, school, church, club, military unit, sports team or league
- "jobs": occupation or job title of the patient or a relative (for example "mail carrier", "retired teacher")
Do NOT list diseases, symptoms, drugs, body parts, tests, eponyms (lyme, parkinson's), medical staff roles
(doctor, nurse, medical assistant), titles (dr, mr) or the words dax/dragon. Lists may be empty.
Answer with JSON only: {"names": [], "places": [], "orgs": [], "jobs": []}

EXCERPT:
"""
# Sentences that carry personal context (work, home, family, school, travel, community, hobbies).
_PERSONAL_CUE = re.compile(
    r"\b(?:work\w*|job|jobs|employ\w*|retir\w*|occupation|career|profession\w*|business|company|factory|office|"
    r"school|college|university|student|teach\w*|team|league|coach\w*|church|live[sd]?|living|moved|from|"
    r"grew up|born|hometown|travel\w*|trip|visit\w*|vacation|married|husband|wife|son|daughter|kids|grand\w*|"
    r"military|army|navy|served|veteran|volunteer\w*|hobby|hobbies|enjoy\w*|i'm a|i am a|as a|"
    r"family|brother|sister|mother|father|mom|dad|own|owns|shop|store|pharmacy|market|restaurant|fan|"
    r"street|road|avenue|corner|downtown|town|city|neighborhood|going to|went to)\b"
)
_WORD = re.compile(r"[a-z][a-z']{3,}")


def _rare_word(unit: str) -> bool:
    """A word that is not ordinary English, not medical and not a drug: a likely proper name
    ("walgreens", "colorado") in the lowercase transcript."""
    return any(
        not L.is_common_word(w, 30)
        and w not in L.medical_vocab()
        and w not in L.drug_tokens()
        and w not in L.drug_names()
        for w in _WORD.findall(unit)
    )


_UNIT = re.compile(r"[^\n.?!]*[\n.?!]?")
_TURN = re.compile(r"[^\n]*\n?")
STAFF_ROLES = frozenset(
    "doctor physician nurse surgeon medical assistant scribe resident attending pa np provider therapist".split()
)


def personal_segments(doc: Doc, max_chars: int = 1500) -> list[tuple[int, int]]:
    """Offsets of the privacy-dense parts of a field (D3): sentences or turns with a personal-context
    cue or a rare non-medical word, the social-history section of a note, and the first and last two
    turns of a dialogue."""
    t = doc.lower
    unit_re = _TURN if doc.field == "dialogue" else _UNIT  # the dialogue is tokenized (" . "): use turns
    units = [(m.start(), m.end()) for m in unit_re.finditer(t) if t[m.start() : m.end()].strip()]
    social = [(s, e) for s, e, name in doc.sections if "SOCIAL" in name] if doc.field == "note" else []
    edge = set(range(2)) | set(range(len(units) - 2, len(units))) if doc.field == "dialogue" else set()
    picked = [
        (s, e)
        for i, (s, e) in enumerate(units)
        if i in edge or _PERSONAL_CUE.search(t[s:e]) or _rare_word(t[s:e]) or any(a <= s < b for a, b in social)
    ]
    merged: list[tuple[int, int]] = []
    for s, e in picked:
        if merged and s - merged[-1][1] <= 1 and e - merged[-1][0] <= max_chars:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


class TargetedLLM(OllamaLLM):
    """F2: the LLM reads only the privacy-dense segments and also returns occupations (D3, D4, D12).

    Replies are cached by (model digest, prompt version, segment text) under outputs/cache, so policy
    changes are re-measured without new model calls. A reply that is not valid JSON stops the run.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        from .. import paths
        from ..environment import load_manifest

        entry = load_manifest().get("ollama", {}).get("detector", {})
        self.digest = entry["digest"] if entry.get("name") == self.model else self.model
        self.cache_path = paths.cache_dir() / "llm_replay.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["answer"]
        self.calls = self.hits = 0

    def _answer(self, chunk: str) -> dict:
        import hashlib

        key = hashlib.sha256(f"{self.digest}|{TARGETED_PROMPT_VERSION}|{chunk}".encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        raw = self._call(TARGETED_PROMPT + chunk, num_predict=600)
        try:
            ans = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"targeted LLM returned invalid JSON ({len(raw)} chars); run stopped") from exc
        ans = ans if isinstance(ans, dict) else {}
        self.calls += 1
        self.cache[key] = ans
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "answer": ans}, ensure_ascii=False) + "\n")
        return ans

    def detect(self, doc: Doc) -> list[Span]:
        text = doc.view.replace("\x00", " ")
        out: list[Span] = []
        for pos, end in personal_segments(doc):
            chunk = text[pos:end]
            ans = self._answer(chunk)
            low = chunk.lower()
            for key, cat in (
                ("names", "PERSON"),
                ("places", "LOCATION"),
                ("orgs", "ORGANIZATION"),
                ("jobs", "OCCUPATION"),
            ):
                items = ans.get(key)
                for item in items if isinstance(items, list) else []:
                    if not isinstance(item, str) or len(item.strip()) < 3:
                        continue
                    needle = item.strip().lower()
                    if cat == "OCCUPATION" and (
                        needle in STAFF_ROLES
                        or all(w in L.medical_vocab() or w in STAFF_ROLES for w in re.findall(r"[a-z']+", needle))
                    ):
                        continue
                    for m in re.finditer(r"(?<![a-z])" + re.escape(needle) + r"(?![a-z])", low):
                        s, e = pos + m.start(), pos + m.end()
                        if cat == "PERSON":
                            out += _person_tokens(doc, s, e, 0.7, "llm:name")
                        else:
                            out += _generic(doc, s, e, cat, 0.65, f"llm:{key}")
        return out


def load(name: str, cfg: Config):
    dev = place(cfg.detectors, cfg.device).get(name, "cpu")
    if name == "deid":
        return DeidRoberta(_with_device(cfg, dev))
    if name == "gliner":
        return Gliner(_with_device(cfg, dev))
    if name == "llm":
        return TargetedLLM(cfg) if "F2" in cfg.features else OllamaLLM(cfg)
    raise ValueError(f"unknown detector {name!r} (rules, deid, gliner, llm)")
