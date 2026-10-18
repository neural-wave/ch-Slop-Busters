"""Records, splits, silver targets and prompts for SFT and PPO (TRAINING_PLAN.md §3, §5, §9).

  python3 rl/data.py splits             write rl/splits.json (silver-val: 14 train encounters; dev 30 / test 37 eval)
  python3 rl/data.py audit [--tokenizer models/Qwen3.5-2B]   token lengths of prompts and silver targets
  python3 rl/data.py show D2N070        the prompt and the silver target of one train encounter

Sources: data/patient_doctor_encounters.jsonl + ../term-generalizer/gold/train_gold.jsonl (silver), data/synth_external_cause.jsonl
+ ../term-generalizer/gold/synth_gold.jsonl (rl/synth.py), and for validation only data/eval_... + ../term-generalizer/gold/eval_gold.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "term-generalizer")]
import generalize as G  # noqa: E402
from common import DATA_DIR, EVAL_DATA, GOLD_EVAL, ROOT, read_jsonl  # noqa: E402

import augment as A  # noqa: E402
import render as R  # noqa: E402
from spec import (NAME_TYPES, SYSTEM_PROMPT, dump_output, dx_prefix, short_reason, user_message)  # noqa: E402

TRAIN_DATA = DATA_DIR / "patient_doctor_encounters.jsonl"
TRAIN_GOLD = ROOT.parent / "term-generalizer" / "gold" / "train_gold.jsonl"
SYNTH_DATA = DATA_DIR / "synth_external_cause.jsonl"
SYNTH_GOLD = ROOT.parent / "term-generalizer" / "gold" / "synth_gold.jsonl"
SPLITS = Path(__file__).resolve().parent / "splits.json"
MODEL_DIR = ROOT / "models" / "Qwen3.5-2B"
SECTION_TITLE = {"dialogue": "DIALOGUE", "note": "PHYSICIAN NOTE"}


def load(data_path, gold_path) -> tuple[dict, dict]:
    recs = {r["encounter_id"]: r for r in read_jsonl(data_path)} if Path(data_path).exists() else {}
    golds = {g["encounter_id"]: g for g in read_jsonl(gold_path)} if Path(gold_path).exists() else {}
    return recs, golds


# ---------------------------------------------------------------- splits

def _stratified(ids: list[str], key, n: int, seed: int) -> list[str]:
    """n ids, allocated to strata in proportion to their size (largest remainder), random within a stratum."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i in sorted(ids):
        groups[key(i)].append(i)
    quota = {k: n * len(v) / len(ids) for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k in sorted(groups):
        out += rng.sample(groups[k], alloc[k])
    return sorted(out)


def make_splits(seed: int = 13) -> dict:
    train, _ = load(TRAIN_DATA, TRAIN_GOLD)
    ev, _ = load(EVAL_DATA, GOLD_EVAL)
    silver_val = _stratified(list(train), lambda i: train[i]["dataset"], 14, seed)
    vy = lambda i: any(c["code"][:1] in "VWXY" for c in ev[i]["icd10_codes"])  # noqa: E731
    dev = _stratified(list(ev), lambda i: (ev[i]["dataset"], vy(i)), 30, seed)
    return {"seed": seed, "silver_val": silver_val, "train": sorted(set(train) - set(silver_val)),
            "dev": dev, "test": sorted(set(ev) - set(dev))}


def splits() -> dict:
    if not SPLITS.exists():
        raise SystemExit(f"{SPLITS} missing: run python3 rl/data.py splits")
    return json.loads(SPLITS.read_text())


# ---------------------------------------------------------------- silver target

def _auto_ctx(text: str, rec: dict, protected: dict) -> list[str]:
    """Longer verbatim phrases around the private uses of a string that also occurs inside medical text."""
    out = []
    for f in R.FIELDS:
        t = rec.get(f) or ""
        for s, e in R.find_all(text, t):
            if R._inside((s, e), protected.get(f, [])):
                continue
            left = re.search(r"(\S+\s+)?$", t[max(0, s - 40):s]).group(0)
            right = re.match(r"(\s+\S+)?", t[e:e + 40]).group(0)
            for ctx in (left + t[s:e], t[s:e] + right, left + t[s:e] + right):
                ctx = ctx.strip()
                if "\n" in ctx or ctx.lower() == text.lower():
                    continue
                occ = R.find_all(ctx, t)
                if occ and not any(R._overlaps(iv, protected.get(f, [])) for iv in occ):
                    if ctx.lower() not in (c.lower() for c in out):
                        out.append(ctx)
                    break
    return out[:3]


def silver_target(rec: dict, gold: dict, *, max_evidence: int = 4) -> dict:
    """The output the policy should write for this record: dx from icd10_codes, evidence from the silver links,
    entities from silver PII (one per distinct string across fields, `ctx` where it collides with medical text),
    keep from non-PII ambiguous terms and regex spans that are not PII."""
    dx = [{"code": c["code"], "desc": str(c.get("description", "")).lower()} for c in rec.get("icd10_codes") or []]
    ev_by_code = {e["code"]: e for e in gold.get("evidence", []) or []}
    evidence = []
    for c in dx:
        spans, seen = [], set()
        for s in (ev_by_code.get(c["code"]) or {}).get("spans", []):
            if s["text"].lower() not in seen and R.find_all(s["text"], rec.get(s["field"]) or ""):
                seen.add(s["text"].lower())
                spans.append(s["text"])
        evidence.append({"code": c["code"], "spans": spans[:max_evidence]})

    protected = defaultdict(list)
    for c in gold.get("medical_concepts", []):
        protected[c["field"]] += R.find_all(c["text"], rec.get(c["field"]) or "")
    for t in gold.get("ambiguous_terms", []):
        if not t.get("is_pii"):
            protected[t["field"]] += R.find_all(t["text"], rec.get(t["field"]) or "")
    terms = R.terms_of([d["desc"] for d in dx])

    # first occurrence order across the encounter (dialogue, then note)
    def first_pos(e):
        offs = 0 if e["field"] == "dialogue" else len(rec.get("dialogue") or "") + 1
        occ = R.find_all(e["text"], rec.get(e["field"]) or "")
        return offs + (occ[0][0] if occ else 0)

    entities, seen = [], set()
    for e in sorted(gold.get("pii", []), key=first_pos):
        t, cat = e["text"], e["category"]
        if t.lower() in seen or len(t.split()) > 6 or "\n" in t:
            continue
        seen.add(t.lower())
        item = {"text": t, "type": cat, "reason": short_reason(e.get("comment")) or cat.lower().replace("_", " ")}
        occ = {f: R.find_all(t, rec.get(f) or "") for f in R.FIELDS}
        collides = cat not in R.UNPROTECTED_TYPES and (
            R.is_term(t, terms) or any(R._inside(iv, protected[f]) for f in R.FIELDS for iv in occ[f]))
        if collides:
            for ctx in _auto_ctx(t, rec, protected):
                entities.append({**item, "ctx": ctx})
            continue
        entities.append(item)
    for ent in entities:   # move "ctx" before "type" for readability: text, ctx, type, reason
        if "ctx" in ent:
            ctx = ent.pop("ctx")
            rest = {k: ent.pop(k) for k in ("type", "reason")}
            ent.update({"ctx": ctx, **rest})

    keep, seen_k = [], set()
    pii_norms = {e["text"].lower() for e in gold.get("pii", [])}
    for t in gold.get("ambiguous_terms", []):
        if not t.get("is_pii") and t["text"].lower() not in seen_k and t["text"].lower() not in pii_norms:
            seen_k.add(t["text"].lower())
            keep.append({"text": t["text"], "reason": short_reason(t.get("comment")) or "not private"})
    pii_ivs = defaultdict(list)
    for e in gold.get("pii", []):
        pii_ivs[e["field"]] += R.find_all(e["text"], rec.get(e["field"]) or "")
    for f in R.FIELDS:
        for s in G.find_spans(rec.get(f) or ""):
            if s["text"].lower() in seen_k:
                continue
            if not any(R._overlaps(iv, pii_ivs[g]) for g in R.FIELDS for iv in R.find_all(s["text"], rec.get(g) or "")):
                seen_k.add(s["text"].lower())
                keep.append({"text": s["text"], "reason": "clinical, not a personal " + s["type"].lower()})
    return {"dx": dx, "evidence": evidence, "entities": entities, "keep": keep}


# ---------------------------------------------------------------- prompts

@dataclass
class Sample:
    eid: str
    record: dict              # the (augmented) record the model sees
    gold: dict                # silver labels of that record
    prefill: bool
    sections: list
    known: list = field(default_factory=list)
    ops: list = field(default_factory=list)

    def messages(self) -> list[dict]:
        secs = [(SECTION_TITLE[s], self.record.get(s) or "") for s in self.sections]
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(secs, self.known)}]

    def prefix(self) -> str:
        return dx_prefix(self.record.get("icd10_codes") or []) if self.prefill else ""

    def prompt_text(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(self.messages(), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) + self.prefix()

    def target(self) -> str:
        """The silver response, without the prefilled prefix."""
        full = dump_output(silver_target(self.record, self.gold))
        p = self.prefix()
        assert full.startswith(p), "the prefix must be a prefix of the serialized target"
        return full[len(p):]


def pii_density(gold: dict) -> int:
    return len(gold.get("pii", []))


class Sampler:
    """Fresh augmentation on every draw; ~70% PII-dense prompts, ~30% PII-free windows (TRAINING_PLAN.md §5).

    With synth_ids and synth_share >= 0, a draw comes from the synthetic records with probability synth_share and from
    the others otherwise (uniform within each group); synth_share < 0 draws uniformly from all ids.
    """

    def __init__(self, ids: list[str], records: dict, golds: dict, *, seed: int = 0, p_prefill: float = 0.5,
                 p_dense: float = 0.7, max_prompt_tokens: int = 6144, tokenizer=None, augment: bool = True,
                 synth_ids=(), synth_share: float = -1.0):
        self.ids = [i for i in ids if i in records and i in golds]
        synth = set(synth_ids)
        self.groups = [([i for i in self.ids if i in synth], synth_share),
                       ([i for i in self.ids if i not in synth], 1.0 - synth_share)]
        if synth_share < 0 or not all(g for g, _ in self.groups):
            self.groups = [(self.ids, 1.0)]
        self.records, self.golds = records, golds
        self.rng = random.Random(seed)
        self.p_prefill, self.p_dense, self.max_prompt = p_prefill, p_dense, max_prompt_tokens
        self.tok, self.do_augment = tokenizer, augment
        self.pools = A.build_pools(records, golds)

    def draw(self) -> Sample:
        want_clean = self.do_augment and self.rng.random() >= self.p_dense
        for attempt in range(60):
            group = self.rng.choices([g for g, _ in self.groups], weights=[w for _, w in self.groups])[0]
            eid = self.rng.choice(group)
            rec, g = self.records[eid], self.golds[eid]
            if self.do_augment:
                clean = want_clean and attempt < 50   # PII-free windows are short; give up after 50 tries
                rec, g, meta = A.augment(rec, g, self.rng, self.pools, p_window=1.0 if clean else 0.3,
                                         window_tokens=self.rng.choice([128, 256, 512]) if clean else 512,
                                         window_max_tokens=1024 if clean else None)
                if clean and pii_density(g):
                    continue   # retry until the window is PII-free
            else:
                meta = {"sections": ["dialogue", "note"], "known": [], "ops": []}
            s = Sample(eid, rec, g, self.rng.random() < self.p_prefill, meta["sections"], meta["known"], meta["ops"])
            if self.tok is None or len(self.tok(s.prompt_text(self.tok))["input_ids"]) <= self.max_prompt:
                return s
        return s


def pii_free_window_rate(sampler: Sampler, n: int = 200) -> float:
    return sum(pii_density(sampler.draw().gold) == 0 for _ in range(n)) / n


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["splits", "audit", "show"])
    ap.add_argument("eid", nargs="?")
    ap.add_argument("--tokenizer", default=str(MODEL_DIR))
    args = ap.parse_args()
    if args.cmd == "splits":
        s = make_splits()
        SPLITS.write_text(json.dumps(s, indent=1) + "\n")
        print({k: len(v) if isinstance(v, list) else v for k, v in s.items()})
        return
    recs, golds = load(TRAIN_DATA, TRAIN_GOLD)
    if args.cmd == "show":
        s = Sample(args.eid, recs[args.eid], golds[args.eid], True, ["dialogue", "note"])
        print(s.messages()[1]["content"][:1500], "\n...\n--- prefix\n", s.prefix(), "\n--- target\n", s.target())
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    srecs, sgolds = load(SYNTH_DATA, SYNTH_GOLD)
    recs.update(srecs)
    golds.update(sgolds)
    rows = []
    for eid in sorted(golds):
        for prefill in (False, True):
            s = Sample(eid, recs[eid], golds[eid], prefill, ["dialogue", "note"])
            rows.append((eid, prefill, len(tok(s.prompt_text(tok))["input_ids"]), len(tok(s.target())["input_ids"])))
    for prefill in (False, True):
        r = [x for x in rows if x[1] == prefill]
        p, t = sorted(x[2] for x in r), sorted(x[3] for x in r)
        print(f"prefill={prefill}: n={len(r)} prompt mean {sum(p) / len(p):.0f} max {p[-1]} | target mean "
              f"{sum(t) / len(t):.0f} p95 {t[int(.95 * len(t))]} max {t[-1]}")
    for eid, pf, pl, tl in sorted(rows, key=lambda x: -x[3])[:5]:
        print(f"  longest target: {eid} prefill={pf} prompt {pl} target {tl}")


if __name__ == "__main__":
    main()
