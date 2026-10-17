"""Shield-first anonymization pipeline (no network calls; optional local models only).

Per encounter:
  1. normalize both fields without rewriting (1:1 offsets) and parse speaker tags / sections
  2. build the medical shield (clinical numbers, eponyms, code evidence, drugs)
  3. detect candidates: high-precision rules, quasi-identifier gazetteers, optional local models
  4. identity graph: link every anchored name across both fields and sweep all mentions
  5. drop candidates that touch the shield (logged as KEEP with the shield reason)
  6. policy: placeholders / generalization per category and preset
  7. verify: cross-field leak check -> repair edits (never reverts), shield integrity, schema
  8. emit one decision record per candidate (explanations sidecar)
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict

from .config import Config
from .detect import formats, quasi, rules
from .io import TEXT_FIELDS
from .link import identity
from .policy import placeholders as policy
from .shield import medical as shield
from .spans import NAME_CATS, Span, resolve_overlaps
from .text.normalize import Doc
from .verify.leaks import find_leaks, repair_spans, shield_integrity


def _worth(sp: Span, text: str) -> bool:
    """A piece left after removing shielded words is anonymized only if it still carries an identifier."""
    from . import lexicons as L

    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text[sp.start : sp.end])
    return any(t[:1].isdigit() or not L.is_common_word(t.lower(), 30) for t in toks)


class Anonymizer:
    """Holds configuration and (lazily) the optional local models."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._models: dict = {}
        self.vram_peak_mb = 0.0
        self.vram_events: list[str] = []

    # -- optional detectors -------------------------------------------------------------
    def _model_spans(self, docs: dict[str, Doc]) -> list[Span]:
        out: list[Span] = []
        for name in self.cfg.detectors:
            if name == "rules":
                continue
            det = self._models.get(name)
            if det is None:
                from .detect import encoders as models

                det = models.load(name, self.cfg)
                self._models[name] = det
            for doc in docs.values():
                out += det.detect(doc)
            self._vram_guard()
        return out

    def _vram_guard(self) -> None:
        """Enforce the 4 GB VRAM budget on the *measured* total (torch + local LLM)."""
        from .detect.encoders import VRAM_BUDGET_MB
        from .eval.resources import ollama_vram_mb, torch_vram_mb

        llm = ollama_vram_mb(self.cfg.llm_model) if "llm" in self.cfg.detectors else 0.0
        if llm != llm:  # NaN: Ollama unreachable -> unknown, fail closed with the measured worst case
            llm = getattr(self._models.get("llm"), "vram_mb", 1600.0)
        total = torch_vram_mb() + llm
        self.vram_peak_mb = max(self.vram_peak_mb, total)
        if total > VRAM_BUDGET_MB:
            for name, det in self._models.items():
                if hasattr(det, "to_cpu") and getattr(det, "device", "cpu") != "cpu":
                    det.to_cpu()
                    self.vram_events.append(f"{name} moved to CPU: measured {total:.0f} MB > {VRAM_BUDGET_MB} MB")

    # -- main ---------------------------------------------------------------------------
    def anonymize(self, rec: dict) -> tuple[dict, list[dict], dict]:
        t0 = time.perf_counter()
        cfg = self.cfg
        eid = rec.get("encounter_id", "")
        docs = {f: Doc(f, rec[f]) for f in TEXT_FIELDS if isinstance(rec.get(f), str)}
        raw = {f: d.raw for f, d in docs.items()}
        codes = rec.get("icd10_codes") if cfg.use_codes else None

        shields = [sh for d in docs.values() for sh in shield.build(d, codes, cfg.injury_context_shield)]
        cands: list[Span] = []
        for d in docs.values():
            cands += rules.detect(d) + quasi.detect(d) + formats.detect(d)
        cands += self._model_spans(docs)

        anchors = [c for c in cands if c.cat in NAME_CATS]
        ents = identity.build_entities(anchors, docs, cfg.weak_name_threshold)
        policy.entity_labels(ents)
        ent_by_key = {e.key: e for e in ents}
        tok2ent = {(t, e.role == "clinician"): e.key for e in ents for t in e.tokens}
        for a in anchors:
            tok = raw[a.field][a.start : a.end].lower().removesuffix("'s")
            a.entity = tok2ent.get((tok, a.meta.get("role") == "clinician")) or tok2ent.get(
                (tok, not a.meta.get("role") == "clinician")
            )
        cands = [c for c in cands if c.cat not in NAME_CATS or (c.entity and c.conf >= cfg.weak_name_threshold)]
        cands += identity.sweep(ents, docs)

        decisions: list[dict] = []
        applied: list[Span] = []
        for rnd in range(cfg.repair_rounds + 1):
            kept, active = self._filter(cands, shields, decisions if rnd == 0 else None, raw)
            chosen = resolve_overlaps(active)
            edits: dict[str, list[tuple[int, int, str]]] = {f: [] for f in docs}
            applied = []
            for sp in chosen:
                s, e, rep, action = policy.replacement(sp, cfg, ent_by_key)
                edits[sp.field].append((s, e, rep))
                sp.meta["action"], sp.meta["replacement"] = action, rep
                applied.append(sp)
            anon = {f: policy.apply_edits(raw[f], edits[f]) for f in docs}
            leaks = find_leaks(applied, raw, anon)
            if not leaks or rnd == cfg.repair_rounds:
                break
            covered = {f: [(s, e) for s, e, _ in edits[f]] for f in docs}
            new = repair_spans(leaks, raw, covered, shields)
            if not new:
                break
            cands += new

        residual = find_leaks(applied, raw, anon)
        integrity = shield_integrity(shields, edits, raw, anon)
        for sp in applied:
            decisions.append(self._decision(eid, sp, sp.meta["action"], sp.meta["replacement"], None, raw))

        out = dict(rec)
        for f in docs:
            out[f] = anon[f]
        stats = {
            "encounter_id": eid,
            "runtime_s": round(time.perf_counter() - t0, 4),
            "n_edits": len(applied),
            "n_entities": len(ents),
            "entities": [{"key": e.key, "role": e.role, "label": e.label, "n_tokens": len(e.tokens)} for e in ents],
            "residual_leaks": len(residual),
            "shield": integrity,
        }
        return out, decisions, stats

    def _filter(
        self, cands: list[Span], shields, decisions: list[dict] | None, raw: dict[str, str]
    ) -> tuple[list[Span], list[Span]]:
        """Split candidates into kept (shielded / preset) and active (to anonymize).

        A candidate is kept verbatim only when a hard shield covers it entirely (or, for an age, covers its
        number). Partial overlaps never keep private text: a direct identifier with a clinical number inside
        stays active, and other candidates are split around the shielded words (`Kierland Golf Club` with
        `golf` shielded -> `Kierland` is still removed).
        """
        kept, active = [], []
        for c in cands:
            overl = [s for s in shields if s.field == c.field and s.hard and s.start < c.end and c.start < s.end]
            if overl and not (c.cat in NAME_CATS and c.source.startswith("rule:title")):
                pieces = self._unshielded_pieces(c, overl, raw[c.field])
                if pieces is None or not pieces:
                    kept.append(c)
                    if decisions is not None:
                        sh = overl[0]
                        decisions.append(self._decision("", c, "KEEP", None, f"shield:{sh.kind} ({sh.reason})", None))
                    continue
                if pieces != [c]:
                    for p in pieces:
                        if p.cat in self.cfg.remove:
                            active.append(p)
                    if decisions is not None:
                        decisions.append(
                            self._decision(
                                "",
                                c,
                                "KEEP",
                                None,
                                f"shield:{overl[0].kind} (shielded words kept; the rest is anonymized separately)",
                                None,
                            )
                        )
                    continue
                c = pieces[0]
            if c.cat not in self.cfg.remove:
                kept.append(c)
                if decisions is not None:
                    decisions.append(
                        self._decision("", c, "KEEP", None, f"preset:{self.cfg.preset} keeps {c.cat}", None)
                    )
                continue
            active.append(c)
        return kept, active

    @staticmethod
    def _unshielded_pieces(c: Span, overl, text: str) -> list[Span] | None:
        """None/[] = keep the candidate; [c] = anonymize it whole; otherwise the unshielded sub-spans."""
        from dataclasses import replace

        if any(s.start <= c.start and s.end >= c.end for s in overl):
            return None
        if c.cat == "AGE":
            ns, ne = c.meta.get("num_start", c.start), c.meta.get("num_end", c.end)
            return None if any(s.start < ne and ns < s.end for s in overl) else [c]
        if c.cat in ("ID", "PHONE", "ADDRESS", "DOB", "EMAIL", "URL", "DATE") and all(
            s.kind in ("clinical_number", "drug") and c.start <= s.start and s.end <= c.end for s in overl
        ):
            return [c]  # a number/drug-looking piece inside an identifier does not protect the identifier
        pieces, pos = [], c.start
        for s in sorted(overl, key=lambda x: x.start):
            if s.start > pos:
                pieces.append((pos, s.start))
            pos = max(pos, s.end)
        if pos < c.end:
            pieces.append((pos, c.end))
        trimmed = []
        for a, b in pieces:  # never eat the whitespace/punctuation next to a shielded word
            while a < b and not text[a].isalnum():
                a += 1
            while b > a and not text[b - 1].isalnum():
                b -= 1
            if b > a:
                trimmed.append((a, b))
        pieces = trimmed
        return [
            p for p in (replace(c, start=a, end=b, meta=dict(c.meta)) for a, b in pieces) if _worth(p, text)
        ] or None

    def _decision(self, eid: str, sp: Span, action: str, rep: str | None, keep_reason: str | None, raw) -> dict:
        d = {
            "encounter_id": eid,
            "field": sp.field,
            "start": sp.start,
            "end": sp.end,
            "category": sp.cat,
            "action": action,
            "replacement": rep,
            "source": sp.source,
            "confidence": round(sp.conf, 3),
            "detector_reason": sp.reason,
            "policy_reason": keep_reason or policy.WHY.get(sp.cat, ""),
            "entity": sp.entity,
        }
        d["explanation"] = (
            f"{action} {sp.cat}"
            + (f" -> {rep}" if rep else "")
            + f": {sp.reason} [{sp.source}, conf {sp.conf:.2f}]; "
            + (keep_reason or policy.WHY.get(sp.cat, ""))
        )
        if self.cfg.explain_text and raw is not None:
            d["text"] = raw[sp.field][sp.start : sp.end]
        return d


def anonymize_records(records: list[dict], cfg: Config):
    """Yield (anonymized_record, decisions, stats) for each record, in order."""
    an = Anonymizer(cfg)
    for rec in records:
        out, dec, st = an.anonymize(rec)
        for d in dec:
            d["encounter_id"] = rec.get("encounter_id", "")
        yield out, dec, st


__all__ = ["Anonymizer", "anonymize_records", "asdict"]
