"""cnjameshu/nbme-patient-notes: 1,000 USMLE Step 2 CS patient notes (10 cases x 100), shipped tokenized.

The note text is rebuilt from the tokens (one space wherever the original offsets had a gap, so some characters
such as the `-` in `3-4` and the `/` in `c/o` are lost upstream). The token labels mark the clinical findings each
case is graded on (`B-<feature>`); they become `keep` spans (CLINICAL_FEATURE): ground truth for medical concept
recall. The exception is each case's age feature (`17yo`, `44 year old`): age is PII (AGE) in the harness taxonomy,
so those spans become `pii` instead (pii_gold partial: relatives' ages such as "father had MI at 52" are unlabelled). Sex (`M`, `female`)
stays a feature. Case number = pn_num // 10000. No ICD-10.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, bio_spans, encounter, parser, read_csv, sample, write_dataset  # noqa: E402

# The patient's age, one feature per case (case 0: "17 yo", case 2: "44 year old", ...), found by reading each
# feature's spans. The other features are findings, durations, history and sex.
AGE_FEATURES = {"11", "104", "216", "315", "409", "517", "602", "707", "805", "901"}


def rebuild(tokens: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    parts, offsets, pos, prev_end = [], [], 0, 0
    for t in tokens:
        if parts and int(t["pos_bos"]) > prev_end:
            parts.append(" ")
            pos += 1
        parts.append(t["vocab"])
        offsets.append((pos, pos + len(t["vocab"])))
        pos += len(t["vocab"])
        prev_end = int(t["pos_eos"])
    return "".join(parts), offsets


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "cnjameshu__nbme-patient-notes" / "nbme_data.csv")
    a = p.parse_args(argv)
    notes: dict[int, list[dict]] = defaultdict(list)
    for r in read_csv(a.src):
        notes[int(r["pn_num"])].append(r)

    def items():
        for pn in sample(sorted(notes), a.limit, a.seed):
            toks = notes[pn]
            text, offsets = rebuild(toks)
            case = pn // 10000
            enc = encounter(f"nbme-{pn:05d}", "nbme_notes", note=text)
            lab = Labels(enc)
            for s, e, feature in bio_spans([t["vocab"] for t in toks], [t["label"] for t in toks], offsets):
                if feature in AGE_FEATURES:
                    lab.add_pii("note", s, e, "AGE", f"nbme case {case} feature {feature} (age)", trim=False)
                else:
                    lab.add_keep("note", s, e, "CLINICAL_FEATURE", f"nbme case {case} feature {feature}", trim=False)
            yield enc, lab.record("none", pii_gold="partial" if lab.pii else "none", meta={"pn_num": pn, "case_num": case})

    write_dataset("nbme_notes", items(), {
        "source": "kaggle:cnjameshu/nbme-patient-notes (NBME Score Clinical Patient Notes)",
        "language": "en",
        "origin": "USMLE Step 2 CS exam notes on standardized patients",
        "license": "unknown",
        "icd10": "none",
        "pii_gold": {"labelled": "AGE (each case's graded age feature)",
                     "notes": "keep spans = the other graded clinical features (concept-recall gold); relatives' ages unlabelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
