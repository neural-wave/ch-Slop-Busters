"""alejopaullier/pii-external-dataset: 4,434 LLM-written first-person essays with injected PII, token-labelled.

Each essay becomes a `note` (not medical: tests the generic PII layer). Token BIO labels (NAME_STUDENT, EMAIL,
PHONE_NUM, STREET_ADDRESS, USERNAME, URL_PERSONAL) become character spans; the injected job and hobby (columns,
not token-labelled) are added by string match as OCCUPATION / OTHER_QI, except the junk hobby "Related". No ICD-10.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, PII_LABELS, Labels, bio_spans, encounter, parser, read_csv, sample, write_dataset  # noqa: E402

# Upstream job/hobby values that are not one: hobby "Related" (159 rows) only ever matches the "related" of "job-related".
NOT_A_VALUE = {"related"}


def align(text: str, tokens: list[str]) -> list[tuple[int, int]] | None:
    """Character offsets of `tokens` in `text`, or None if they skip over non-space text."""
    offsets, pos = [], 0
    for t in tokens:
        if not t.strip():
            offsets.append((pos, pos))
            continue
        i = text.find(t, pos)
        if i < 0 or text[pos:i].strip():
            return None
        offsets.append((i, i + len(t)))
        pos = i + len(t)
    return offsets


def joined(tokens: list[str], spaces: list[bool]) -> tuple[str, list[tuple[int, int]]]:
    """spaCy-style reconstruction: token + one space where trailing_whitespace is True."""
    parts, offsets, pos = [], [], 0
    for t, ws in zip(tokens, spaces):
        offsets.append((pos, pos + len(t)))
        parts.append(t + (" " if ws else ""))
        pos += len(parts[-1])
    return "".join(parts), offsets


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "alejopaullier__pii-external-dataset" / "pii_dataset.csv")
    a = p.parse_args(argv)
    rows = list(enumerate(read_csv(a.src)))
    rebuilt = 0

    def items():
        nonlocal rebuilt
        for i, r in sample(rows, a.limit, a.seed):
            tokens, labels = ast.literal_eval(r["tokens"]), ast.literal_eval(r["labels"])
            text, offsets = r["text"], align(r["text"], tokens)
            if offsets is None:  # text and tokens disagree: use the token reconstruction so spans stay exact
                text, offsets = joined(tokens, ast.literal_eval(r["trailing_whitespace"]))
                rebuilt += 1
            enc = encounter(f"piiext-{i:05d}", "pii_external", note=text)
            lab = Labels(enc)
            for s, e, typ in bio_spans(tokens, labels, offsets):
                lab.add_pii("note", s, e, PII_LABELS[typ], typ)
            for col, cat in (("job", "OCCUPATION"), ("hobby", "OTHER_QI")):
                value = r[col].strip()
                if value and value.lower() not in NOT_A_VALUE:
                    lab.match_pii(value, cat, col, fields=["note"])
            yield enc, lab.record("none", pii_gold="partial", meta={"document": r["document"], "prompt_id": r["prompt_id"]})

    recs = list(items())
    write_dataset("pii_external", recs, {
        "source": "kaggle:alejopaullier/pii-external-dataset",
        "language": "en",
        "origin": "LLM-generated essays with injected synthetic PII (not medical)",
        "license": "Apache 2.0",
        "icd10": "none",
        "pii_gold": {"labelled": "PATIENT_NAME (essay author), ID (email, phone, username, URL), LOCATION (street address), OCCUPATION, OTHER_QI (hobby)",
                     "notes": f"other people's names, cities, dates are unlabelled; {rebuilt} essays use the token reconstruction as text"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
