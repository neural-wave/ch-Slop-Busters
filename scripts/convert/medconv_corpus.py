"""thedevastator/medical-conversation-corpus-100k: ChatDoctor / HealthCareMagic Q&A (106,556 train + 5,609 test).

`[|Human|]` / `[|AI|]` turns become `[patient]` / `[doctor]` dialogue lines. Default: the test split (5,609);
`--split train|all` and `--limit` for more. Real user posts with names, ages, countries, all unlabelled (the
upstream anonymization was crude and left "Chat Doctor" artifacts in the text). No ICD-10, no PII gold.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, dialogue_text, encounter, parser, read_csv, sample, write_dataset  # noqa: E402

TURN = re.compile(r"\[\|(Human|AI)\|\](.*?)(?=\[\|(?:Human|AI)\|\]|\Z)", re.S)
SPEAKER = {"Human": "patient", "AI": "doctor"}


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "thedevastator__medical-conversation-corpus-100k")
    p.add_argument("--split", choices=["test", "train", "all"], default="test")
    a = p.parse_args(argv)
    rows = [(split, i, r) for split in (["train", "test"] if a.split == "all" else [a.split])
            for i, r in enumerate(read_csv(a.src / f"{split}.csv"))]  # fmt: skip

    def items():
        for split, i, r in sample(rows, a.limit, a.seed):
            dialogue = dialogue_text((SPEAKER[s], t) for s, t in TURN.findall(r["Conversation"]))
            if not dialogue:
                continue
            enc = encounter(f"medconv-{split}-{i:06d}", "medconv_corpus", dialogue=clean(dialogue))
            yield enc, Labels(enc).record("none", pii_gold="none", meta={"split": split, "row": i})

    write_dataset("medconv_corpus", items(), {
        "source": "kaggle:thedevastator/medical-conversation-corpus-100k",
        "language": "en",
        "origin": "real user questions + doctor answers (HealthCareMagic via ChatDoctor)",
        "license": "CC0 (doubtful: scraped)",
        "icd10": "none",
        "pii_gold": {"labelled": None, "notes": f"split={a.split}; real names/ages/places present but unlabelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
