"""Text helpers of the two lexical coders (knn, lexical): content terms, the clinical vocabulary, the ACI train set."""

from __future__ import annotations

from functools import cache

from eval.common import CONVERTED, load_jsonl, words
from eval.metrics import clinical_vocabulary

STOP = frozenset(
    """the and that this with have has had was were are for you your yeah okay just like about what know there they
    them then than when going right also some she her his him not but can will would could should been being does
    did doing get got all any how out our ours one two very really well yes from into over more much here thing
    things let think want see said says say um uh mm hmm patient doctor today time back good little bit kind sure
    because which where who whom why these those its it's i'm don't that's we're you're""".split()
)


def stem(t: str) -> str:
    return t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t


def terms(text: str, vocab: frozenset[str] | None = None) -> list[str]:
    """Lowercased, crudely stemmed content words (3+ letters, no stop word, no number), optionally only `vocab`."""
    out = []
    for t, _, _ in words(text):
        if len(t) < 3 or t in STOP or t.isdigit():
            continue
        t = stem(t)
        if vocab is None or t in vocab:
            out.append(t)
    return out


@cache
def clinical_stems() -> frozenset[str]:
    return frozenset(stem(t) for t in clinical_vocabulary())


def record_text(rec: dict) -> str:
    return f"{rec.get('note') or ''}\n{rec.get('dialogue') or ''}"


@cache
def aci() -> tuple[list[dict], list[dict]]:
    """(the 140 ACI train encounters, every ACI encounter) from data/converted/aci_bench."""
    encs = load_jsonl(CONVERTED / "aci_bench" / "encounters.jsonl")
    labs = load_jsonl(CONVERTED / "aci_bench" / "labels.jsonl")
    train = [e for e, lab in zip(encs, labs) if lab["meta"].get("split") == "train"]
    return train, encs
