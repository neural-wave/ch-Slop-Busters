"""serdarcaglar/medprocner-codiesp-spanish-medical-texts: 3,751 Spanish clinical case reports (CodiEsp corpus).

Each `.txt` becomes a `note`. This upload has no labels. The ICD-10-CM diagnosis codes of the 1,000 annotated
CodiEsp cases are on Zenodo ("CodiEsp corpus", CC BY 4.0): download the `codiesp-D` TSVs
(`<articleID>\\t<code>` per line) and pass them with `--codes` to get `icd10_source: gold` on those cases.
Descriptions are the English ICD-10-CM 2023 ones (empty for codes no longer in that release). No PII gold:
published cases are already de-identified but keep ages, dates and places.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, dotted, encounter, icd10cm, parser, sample, write_dataset  # noqa: E402


def load_codes(paths: list[Path]) -> dict[str, list[str]]:
    codes: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2 and dotted(parts[1]) not in codes[parts[0]]:
                codes[parts[0]].append(dotted(parts[1]))
    return codes


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "serdarcaglar__medprocner-codiesp-spanish-medical-texts" / "medprocner-codiesp")
    p.add_argument("--codes", type=Path, nargs="*", default=[], help="CodiEsp-D TSV files from Zenodo")
    a = p.parse_args(argv)
    codes = load_codes(a.codes)
    unknown: set[str] = set()

    def items():
        for path in sample(sorted(a.src.glob("*.txt")), a.limit, a.seed):
            art = path.stem
            entries = []
            for c in codes.get(art, []):
                desc = icd10cm().get(c)
                if desc is None:
                    unknown.add(c)
                entries.append({"code": c, "description": desc or ""})
            enc = encounter(f"codiesp-{art}", "codiesp", note=clean(path.read_text(encoding="utf-8")), codes=entries)
            yield enc, Labels(enc).record("gold" if entries else "none", pii_gold="none", meta={"article_id": art})

    recs = list(items())
    write_dataset("codiesp", recs, {
        "source": "kaggle:serdarcaglar/medprocner-codiesp-spanish-medical-texts (CodiEsp, Miranda-Escalada et al. 2020)",
        "language": "es",
        "origin": "real published clinical case reports",
        "license": "unknown on Kaggle (CodiEsp: CC BY 4.0)",
        "icd10": f"gold from {[str(x) for x in a.codes]}; {len(unknown)} codes not in ICD-10-CM 2023 (empty description)" if codes else "none (pass --codes)",
        "pii_gold": {"labelled": None, "notes": "ages, dates, places present but unlabelled"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
