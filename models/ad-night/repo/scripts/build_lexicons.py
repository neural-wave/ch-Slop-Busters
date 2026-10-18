"""Build the compact lexicons shipped in `src/neural_wave/data/` from public raw sources.

Run once (development time, needs the raw downloads listed in README "Rebuilding lexicons"):

    uv run python scripts/build_lexicons.py

Sources (all public):
- ICD-10-CM 2026 order file, alphabetic index, external-cause index, drug table (CDC/CMS)
- FDA NDC product file (drug proprietary / non-proprietary / substance names)
- US Census 2010 surnames; US baby names 1880-2008 (hadley/data-baby-names, SSA derived)
- GeoNames cities15000, admin1 codes, country info
- Kaggle `thedevastator/medical-conversation-corpus-100k` (cased English medical Q&A) for word
  frequencies: how often a token is used *lowercase* in cased text tells us whether a name is
  also an ordinary word (`will`, `grace`, `white`) — the collision signal the pipeline uses.

Outputs are gzip TSV/text files, a few MB in total, read lazily by `neural_wave.lexicons`.
"""

from __future__ import annotations

import csv
import gzip
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "resources/external/raw"
OUT = ROOT / "src/neural_wave/data"
MEDCONV = Path.home() / ".cache/kagglehub/datasets/thedevastator/medical-conversation-corpus-100k/versions/2/train.csv"


def wgz(name: str, rows: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / name, "wt", encoding="utf-8", compresslevel=9) as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"[lexicon] {name}: {len(rows)} rows, {(OUT / name).stat().st_size / 1e3:.0f} kB")


def dotted(code: str) -> str:
    return code if len(code) <= 3 else code[:3] + "." + code[3:]


def build_icd() -> None:
    rows = []
    for line in (RAW / "icd/icd10cm-order-2026.txt").open(encoding="latin-1"):
        code, billable, desc = line[6:13].strip(), line[14], line[77:].strip()
        rows.append(f"{dotted(code)}\t{billable}\t{desc}")
    wgz("icd10cm_2026.tsv.gz", rows)


def _title(el: ET.Element) -> str:
    """Index title text without the non-essential modifiers in <nemod>."""
    t = el.find("title")
    if t is None:
        return ""
    text = (t.text or "") + "".join((c.tail or "") for c in t)
    return re.sub(r"\s+", " ", text).strip(" ,")


def build_index() -> None:
    """code -> index paths (lay synonyms), and the set of index main terms."""
    paths: list[str] = []
    main_terms: set[str] = set()
    for fname, kind in (("icd10cm_index_2026.xml", "dx"), ("icd10cm_eindex_2026.xml", "ext")):
        root = ET.parse(RAW / "icd/Table and Index" / fname).getroot()

        def walk(el: ET.Element, prefix: list[str]) -> None:
            title = _title(el)
            chain = prefix + [title] if title else prefix
            code = el.findtext("code")
            if code:
                paths.append(f"{code.strip()}\t{kind}\t{' > '.join(chain)}")
            for sub in el.findall("term"):
                walk(sub, chain)

        for mt in root.iter("mainTerm"):
            t = _title(mt)
            if t:
                main_terms.add(t.lower())
            walk(mt, [])
    wgz("icd_index_paths.tsv.gz", sorted(set(paths)))
    wgz("icd_index_main_terms.txt.gz", sorted(main_terms))


def build_drugs() -> None:
    names: set[str] = set()

    def add(s: str) -> None:
        for part in re.split(r",|;| and |/", s.lower()):
            part = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 \-']", " ", part)).strip()
            if 3 <= len(part) <= 40 and len(part.split()) <= 4 and re.search(r"[a-z]{3}", part):
                names.add(part)

    with (RAW / "product.txt").open(encoding="latin-1") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if "HUMAN" not in r["PRODUCTTYPENAME"]:
                continue
            for k in ("PROPRIETARYNAME", "NONPROPRIETARYNAME", "SUBSTANCENAME"):
                add(r[k] or "")
    root = ET.parse(RAW / "icd/Table and Index/icd10cm_drug_2026.xml").getroot()
    for mt in root.iter("mainTerm"):
        add(_title(mt))
    wgz("drugs.txt.gz", sorted(names))
    # Single-token drug names: drop ordinary English words (`shield`, `clear`, `relief` are NDC product names).
    english = set()
    for dict_path in (Path("/usr/share/dict/words"), Path("/usr/share/dict/web2")):
        if dict_path.exists():
            english |= {w.strip().lower() for w in dict_path.open(encoding="utf-8", errors="ignore")}
    toks = {t for d in names for t in d.split() if len(t) >= 5 and t.isalpha() and t not in english}
    wgz("drug_tokens.txt.gz", sorted(toks))


def build_names() -> None:
    score: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    with (RAW / "baby-names.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["year"]) < 1925:
                continue
            score[r["name"].lower()][0 if r["sex"] == "boy" else 1] += float(r["percent"])
    rows = [f"{n}\t{b + g:.5f}\t{b / (b + g):.2f}" for n, (b, g) in score.items() if b + g >= 0.0005]
    rows.sort(key=lambda s: -float(s.split("\t")[1]))
    wgz("first_names.tsv.gz", rows)

    rows = []
    with (RAW / "Names_2010Census.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["name"] == "ALL OTHER NAMES":
                continue
            c = int(r["count"])
            if c >= 800:
                rows.append(f"{r['name'].lower()}\t{c}")
    wgz("surnames.tsv.gz", rows)


def build_wordfreq() -> None:
    """Lowercase vs capitalised usage counts of every word in a cased medical corpus."""
    if not MEDCONV.exists():
        sys.exit(f"missing {MEDCONV}; download the Kaggle medical-conversation-corpus-100k first")
    csv.field_size_limit(10**9)
    lower, cap = Counter(), Counter()
    tok = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
    sent_start = re.compile(r"(?:^|[.!?]\s+|\]\s*)$")
    with MEDCONV.open(encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            text = r.get("Conversation") or ""
            for m in tok.finditer(text):
                w = m.group(0)
                if w.islower():
                    lower[w] += 1
                elif w[0].isupper() and w[1:].islower():
                    # sentence-initial capitals carry no name signal
                    if not sent_start.search(text[max(0, m.start() - 3) : m.start()]):
                        cap[w.lower()] += 1
    words = {w for w, c in lower.items() if c >= 3} | {w for w, c in cap.items() if c >= 3}
    rows = [f"{w}\t{lower[w]}\t{cap[w]}" for w in words]
    rows.sort(key=lambda s: -int(s.split("\t")[1]))
    wgz("wordfreq_medconv.tsv.gz", rows)


def build_places() -> None:
    admin1 = {}
    for line in (RAW / "admin1CodesASCII.txt").open(encoding="utf-8"):
        code, name, ascii_name, _ = line.rstrip("\n").split("\t")
        admin1[code] = ascii_name
    rows = set()
    for line in (RAW / "countryInfo.txt").open(encoding="utf-8"):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        rows.add(f"{f[4].lower()}\tcountry\t{f[7] or 0}\t{f[0]}")
    for code, name in admin1.items():
        cc = code.split(".")[0]
        if cc in {"US", "IN", "CA", "GB", "AU", "MX"}:
            rows.add(f"{name.lower()}\tstate\t0\t{cc}")
    for line in (RAW / "cities15000.txt").open(encoding="utf-8"):
        f = line.split("\t")
        name, cc, pop = f[2], f[8], int(f[14] or 0)
        if (cc in {"US", "IN"} and pop >= 30000) or pop >= 300000:
            rows.add(f"{name.lower()}\tcity\t{pop}\t{cc}")
    wgz("places.tsv.gz", sorted(rows))


if __name__ == "__main__":
    build_icd()
    build_index()
    build_drugs()
    build_names()
    build_places()
    build_wordfreq()
