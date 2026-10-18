"""Phase 1 - structure and format.

Schema, lengths, per-dataset differences, speaker-tag inventory, note
section-header inventory, bullet formatting, non-ASCII / control characters.
Writes outputs/eda/headers_by_dataset.json.
"""
import statistics
import unicodedata

from common import *

recs = load_all()
S = by_split(recs)

# ---------------------------------------------------------------- schema
head("1.1 Schema")
keysets = Counter(tuple(sorted(k for k in r if k != "split")) for r in recs)
print("distinct key sets:", dict(keysets))
types = Counter((k, type(v).__name__) for r in recs for k, v in r.items() if k != "split")
print("key/type pairs:", dict(types))
bad_codes = [r["encounter_id"] for r in recs if not all(set(c) == {"code", "description"} for c in r["icd10_codes"])]
print("icd10 entries with keys != {code,description}:", bad_codes)
dup_ids = [k for k, v in Counter(r["encounter_id"] for r in recs).items() if v > 1]
print("duplicate encounter_id across splits:", dup_ids)
print("encounter_id pattern:", Counter(re.sub(r"\d", "9", r["encounter_id"]) for r in recs))
print("empty fields:", [(r["encounter_id"], k) for r in recs for k in ("dialogue", "note") if not r[k].strip()])

head("1.2 Counts per split x dataset")
per_dataset_table(recs, len, "n")

# ---------------------------------------------------------------- lengths
head("1.3 Lengths (chars) dialogue / note  [median (min-max)]")
def med(xs): return f"{int(statistics.median(xs))} ({min(xs)}-{max(xs)})" if xs else "-"
for field in ("dialogue", "note"):
    print(f"-- {field}")
    per_dataset_table(recs, lambda rs: med([len(r[field]) for r in rs]) if rs else "-", "chars")
print("-- dialogue turns (lines)")
per_dataset_table(recs, lambda rs: med([r["dialogue"].count("\n") + 1 for r in rs]) if rs else "-", "turns")
print("-- note lines")
per_dataset_table(recs, lambda rs: med([r["note"].count("\n") + 1 for r in rs]) if rs else "-", "lines")
print("-- words dialogue")
per_dataset_table(recs, lambda rs: med([len(r["dialogue"].split()) for r in rs]) if rs else "-", "words")
print("-- words note")
per_dataset_table(recs, lambda rs: med([len(r["note"].split()) for r in rs]) if rs else "-", "words")

head("1.4 ICD-10 codes")
per_dataset_table(recs, lambda rs: med([len(r["icd10_codes"]) for r in rs]) if rs else "-", "codes/enc")
for split in ("train", "eval"):
    codes = Counter(c["code"] for r in S[split] for c in r["icd10_codes"])
    print(f"{split}: {sum(codes.values())} code slots, {len(codes)} distinct; top: {codes.most_common(8)}")
    n_counts = Counter(len(r["icd10_codes"]) for r in S[split])
    print(f"   codes per encounter: {dict(sorted(n_counts.items()))}")
# same code, different description?
desc_by_code = defaultdict(set)
for r in recs:
    for c in r["icd10_codes"]:
        desc_by_code[c["code"]].add(c["description"])
multi = {k: v for k, v in desc_by_code.items() if len(v) > 1}
print("codes with >1 distinct description:", len(multi))
for k, v in list(multi.items())[:6]:
    print("  ", k, "->", v)
# duplicate codes within an encounter
dups = [(r["encounter_id"], [c for c, n in Counter(x["code"] for x in r["icd10_codes"]).items() if n > 1]) for r in recs]
dups = [d for d in dups if d[1]]
print("encounters with a repeated code:", dups[:10], "... total", len(dups))
# code format
print("code shapes:", Counter(re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", c["code"])) for r in recs for c in r["icd10_codes"]))

# ---------------------------------------------------------------- speaker tags
head("1.5 Speaker tags (dialogue)")
tag_counter = Counter()
tag_enc = defaultdict(set)
untagged = []
weird_tags = []
for r in recs:
    for sp, txt, off in dialogue_lines(r["dialogue"]):
        if sp is None:
            untagged.append((r["encounter_id"], txt[:80]))
        else:
            tag_counter[sp] += 1
            tag_enc[sp].add(r["encounter_id"])
            if not re.fullmatch(r"(doctor|patient|patient_guest)", sp):
                weird_tags.append((r["encounter_id"], sp, txt[:60]))
for t, n in tag_counter.most_common():
    print(f"  [{t}]: {n} lines in {len(tag_enc[t])} encounters")
print("lines without a leading tag:", len(untagged), untagged[:8])
print("non-standard tags:", weird_tags[:10])
pg_by_ds = Counter(r["dataset"] + "/" + r["split"] for r in recs if "[patient_guest]" in r["dialogue"])
print("encounters with [patient_guest] per dataset/split:", dict(pg_by_ds))
# any tag not at line start / multiple tags on one line
inline_tags = [(r["encounter_id"], m.group(0)) for r in recs for m in re.finditer(r"(?<!^)(?<!\n)\[(doctor|patient|patient_guest)\]", r["dialogue"])]
print("tags not at line start:", len(inline_tags), inline_tags[:5])
# bracketed artifacts other than speaker tags
brack = Counter()
brack_ex = defaultdict(list)
for r in recs:
    for m in re.finditer(r"\[(?!doctor\]|patient\]|patient_guest\])[^\]]*\]", r["dialogue"]):
        key = re.sub(r"\d", "9", m.group(0))
        brack[key] += 1
        if len(brack_ex[key]) < 3:
            brack_ex[key].append((r["encounter_id"], ctx(r["dialogue"], m.start(), m.end(), 30)))
print("other bracketed artifacts in dialogue:")
for k, n in brack.most_common():
    print(f"  {k!r}: {n}  e.g. {brack_ex[k]}")
brack_note = Counter(re.sub(r"\d", "9", m.group(0)) for r in recs for m in re.finditer(r"\[[^\]]*\]", r["note"]))
print("bracketed things in note:", brack_note.most_common(10))
# casing in dialogue
upper_in_dialogue = Counter()
for r in recs:
    for m in re.finditer(r"\b[A-Z][A-Za-z]*\b", r["dialogue"]):
        upper_in_dialogue[m.group(0)] += 1
print("tokens with an uppercase letter in dialogue:", sum(upper_in_dialogue.values()), upper_in_dialogue.most_common(15))
# turn structure: consecutive same speaker?
same = sum(1 for r in recs for a, b in zip(dialogue_lines(r["dialogue"]), dialogue_lines(r["dialogue"])[1:]) if a[0] == b[0])
print("consecutive lines with same speaker:", same)
# empty turns
empty_turns = [(r["encounter_id"], sp) for r in recs for sp, txt, _ in dialogue_lines(r["dialogue"]) if not txt.strip()]
print("empty turns:", len(empty_turns), empty_turns[:5])

# ---------------------------------------------------------------- note headers
head("1.6 Note section headers")
hdr_by_ds = {}
hdr_ex = defaultdict(list)
order_by_ds = defaultdict(Counter)
empty_sections = []
no_header_notes = []
pre_header_text = []
for ds in DATASETS:
    cnt = Counter(); enc = defaultdict(set)
    for r in recs:
        if r["dataset"] != ds:
            continue
        secs = note_sections(r["note"])
        hs = [h for h, _, _ in secs if h]
        if not hs:
            no_header_notes.append(r["encounter_id"])
        order_by_ds[ds][tuple(hs)] += 1
        for h, body, off in secs:
            if h is None:
                if body.strip():
                    pre_header_text.append((r["encounter_id"], body.strip()[:70]))
                continue
            cnt[h] += 1; enc[h].add(r["encounter_id"])
            if not body.strip():
                empty_sections.append((r["encounter_id"], h))
            if len(hdr_ex[h]) < 2:
                hdr_ex[h].append(r["encounter_id"])
    n = len([r for r in recs if r["dataset"] == ds])
    hdr_by_ds[ds] = {h: {"lines": c, "encounters": len(enc[h]), "pct_encounters": round(100 * len(enc[h]) / n, 1)} for h, c in cnt.most_common()}
    print(f"-- {ds} ({n} notes): {len(cnt)} distinct headers")
    for h, c in cnt.most_common():
        print(f"   {h:<45} {c:>4} lines  {pct(len(enc[h]), n)}")
print("notes with no header at all:", no_header_notes)
print("text before first header:", len(pre_header_text), pre_header_text[:5])
print("empty sections:", len(empty_sections), empty_sections[:10])
print("-- most common header sequences per dataset")
for ds in DATASETS:
    print(f"   {ds}: {len(order_by_ds[ds])} distinct sequences")
    for seq, n in order_by_ds[ds].most_common(3):
        print(f"      x{n}: {' > '.join(seq)}")
# variants grouped by first word / normalized
norm = defaultdict(set)
for ds in DATASETS:
    for h in hdr_by_ds[ds]:
        key = re.sub(r"[^A-Z]", "", h.split()[0]) if h.split() else h
        norm[key].add(h)
print("-- header spelling variants (same first word):")
for k, v in sorted(norm.items()):
    if len(v) > 1:
        print(f"   {k}: {sorted(v)}")
# header lines with trailing colon or mixed case headers
colon = sum(1 for r in recs for line in r["note"].split("\n") if HEADER_RE.match(line.strip()) and line.strip().endswith(":"))
print("header lines ending with ':':", colon)
mixed = Counter()
for r in recs:
    for line in r["note"].split("\n"):
        s = line.strip()
        if re.fullmatch(r"[A-Z][a-z]+( [A-Za-z]+){0,3}:?", s) and len(s) < 40:
            mixed[s] += 1
print("short Title-case lines that may be headers:", mixed.most_common(15))
# inline headers like "CHIEF COMPLAINT: foo"
inline_h = Counter(m.group(1) for r in recs for m in re.finditer(r"^([A-Z][A-Z ]{3,}):\s*\S", r["note"], re.M))
print("inline 'HEADER: text' lines:", inline_h.most_common(15))

# ---------------------------------------------------------------- bullets
head("1.7 Bullets / list formatting in notes")
bullet_re = re.compile(r"^\s*([•\-\*·]|\d+[\.\)])\s", re.M)
bul = Counter(); bul_enc = defaultdict(set)
for r in recs:
    for m in bullet_re.finditer(r["note"]):
        bul[m.group(1)] += 1; bul_enc[m.group(1)].add(r["dataset"] + "/" + r["encounter_id"])
for k, n in bul.most_common():
    dss = Counter(x.split("/")[0] for x in bul_enc[k])
    print(f"  {k!r}: {n} lines in {len(bul_enc[k])} encounters {dict(dss)}")
# which sections carry bullets
sec_bul = Counter()
for r in recs:
    for h, body, _ in note_sections(r["note"]):
        if bullet_re.search(body):
            sec_bul[h] += 1
print("sections containing bullets:", sec_bul.most_common(12))
# numbered items inside sections like "1. Hypertension"
num_items = Counter(r["dataset"] for r in recs if re.search(r"^\s*\d+\.\s", r["note"], re.M))
print("notes with numbered items per dataset:", dict(num_items))
# leading-space / indentation
indent = Counter(r["dataset"] for r in recs if re.search(r"^[ \t]+\S", r["note"], re.M))
print("notes with indented lines:", dict(indent))
blank = Counter(r["dataset"] for r in recs if "\n\n" in r["note"])
print("notes with blank lines:", dict(blank))

# ---------------------------------------------------------------- non-ascii
head("1.8 Non-ASCII and control characters")
inv = {}
for r in recs:
    for field in ("dialogue", "note"):
        t = r[field]
        for i, ch in enumerate(t):
            if ord(ch) < 32 and ch != "\n" or ord(ch) > 126:
                key = ch
                d = inv.setdefault(key, {"name": unicodedata.name(ch, f"U+{ord(ch):04X}"), "codepoint": f"U+{ord(ch):04X}",
                                         "count": 0, "encounters": set(), "fields": Counter(), "datasets": Counter(), "examples": []})
                d["count"] += 1; d["encounters"].add(r["encounter_id"]); d["fields"][field] += 1; d["datasets"][r["dataset"]] += 1
                if len(d["examples"]) < 3:
                    d["examples"].append((r["encounter_id"], field, ctx(t, i, i + 1, 25)))
for ch, d in sorted(inv.items(), key=lambda kv: -kv[1]["count"]):
    print(f"  {d['codepoint']} {d['name']:<35} n={d['count']:<5} enc={len(d['encounters']):<4} fields={dict(d['fields'])} ds={dict(d['datasets'])}")
    for ex in d["examples"][:2]:
        print(f"       {ex[0]} {ex[1]}: {ex[2]}")
print("\\r present:", sum(r["dialogue"].count("\r") + r["note"].count("\r") for r in recs),
      " \\t present:", sum(r["dialogue"].count("\t") + r["note"].count("\t") for r in recs))
print("trailing whitespace lines (note):", sum(1 for r in recs for l in r["note"].split("\n") if l != l.rstrip()))
print("double spaces (note):", sum(r["note"].count("  ") for r in recs), " (dialogue):", sum(r["dialogue"].count("  ") for r in recs))
print("NFC != text (combining chars):", [r["encounter_id"] for r in recs for f in ("dialogue", "note") if unicodedata.normalize("NFC", r[f]) != r[f]])

save_json("headers_by_dataset.json", {
    "headers": hdr_by_ds,
    "header_sequences_top3": {ds: [{"n": n, "seq": list(seq)} for seq, n in order_by_ds[ds].most_common(3)] for ds in DATASETS},
    "empty_sections": empty_sections,
    "spelling_variants": {k: sorted(v) for k, v in norm.items() if len(v) > 1},
    "non_ascii": {d["codepoint"]: {"name": d["name"], "count": d["count"], "encounters": len(d["encounters"]), "fields": dict(d["fields"]), "datasets": dict(d["datasets"]), "examples": d["examples"]} for d in inv.values()},
})
