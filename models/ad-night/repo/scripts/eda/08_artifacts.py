"""Phase 8 - synthetic-data artifacts.

Boilerplate sentences repeated across notes, note sentences copied from the
dialogue, near-duplicate encounters, templated openings, uniformity of ages /
sex / phrasing, vocabulary cleanliness, train/eval overlap.
"""
import itertools

from namelib import *

recs = load_all()
by_id = {r["encounter_id"]: r for r in recs}
WEB2 = {w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore")}


def sentences(text):
    out = []
    for line in text.split("\n"):
        line = line.strip().lstrip("•-*0123456789. ").strip()
        if not line or HEADER_RE.match(line):
            continue
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip()
            if len(s.split()) >= 4:
                out.append(s)
    return out


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


head("8.1 Boilerplate sentences repeated across notes")
sent_enc = defaultdict(set)
for r in recs:
    for s in set(norm(x) for x in sentences(r["note"])):
        sent_enc[s].add(r["encounter_id"])
rep = {s: v for s, v in sent_enc.items() if len(v) >= 3}
print(f"  distinct note sentences: {len(sent_enc)}; appearing in >=3 notes: {len(rep)}")
for s, v in sorted(rep.items(), key=lambda kv: -len(kv[1]))[:22]:
    ds = Counter(by_id[e]["dataset"] for e in v)
    print(f"   x{len(v):<3} {dict(ds)!s:<45} {s[:95]}")
# boilerplate labels like "Medical Reasoning:"
labels = Counter()
for r in recs:
    for m in re.finditer(r"^[•\-\s]*([A-Z][A-Za-z ]{3,40}):", r["note"], re.M):
        labels[m.group(1)] += 1
print("  bullet labels:", labels.most_common(12))

head("8.2 Note sentences copied verbatim from the dialogue")
copy_stats = defaultdict(lambda: [0, 0]); examples = []
for r in recs:
    d = norm(r["dialogue"])
    for s in sentences(r["note"]):
        ns = norm(s)
        if len(ns.split()) < 6:
            continue
        copy_stats[r["dataset"]][1] += 1
        if ns in d:
            copy_stats[r["dataset"]][0] += 1
            if len(examples) < 5: examples.append((r["encounter_id"], s[:90]))
print("  note sentences (>=6 words) found verbatim in dialogue:", {k: f"{v[0]}/{v[1]} ({100*v[0]//max(v[1],1)}%)" for k, v in copy_stats.items()}, examples[:4])
# 8-gram overlap
def ngrams(t, n=8):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
ov = defaultdict(list)
for r in recs:
    nd, nn = ngrams(r["dialogue"]), ngrams(r["note"])
    ov[r["dataset"]].append(len(nd & nn) / max(1, len(nn)))
print("  share of note 8-grams present in dialogue (median per dataset):", {k: round(sorted(v)[len(v)//2], 3) for k, v in ov.items()})
# dialogue 8-grams shared across different encounters (scripted dialogue lines)
dlg_ng = defaultdict(set)
for r in recs:
    for g in ngrams(r["dialogue"], 10):
        dlg_ng[g].add(r["encounter_id"])
shared_lines = {g: v for g, v in dlg_ng.items() if len(v) >= 3}
print(f"  dialogue 10-grams shared by >=3 encounters: {len(shared_lines)}; examples:")
for g, v in sorted(shared_lines.items(), key=lambda kv: -len(kv[1]))[:10]:
    print(f"     x{len(v)} {g}")

head("8.3 Near-duplicate encounters (Jaccard on 5-gram shingles)")
sh_note = {r["encounter_id"]: ngrams(r["note"], 5) for r in recs}
sh_dlg = {r["encounter_id"]: ngrams(r["dialogue"], 5) for r in recs}
pairs = []
ids = list(sh_note)
for a, b in itertools.combinations(ids, 2):
    jn = len(sh_note[a] & sh_note[b]) / max(1, len(sh_note[a] | sh_note[b]))
    jd = len(sh_dlg[a] & sh_dlg[b]) / max(1, len(sh_dlg[a] | sh_dlg[b]))
    if jn > 0.15 or jd > 0.15:
        pairs.append((round(jn, 2), round(jd, 2), a, b, by_id[a]["split"], by_id[b]["split"]))
pairs.sort(reverse=True)
print(f"  pairs with Jaccard > 0.15 (note or dialogue): {len(pairs)}")
for p in pairs[:15]: print("    ", p)
a, b = by_id["D2N181"], by_id["D2N182"]
print("  D2N181 vs D2N182:", "note J=%.2f dlg J=%.2f" % (len(sh_note["D2N181"] & sh_note["D2N182"]) / len(sh_note["D2N181"] | sh_note["D2N182"]), len(sh_dlg["D2N181"] & sh_dlg["D2N182"]) / len(sh_dlg["D2N181"] | sh_dlg["D2N182"])))
print("     181:", a["dialogue"][:150].replace("\n", " / "))
print("     182:", b["dialogue"][:150].replace("\n", " / "))
print("     181 codes:", [c["code"] for c in a["icd10_codes"]], " 182 codes:", [c["code"] for c in b["icd10_codes"]])

head("8.4 Templated openings and uniformity")
def tmpl(line):
    t = line.lower()
    t = re.sub(r"\d+", "N", t)
    t = re.sub(r"\b(?:" + "|".join(sorted(FIRST_NAMES | SURNAMES, key=len, reverse=True)) + r")\b", "NAME", t)
    return t
first = Counter(); first_ds = defaultdict(Counter)
for r in recs:
    l0 = dialogue_lines(r["dialogue"])[0][1]
    t = " ".join(tmpl(l0).split()[:8])
    first[t] += 1; first_ds[r["dataset"]][t] += 1
print("  distinct 8-token opening templates:", len(first))
for ds in DATASETS:
    print(f"   {ds}: {first_ds[ds].most_common(4)}")
dax = sum(1 for r in recs if re.search(r"nurse told you about dax|tell dax|using dax|use dax", r["dialogue"]))
print("  dax preamble:", dax, "encounters;", "'hey , dragon' commands:", sum(1 for r in recs if "dragon" in r["dialogue"]), "encounters")
hpi = Counter()
for r in recs:
    m = re.search(r"\b(?:[A-Z][a-z]+ ){1,3}is an? (?:very )?(?:pleasant )?\d+-(?:year|month)-old (?:male|female|man|woman|gentleman|lady|boy|girl)\b", r["note"])
    hpi[r["dataset"]] += bool(m)
print("  notes with '<Name> is a NN-year-old <sex>' sentence:", dict(hpi), "of", {ds: len([r for r in recs if r['dataset']==ds]) for ds in DATASETS})
greet = sum(1 for r in recs if re.match(r"\[doctor\] (?:hi|hello|hey|good morning|good afternoon)\s?,?\s[a-z]+\s?[,.]? (?:how are you|how're you|good to see you|nice to see you)", r["dialogue"]))
print("  dialogues opening with '[doctor] hi , <name> . how are you ?'-type greeting:", greet)
ages = []
for r in recs:
    m = re.search(r"(\d+)-year-old", r["note"]) or re.search(r"(\d+) ?-?year ?-?old", r["dialogue"])
    if m: ages.append(int(m.group(1)))
ages.sort()
print(f"  ages: n={len(ages)} min={ages[0]} p10={ages[len(ages)//10]} median={ages[len(ages)//2]} p90={ages[9*len(ages)//10]} max={ages[-1]}; <18: {sum(a<18 for a in ages)}; >=80: {sum(a>=80 for a in ages)}; >=90: {sum(a>=90 for a in ages)}")
print("  age histogram (decades):", dict(sorted(Counter(a // 10 * 10 for a in ages).items())))
sexes = Counter()
for r in recs:
    m = re.search(r"\b(male|female)\b", r["note"]) or re.search(r"\b(male|female)\b", r["dialogue"])
    sexes[m.group(1) if m else "unknown"] += 1
print("  sex:", dict(sexes))
print("  'pleasant' in HPI:", sum(1 for r in recs if "pleasant" in r["note"]), "; 'presents to the clinic today':", sum(1 for r in recs if "presents to the clinic today" in r["note"]))

head("8.5 Vocabulary: cleanliness / noise")
for field in ("dialogue", "note"):
    toks = Counter()
    for r in recs:
        toks.update(re.findall(r"[a-z]+(?:'[a-z]+)?", r[field].lower()))
    oov = {w: c for w, c in toks.items() if w not in WEB2 and len(w) > 3 and not re.search(r"'", w)}
    print(f"  {field}: tokens={sum(toks.values())} types={len(toks)} TTR={len(toks)/sum(toks.values()):.3f} OOV types (not in web2, len>3)={len(oov)} OOV tokens={sum(oov.values())} ({100*sum(oov.values())/sum(toks.values()):.1f}%)")
    print("     top OOV:", sorted(oov.items(), key=lambda kv: -kv[1])[:25])

head("8.6 Train / eval overlap")
tr = [r for r in recs if r["split"] == "train"]; ev = [r for r in recs if r["split"] == "eval"]
sent_tr = set(); sent_ev = set()
for r in tr: sent_tr.update(norm(s) for s in sentences(r["note"]) if len(s.split()) >= 6)
for r in ev: sent_ev.update(norm(s) for s in sentences(r["note"]) if len(s.split()) >= 6)
print(f"  note sentences (>=6 words): train {len(sent_tr)}, eval {len(sent_ev)}, identical across splits: {len(sent_tr & sent_ev)}")
print("     examples:", [s[:80] for s in list(sent_tr & sent_ev)[:6]])
dl_tr = set(); dl_ev = set()
for r in tr: dl_tr.update(norm(t) for _, t, _ in dialogue_lines(r["dialogue"]) if len(t.split()) >= 8)
for r in ev: dl_ev.update(norm(t) for _, t, _ in dialogue_lines(r["dialogue"]) if len(t.split()) >= 8)
print(f"  dialogue lines (>=8 words): train {len(dl_tr)}, eval {len(dl_ev)}, identical: {len(dl_tr & dl_ev)}; examples: {[s[:70] for s in list(dl_tr & dl_ev)[:4]]}")
cs_tr = Counter(tuple(sorted(c["code"] for c in r["icd10_codes"])) for r in tr); cs_ev = Counter(tuple(sorted(c["code"] for c in r["icd10_codes"])) for r in ev)
print(f"  code SETS: train {len(cs_tr)} distinct, eval {len(cs_ev)}, eval sets seen in train: {sum(1 for k in cs_ev if k in cs_tr)} ({sum(v for k, v in cs_ev.items() if k in cs_tr)} encounters)")
c_tr = {c["code"] for r in tr for c in r["icd10_codes"]}; c_ev = Counter(c["code"] for r in ev for c in r["icd10_codes"])
print(f"  codes: eval distinct {len(c_ev)}, unseen in train: {sum(1 for c in c_ev if c not in c_tr)} codes / {sum(v for c, v in c_ev.items() if c not in c_tr)} of {sum(c_ev.values())} slots; eval encounters with >=1 unseen code: {sum(1 for r in ev if any(c['code'] not in c_tr for c in r['icd10_codes']))}")
