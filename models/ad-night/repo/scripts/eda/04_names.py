"""Phase 4 - names deep-dive.

Builds a per-encounter name lexicon from both fields (patterns + gazetteer),
checks cross-field consistency, positions, title usage, collisions with
ordinary/medical words and names shared across encounters.
Writes outputs/eda/name_lexicon.json and outputs/eda/collisions.json.
"""
from namelib import *

recs = load_all()
by_id = {r["encounter_id"]: r for r in recs}
STOP = corpus_stopwords(recs) | GREET_STOP
WEB2 = {w.strip() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore")}
WEB2_LOWER = {w for w in WEB2 if w.islower()}

CLINICIAN_WORDS = {"doctor", "dr", "nurse", "doc"}
SENT_START = re.compile(r"(?:^|[.!?•\-]\s+|\n)")

# ------------------------------------------------------------- extraction
def note_names(note):
    """Return dict role -> Counter(name) from the note, plus ordered mentions."""
    out = defaultdict(Counter); mentions = []
    # patient full name: "First Last is a 59-year-old", "First Last a 54-year-old", "Ms. First Last is"
    for m in re.finditer(r"(?:\b(?P<t>Mr|Mrs|Ms|Miss|Mister)\.? )?(?P<f>[A-Z][a-z]+(?:-[A-Z][a-z]+)?)(?: (?P<l>[A-Z][a-z]+(?:-[A-Z][a-z]+)?))?,? (?:is |was )?an? (?:very )?(?:pleasant |lovely |delightful )?(?:\d+|[a-z]+)[- ](?:year|month|week|day)", note):
        f, l = m.group("f"), m.group("l")
        if f in {"The", "This", "She", "He", "Patient"}:
            continue
        if "is a" not in m.group(0) and "was a" not in m.group(0) and not (l and f.lower() in FIRST_NAMES):
            continue  # "Ordered a 24-hour ..." style false positives
        if m.group("t") and not l:
            out["patient_last"][f] += 1; mentions.append(("note", m.start(), m.group(0), "title_last")); continue
        out["patient_first"][f] += 1
        if l: out["patient_last"][l] += 1
        mentions.append(("note", m.start(), f + (" " + l if l else ""), "patient_full"))
    for m in NOTE_TITLE_RE.finditer(note):
        l = m.group("last")
        out["patient_title_last"][l] += 1; mentions.append(("note", m.start(), m.group(0), "title_last"))
    for m in NOTE_DR_RE.finditer(note):
        out["doctor"][m.group("last")] += 1; mentions.append(("note", m.start(), m.group(0), "doctor"))
    # bare first-name mentions of an already-known patient first name (e.g. "Diana has a 4/6 murmur")
    for f in list(out["patient_first"]):
        for m in re.finditer(r"\b" + re.escape(f) + r"\b", note):
            if not any(mm[1] <= m.start() < mm[1] + len(mm[2]) + 5 for mm in mentions):
                out["patient_first_bare"][f] += 1; mentions.append(("note", m.start(), f, "patient_first_bare"))
    # relatives with names: "his daughter, Theresa"
    for m in re.finditer(r"\b(?:his|her|their) (?:" + "|".join(sorted(RELATIVES)) + r"),? (?P<n>[A-Z][a-z]+)\b", note):
        out["relative"][m.group("n")] += 1; mentions.append(("note", m.start(), m.group(0), "relative"))
    return out, mentions


def dialogue_names(dlg):
    out = defaultdict(Counter); mentions = []
    lines = dialogue_lines(dlg)
    def add(role, name, pos, form, src):
        out[role][name] += 1; mentions.append(("dialogue", pos, form, role, src))
    # dictated headers / intros
    for rx, src in ((DLG_HEADER_RE, "header"), (DLG_HEADER2_RE, "header2"), (DLG_INTRO_RE, "intro")):
        for m in rx.finditer(dlg):
            f, l = m.group("first"), m.group("last")
            if f in STOP and l and l not in STOP:
                f, l = l, None
            if f in STOP or f in CLINICIAN_WORDS:
                continue
            if l and (l in STOP or l in {"is", "a"}):
                l = None
            add("patient_first", f, m.start(), m.group(0), src)
            if l: add("patient_last", l, m.start(), m.group(0), src)
    for m in DLG_GREET_RE.finditer(dlg):
        n = m.group("name")
        if n in STOP or n in CLINICIAN_WORDS:
            continue
        t = re.search(TITLES_DLG, m.group(0))
        if t:
            add("title_name", n, m.start(), m.group(0), "greet_title")
        else:
            add("patient_first", n, m.start(), m.group(0), "greet")
    for m in DLG_TITLE_RE.finditer(dlg):
        t, n = m.group("title"), m.group("name")
        if n in STOP or n == "arthrogram":
            continue
        if t in ("dr", "doctor"):
            add("doctor", n, m.start(), m.group(0), "title")
        else:
            add("title_name", n, m.start(), m.group(0), "title")
    # gazetteer vocatives in the first 3 non-empty lines and "so|alright|okay <name>" anywhere
    for i, (sp, txt, off) in enumerate(lines[:3]):
        for m in re.finditer(r"\b([a-z]+)\b", txt):
            w = m.group(1)
            if w in FIRST_NAMES and w not in STOP and w not in CLINICIAN_WORDS:
                add("patient_first_gaz", w, off + m.start(), w, "gaz_first3")
    for m in re.finditer(r"\b(?:so|alright|all right|okay|hey|thanks|thank you|bye|now|well|great|good|yeah|and|right|yes)\s?,?\s([a-z]+)\s?,?\s(?:so|i|it's|you|how|good|what|let's|we|tell|can|the|do|that|your|is|are|if|nice|thanks|thank|take|have|,|\.|\?|$)", dlg):
        w = m.group(1)
        if w in FIRST_NAMES and w not in STOP:
            add("patient_first_gaz", w, m.start(), m.group(0), "gaz_vocative")
    # staff: "<name> will come in", "with <name>"
    for m in re.finditer(r"\b(?:have|with|ask|see|tell|told|and|to) ([a-z]+) (?:come in|will|'ll|can|at the front|up front|in the front|the nurse|our nurse|my nurse|our ma|my ma|our assistant|my assistant)\b", dlg):
        w = m.group(1)
        if w in FIRST_NAMES and w not in STOP:
            add("staff", w, m.start(), m.group(0), "staff")
    for m in re.finditer(r"\b(?:my|his|her|your|our) (?:" + "|".join(sorted(RELATIVES)) + r")\s?,?\s(?:named )?([a-z]+)\b", dlg):
        w = m.group(1)
        if w in FIRST_NAMES and w not in STOP:
            add("relative", w, m.start(), m.group(0), "relative")
    return out, mentions, lines


lexicon = {}
all_first = Counter(); all_last = Counter()
pos_stats = []
consistency = Counter(); cons_examples = defaultdict(list)
title_usage = Counter(); title_examples = defaultdict(list)
header_pct = Counter()
for r in recs:
    eid = r["encounter_id"]
    nn, nm = note_names(r["note"])
    dn, dm, lines = dialogue_names(r["dialogue"])
    # resolve patient first/last
    note_first = nn["patient_first"].most_common(1)[0][0] if nn["patient_first"] else None
    note_last = nn["patient_last"].most_common(1)[0][0] if nn["patient_last"] else (nn["patient_title_last"].most_common(1)[0][0] if nn["patient_title_last"] else None)
    doctors = set(dn["doctor"]) | {x.lower() for x in nn["doctor"]}
    dlg_first_c = Counter({k: v for k, v in dn["patient_first"].items() if k not in doctors})
    if not dlg_first_c:
        dlg_first_c = Counter({k: v for k, v in dn["patient_first_gaz"].items() if k not in doctors})
    dlg_first = dlg_first_c.most_common(1)[0][0] if dlg_first_c else None
    dlg_last = dn["patient_last"].most_common(1)[0][0] if dn["patient_last"] else None
    dlg_title = dn["title_name"].most_common(1)[0][0] if dn["title_name"] else None
    # title_name in dialogue: first or last?
    for n, c in dn["title_name"].items():
        if note_last and n == note_last.lower(): title_usage["title+LAST"] += c
        elif note_first and n == note_first.lower(): title_usage["title+FIRST"] += c; title_examples["title+FIRST"].append((eid, n, note_first, note_last))
        else: title_usage["title+unknown"] += c; title_examples["title+unknown"].append((eid, n, note_first, note_last))
    if dlg_last is None and dlg_title and note_last and dlg_title == note_last.lower():
        dlg_last = dlg_title
    # consistency
    nf, nl, df, dl = (note_first or "").lower(), (note_last or "").lower(), (dlg_first or "").lower(), (dlg_last or "").lower()
    if nf and df:
        if nf == df: consistency["first: same"] += 1
        elif nf.startswith(df) or df.startswith(nf) or (len(nf) > 3 and len(df) > 3 and sum(a != b for a, b in zip(nf, df)) + abs(len(nf) - len(df)) <= 2):
            consistency["first: variant/typo"] += 1; cons_examples["first: variant/typo"].append((eid, df, nf))
        else:
            consistency["first: different"] += 1; cons_examples["first: different"].append((eid, df, nf))
    elif nf and not df: consistency["first: note only"] += 1; cons_examples["first: note only"].append((eid, r["dataset"], nf))
    elif df and not nf: consistency["first: dialogue only"] += 1; cons_examples["first: dialogue only"].append((eid, r["dataset"], df))
    else: consistency["first: neither"] += 1; cons_examples["first: neither"].append((eid, r["dataset"]))
    if nl and dl:
        consistency["last: both"] += 1
        if nl != dl: cons_examples["last: mismatch"].append((eid, dl, nl))
    elif nl: consistency["last: note only"] += 1
    elif dl: consistency["last: dialogue only"] += 1; cons_examples["last: dialogue only"].append((eid, r["dataset"], dl))
    else: consistency["last: neither"] += 1
    # positions
    pat_mentions = sorted(m for m in dm if m[3] in ("patient_first", "patient_first_gaz", "patient_last", "title_name"))
    if pat_mentions:
        first_pos = pat_mentions[0][1]
        first_line = max(i for i, (_, _, o) in enumerate(lines) if o <= first_pos)
        later = sum(1 for m in pat_mentions if max(i for i, (_, _, o) in enumerate(lines) if o <= m[1]) >= 5)
        pos_stats.append((eid, r["dataset"], first_line, len(pat_mentions), later, len(lines)))
    if DLG_HEADER_RE.search(r["dialogue"][:400]) or DLG_HEADER2_RE.search(r["dialogue"][:400]) or re.search(r"date of birth|medical record|mrn", r["dialogue"][:300]):
        header_pct[r["dataset"]] += 1
    lexicon[eid] = {
        "dataset": r["dataset"], "split": r["split"],
        "note": {k: dict(v) for k, v in nn.items() if v},
        "dialogue": {k: dict(v) for k, v in dn.items() if v},
        "resolved": {"patient_first_note": note_first, "patient_last_note": note_last, "patient_first_dialogue": dlg_first, "patient_last_dialogue": dlg_last,
                     "doctor_note": sorted(nn["doctor"]), "doctor_dialogue": sorted(dn["doctor"]), "staff_dialogue": sorted(dn["staff"]), "relative": sorted(set(dn["relative"]) | set(nn["relative"]))},
    }
    if note_first: all_first[note_first.lower()] += 1
    if dlg_first and not note_first: all_first[dlg_first.lower()] += 1
    if note_last: all_last[note_last.lower()] += 1

head("4.1 Cross-field consistency (patient name)")
for k, v in sorted(consistency.items()): print(f"  {k:<24} {v}")
for k in ("first: variant/typo", "first: different", "first: note only", "first: dialogue only", "first: neither", "last: mismatch", "last: dialogue only"):
    if cons_examples[k]: print(f"  {k}: {cons_examples[k][:12]}")
print("  encounters where dialogue has only the first name and note has first+last:",
      sum(1 for e in lexicon.values() if e["resolved"]["patient_first_dialogue"] and not e["resolved"]["patient_last_dialogue"] and e["resolved"]["patient_last_note"]))
print("  encounters where note uses first name alone (bare) after the full name:",
      sum(1 for e in lexicon.values() if e["note"].get("patient_first_bare")), [ (k, e["note"]["patient_first_bare"]) for k, e in lexicon.items() if e["note"].get("patient_first_bare")][:8])
print("  patient-name mentions per note (median/max):", statistics_median := sorted(sum(sum(v.values()) for k, v in e["note"].items() if k.startswith("patient")) for e in lexicon.values())[len(lexicon)//2], max(sum(sum(v.values()) for k, v in e["note"].items() if k.startswith("patient")) for e in lexicon.values()))

head("4.2 Positions in dialogue")
print("  encounters with a patient-name mention:", len(pos_stats), "/", len(recs))
fl = Counter(min(p[2], 5) for p in pos_stats)
print("  line of first mention (0..4, 5=later):", dict(sorted(fl.items())))
print("  per dataset, first mention in line 0:", {ds: f"{sum(1 for p in pos_stats if p[1]==ds and p[2]==0)}/{sum(1 for p in pos_stats if p[1]==ds)}" for ds in DATASETS})
print("  mentions per dialogue (median, max):", sorted(p[3] for p in pos_stats)[len(pos_stats)//2], max(p[3] for p in pos_stats))
print("  encounters with >=1 mention at line >=5:", sum(1 for p in pos_stats if p[4] > 0), " total such mentions:", sum(p[4] for p in pos_stats))
print("  dictated header (DOB/MRN/'next patient is'/'X Y is a NN-year-old') in first lines, per dataset:", {ds: f"{header_pct[ds]}/{len([r for r in recs if r['dataset']==ds])}" for ds in DATASETS})

head("4.3 'mr . james': title + first or last name?")
print(" ", dict(title_usage))
print("  title+FIRST examples:", title_examples["title+FIRST"][:10])
print("  title+unknown examples:", title_examples["title+unknown"][:10])

head("4.4 Doctor / staff / relative names")
doc_d = Counter(); doc_n = Counter(); staff = Counter(); rel = Counter()
for e, L in lexicon.items():
    doc_d.update(L["dialogue"].get("doctor", {})); doc_n.update(L["note"].get("doctor", {})); staff.update(L["dialogue"].get("staff", {})); rel.update(L["dialogue"].get("relative", {})); rel.update(L["note"].get("relative", {}))
print("  doctor names in dialogue:", len(doc_d), "distinct;", doc_d.most_common(12))
print("  doctor names in note:", len(doc_n), "distinct;", doc_n.most_common(12))
print("  encounters with any doctor name:", sum(1 for L in lexicon.values() if L["resolved"]["doctor_note"] or L["resolved"]["doctor_dialogue"]))
print("  doctor first names (doctor name in FIRST_NAMES and not in SURNAMES):", sorted({n for n in list(doc_d) + [x.lower() for x in doc_n] if n in FIRST_NAMES and n not in SURNAMES}))
print("  staff first names:", dict(staff))
print("  named relatives:", dict(rel))
print("  doctor surname == patient surname in same encounter:", [(e, L["resolved"]) for e, L in lexicon.items() if L["resolved"]["patient_last_note"] and L["resolved"]["patient_last_note"].lower() in L["resolved"]["doctor_dialogue"] + [x.lower() for x in L["resolved"]["doctor_note"]]])

head("4.5 Collisions: name tokens that are also ordinary / medical words")
names_in_data = (set(all_first) | set(all_last) | set(doc_d) | {x.lower() for x in doc_n} | set(staff) | set(rel)) - {"ordered", "gave", "mister", "miss"}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
drug_tokens = Counter()
for r in recs:
    for m in re.finditer(r"\b([A-Z][a-z]{3,})\b(?=[ ,]+(?:\d+(?:\.\d+)? ?(?:mg|mcg|units?|ml|%)|\d+ ?(?:mg|mcg)))", r["note"]):
        drug_tokens[m.group(1).lower()] += 1
BODY = {"bell", "ray", "hip", "shoulder", "knee", "foot", "hand", "back", "neck", "head", "chest", "heel", "toe", "chin", "lip", "brow", "temple", "arm", "leg"}
collisions = {}
for n in sorted(names_in_data):
    kinds = []
    if n in MONTHS: kinds.append("month")
    if n in MEDICAL_EPONYMS: kinds.append("eponym")
    if n in drug_tokens: kinds.append("drug")
    if n in BODY: kinds.append("body/physics")
    if n in WEB2_LOWER and n not in FIRST_NAMES | SURNAMES - {"may", "june", "april", "bill", "mark", "ray", "pat", "sue", "hope", "grace", "will", "rose", "lily", "holly", "summer", "dawn", "jack", "wood", "woods", "rice", "price", "stone", "long", "little", "young", "white", "black", "green", "gray", "brown", "hill", "park", "lane", "fox", "day", "west", "cook", "baker", "fisher", "hunter", "carpenter", "mason", "walker", "ward", "wells", "banks", "fields", "bishop", "dean", "knight", "king", "berry", "hall", "chase", "bell", "reed", "wright", "may", "cole", "miles", "glen", "chip", "art", "bob", "rob", "rich", "ken", "ed", "jay", "guy", "ivy", "candy", "ginger", "joy", "faith", "pat", "sunny", "daisy", "violet", "autumn", "june", "peter", "frank", "grant", "lance", "tanner", "chuck", "butch", "duke", "earl", "major", "wade", "pierce", "curtis", "norman", "carter", "porter", "turner", "weaver", "bush", "hedge", "moss", "burns", "snow", "frost", "storm", "rain", "love", "hart", "sharp", "savage", "powers", "gross", "small", "short", "swift", "quick", "strong", "wise", "noble", "best", "goodman", "newman", "freeman"} or n in {"may", "june", "april", "bill", "mark", "ray", "pat", "sue", "hope", "grace", "will", "rose", "lily", "holly", "summer", "dawn", "jack", "wood", "rice", "price", "stone", "long", "little", "young", "white", "black", "green", "gray", "brown", "hill", "park", "lane", "fox", "day", "west", "cook", "baker", "fisher", "hunter", "carpenter", "mason", "walker", "ward", "wells", "banks", "fields", "bishop", "dean", "knight", "king", "berry", "hall", "chase", "bell", "reed", "wright", "cole", "miles", "frank", "grant", "wade", "pierce", "carter", "porter", "turner", "weaver", "burns", "love", "hart", "sharp", "powers", "gross", "best", "cooper", "butler", "miller", "taylor", "smith", "brooks", "rogers", "ross", "cross", "marsh", "moore", "hayes", "lee", "gay", "guy", "joy", "faith", "dale", "glen", "cliff", "rock", "sandy", "rusty", "candy", "ginger", "sunny", "daisy", "ivy", "violet", "autumn", "art", "chip", "bob", "rob", "rich", "ken", "jay", "ed", "major", "earl", "duke", "chuck", "norman", "bush", "snow", "frost", "storm", "rain"}:
        kinds.append("common_word")
    if not kinds:
        continue
    # occurrences in data as non-name: lowercase mid-sentence in NOTE, or in dialogue not adjacent to a name cue
    note_word = []; dlg_word = []; note_name = 0; dlg_name = 0
    for r in recs:
        for m in re.finditer(r"\b" + re.escape(n) + r"\b", r["note"]):
            if r["note"][m.start()].islower():
                note_word.append((r["encounter_id"], ctx(r["note"], m.start(), m.end(), 30)))
        for m in re.finditer(r"(?<![A-Za-z])" + re.escape(n[0].upper() + n[1:]) + r"\b", r["note"]):
            note_name += 1
        for m in re.finditer(r"\b" + re.escape(n) + r"\b", r["dialogue"]):
            c = r["dialogue"][max(0, m.start() - 25):m.end() + 12]
            if re.search(r"\b(hi|hello|hey|so|alright|okay|thanks|bye|mr|mrs|ms|dr|doctor|is a|is an)\b ?,? ?\.? ?" + re.escape(n) + r"\b|\b" + re.escape(n) + r" ?,", c):
                dlg_name += 1
            else:
                dlg_word.append((r["encounter_id"], ctx(r["dialogue"], m.start(), m.end(), 30)))
    collisions[n] = {"kinds": kinds, "as_name_note_capitalized": note_name, "as_name_dialogue_cued": dlg_name,
                     "as_word_note": len(note_word), "as_word_dialogue_uncued": len(dlg_word),
                     "examples_word": (note_word + dlg_word)[:3]}
print(f"  {len(collisions)} colliding name tokens out of {len(names_in_data)} names in the data")
for n, d in sorted(collisions.items(), key=lambda kv: -(kv[1]["as_word_note"] + kv[1]["as_word_dialogue_uncued"]))[:40]:
    print(f"  {n:<12} {','.join(d['kinds']):<22} name(note cap)={d['as_name_note_capitalized']:<3} name(dlg cued)={d['as_name_dialogue_cued']:<3} word(note)={d['as_word_note']:<3} word(dlg)={d['as_word_dialogue_uncued']:<3} e.g. {d['examples_word'][:2]}")
# eponyms present in data at all (as disease)
head("4.6 Medical eponyms present in the corpus (disease sense) and gazetteer names present as words")
ep = Counter()
for r in recs:
    for f in ("dialogue", "note"):
        for m in re.finditer(r"\b(" + "|".join(re.escape(e) for e in sorted(MEDICAL_EPONYMS, key=len, reverse=True)) + r")\b", r[f], re.I):
            ep[m.group(1).lower()] += 1
print("  ", ep.most_common(30))
gaz_as_word = Counter()
for r in recs:
    for m in re.finditer(r"\b([a-z]+)\b", r["dialogue"]):
        if m.group(1) in FIRST_NAMES and m.group(1) not in names_in_data:
            gaz_as_word[m.group(1)] += 1
print("  gazetteer first names occurring in dialogue but NOT a name in this corpus (would be false positives of a pure gazetteer):", gaz_as_word.most_common(25))

head("4.7 Names shared across encounters")
full = defaultdict(list)
for e, L in lexicon.items():
    f, l = L["resolved"]["patient_first_note"] or L["resolved"]["patient_first_dialogue"], L["resolved"]["patient_last_note"] or L["resolved"]["patient_last_dialogue"]
    if f and l: full[(f.lower(), l.lower())].append(e)
shared = {k: v for k, v in full.items() if len(v) > 1}
print(f"  distinct full names: {len(full)}; shared by >1 encounter: {len(shared)}")
for (f, l), es in sorted(shared.items()):
    rows = []
    for e in es:
        r = by_id[e]
        age = re.search(r"(\d+)[- ]?(?:year|month)s?[- ]old", r["note"]) or re.search(r"(\d+) ?-?year", r["dialogue"])
        sex = "F" if re.search(r"\b(female|woman|she)\b", r["note"][:600], re.I) else ("M" if re.search(r"\b(male|man|he)\b", r["note"][:600], re.I) else "?")
        rows.append(f"{e}/{r['dataset'][:4]}/{r['split']}: age={age.group(1) if age else '?'} sex={sex} codes={[c['code'] for c in r['icd10_codes']]}")
    print(f"  {f} {l}: " + " | ".join(rows))
print("  first names shared (top):", [(n, c) for n, c in all_first.most_common(15) if c > 1])
print("  last names shared (top):", [(n, c) for n, c in all_last.most_common(15) if c > 1])
tr_f = {L["resolved"]["patient_first_note"] or L["resolved"]["patient_first_dialogue"] for L in lexicon.values() if L["split"] == "train"} - {None}
ev_f = {L["resolved"]["patient_first_note"] or L["resolved"]["patient_first_dialogue"] for L in lexicon.values() if L["split"] == "eval"} - {None}
tr_full = {k for k, v in full.items() if any(by_id[e]["split"] == "train" for e in v)}
ev_full = {k for k, v in full.items() if any(by_id[e]["split"] == "eval" for e in v)}
print(f"  first names: train {len(tr_f)}, eval {len(ev_f)}, eval names also in train: {len({x.lower() for x in ev_f} & {x.lower() for x in tr_f})}")
print(f"  full names: train {len(tr_full)}, eval {len(ev_full)}, shared: {sorted(tr_full & ev_full)}")

save_json("name_lexicon.json", {"per_encounter": lexicon, "patient_first_names": all_first, "patient_last_names": all_last,
                                "doctor_names_dialogue": doc_d, "doctor_names_note": doc_n, "staff_first_names": staff, "named_relatives": rel,
                                "consistency": consistency, "title_usage": title_usage, "shared_full_names": {" ".join(k): v for k, v in shared.items()}})
save_json("collisions.json", collisions)
