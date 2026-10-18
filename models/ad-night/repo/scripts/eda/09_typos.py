"""Phase 9 - typos and noise.

Non-dictionary tokens per field (candidate misspellings / ASR errors),
abbreviation variants, casing errors in notes, broken sentences, double words,
bracketed artifacts; with examples.
"""
from namelib import *

recs = load_all()
WEB2 = {w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore")}
known = set()
for r in recs:  # medical vocabulary proxy: tokens that appear capitalized before a dose or in RESULTS
    known.update(w.lower() for w in re.findall(r"\b([A-Z][a-z]{3,})\b(?=[ ,]+\d+(?:\.\d+)? ?(?:mg|mcg|units?|ml|%))", r["note"]))
known |= FIRST_NAMES | SURNAMES | BRANDS | {"covid", "dax", "dragon", "mrn", "ekg", "ecg", "copd", "gerd", "chf", "cad", "ckd", "uti", "dvt", "stemi", "a1c", "hba1c", "mri", "cbc", "cmp", "bmp", "tsh", "psa", "ldl", "hdl", "bnp", "pvcs", "pvc", "tavr", "savr", "pcp", "ent", "gi", "pt", "ot", "ros", "hpi", "nsaid", "nsaids", "ibuprofen", "tylenol", "metformin", "lisinopril", "amlodipine", "atorvastatin", "omeprazole", "gabapentin", "prednisone", "albuterol", "doxycycline", "hydrochlorothiazide", "losartan", "metoprolol", "carvedilol", "simvastatin", "levothyroxine", "meclizine", "sertraline", "fluoxetine", "trazodone", "zyrtec", "flonase", "robitussin", "mucinex", "motrin", "advil", "aleve", "aspirin", "brilinta", "lipitor", "toprol", "norvasc", "lasix", "protonix", "nexium", "prilosec", "zofran", "flexeril", "mobic", "ultram", "cozaar", "bumex", "flovent", "symbicort", "wixela", "alvesco", "epipen", "coumadin", "eliquis", "xarelto", "plavix", "januvia", "ozempic", "lantus", "glimepiride", "jardiance", "farxiga", "entresto", "spironolactone", "aldactone", "furosemide", "torsemide", "zoloft", "prozac", "lexapro", "wellbutrin", "buspar", "ativan", "xanax", "ambien", "melatonin", "imitrex", "topamax", "nurtec", "cymbalta", "lyrica", "naprosyn", "naproxen", "celebrex", "meloxicam", "tramadol", "oxycodone", "percocet", "norco", "lortab", "vicodin", "hydrocodone", "lidocaine", "cortisone", "kenalog", "depo", "augmentin", "amoxicillin", "azithromycin", "zpack", "cipro", "keflex", "bactrim", "clindamycin", "tretinoin", "differin", "retin", "benzoyl", "panoxyl", "cetaphil", "elidel", "protopic", "hydroxychloroquine", "plaquenil", "methotrexate", "humira", "enbrel", "allopurinol", "colchicine", "uloric", "flomax", "tamsulosin", "finasteride", "viagra", "cialis", "fosamax", "boniva", "prolia", "synthroid", "armour", "tirosint", "levoxyl"}
med_from_notes = set()
for r in recs:
    med_from_notes.update(w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", r["note"]))

head("9.1 Non-dictionary tokens (candidate typos / ASR errors)")
for field in ("dialogue", "note"):
    cnt = Counter(); ex = {}
    for r in recs:
        for m in re.finditer(r"\b[a-z]{4,}\b", r[field].lower()):
            w = m.group(0)
            if w in WEB2 or w in known or w.endswith("ing") and w[:-3] in WEB2 or w.endswith("ed") and (w[:-2] in WEB2 or w[:-1] in WEB2) or w.endswith("s") and w[:-1] in WEB2 or w.endswith("ly") and w[:-2] in WEB2 or w.endswith("es") and w[:-2] in WEB2:
                continue
            cnt[w] += 1
            if w not in ex: ex[w] = (r["encounter_id"], ctx(r[field], m.start(), m.end(), 22))
    singles = [w for w, c in cnt.items() if c == 1]
    print(f"-- {field}: {len(cnt)} non-dictionary types, {sum(cnt.values())} tokens; singletons {len(singles)}")
    # dialogue tokens that never occur in any note (ASR garble candidates) vs note-only tokens
    if field == "dialogue":
        garble = [(w, c, ex[w]) for w, c in cnt.items() if w not in med_from_notes]
        print(f"   dialogue non-dictionary tokens absent from all notes (ASR-garble candidates): {len(garble)}")
        for w, c, e in sorted(garble, key=lambda x: -x[1])[:45]:
            print(f"     {w:<16} x{c:<3} {e[0]} {e[1]}")
    else:
        for w, c in cnt.most_common(45):
            print(f"     {w:<16} x{c:<3} {ex[w][0]} {ex[w][1]}")

head("9.2 Abbreviation / spelling variants of the same concept")
VARIANTS = {"a1c": r"\b(hemoglobin a1c|hemoglobin a1 c|hba1c|a1c|a-1-c|a one c|a1 c)\b", "x-ray": r"\b(x-ray|x ray|xray|x-rays|x rays|xrays|x - ray)\b", "ekg": r"\b(ekg|ecg|electrocardiogram|ekgs)\b",
            "blood pressure": r"\b(blood pressure|bp|b\.p\.|pressure)\b", "copd": r"\b(copd|chronic obstructive pulmonary disease|emphysema)\b", "covid": r"\b(covid|covid-19|covid 19|coronavirus|corona|the pandemic|pandemic)\b",
            "mri": r"\b(mri|magnetic resonance|mr arthrogram|mra)\b", "twice daily": r"\b(twice a day|twice daily|two times a day|two times per day|bid|b\.i\.d\.|2 times a day|2 times daily|2x a day|twice per day)\b",
            "milligrams": r"\b(mg|milligrams?|mgs)\b", "follow up": r"\b(follow-up|follow up|followup|f/u|recheck|re-check)\b", "pcp": r"\b(pcp|primary care|primary care physician|primary care provider|primary doctor|family doctor|regular doctor)\b",
            "ultrasound": r"\b(ultrasound|sonogram|echo|echocardiogram|echocardiocardiogram|us\b)\b", "shortness of breath": r"\b(shortness of breath|short of breath|sob|dyspnea|dyspneic|winded|breathless)\b"}
for k, pat in VARIANTS.items():
    c = Counter(); rx = re.compile(pat, re.I)
    for r in recs:
        for f in ("dialogue", "note"):
            for m in rx.finditer(r[f]): c[(f[:4], m.group(1).lower())] += 1
    print(f"  {k:<20} {sorted(c.items(), key=lambda kv: -kv[1])[:9]}")

head("9.3 Casing / punctuation errors inside notes")
checks = {"lowercase sentence start": r"(?:(?<=\. )|(?<=\n))[a-z][a-z]+ [a-z]", "double word": r"\b(\w+) \1\b", "missing space after period": r"[a-z]\.[A-Z][a-z]", "double punctuation": r"[.,]{2,}|\.\s\.",
          "space before punctuation": r"\s[,.;:](?!\d)", "glued camelCase": r"\b[a-z]+[A-Z][a-z]+\b", "stray character": r"[ß§¤¶]|​", "line ends without period (prose line >60 chars)": r"^[A-Za-z][^\n]{60,}[a-zA-Z0-9]$",
          "unbalanced parenthesis line": r"^[^()\n]*\([^()\n]*$", "lowercase 'i ' pronoun": r"\bi (?:am|have|will|recommend|advised|would)\b", "'ff'/'teh'-style typos": r"\b(ff|teh|adn|hte|fo|ot|si|ti)\b"}
for k, pat in checks.items():
    rx = re.compile(pat, re.M)
    c = 0; ex = []
    for r in recs:
        for m in rx.finditer(r["note"]):
            if k == "double word" and m.group(1).lower() in {"that", "had", "is", "very"}: pass
            c += 1
            if len(ex) < 3: ex.append((r["encounter_id"], ctx(r["note"], m.start(), m.end(), 25)))
    print(f"  {k:<48} {c:<5} {ex[:2]}")

head("9.4 Broken / truncated turns and bracketed artifacts in dialogue")
trunc = Counter(); ex = defaultdict(list)
for r in recs:
    for sp, t, off in dialogue_lines(r["dialogue"]):
        t = t.strip()
        if not t: trunc["empty line"] += 1; continue
        if t.endswith("-"): trunc["ends with '-' (cut off)"] += 1; ex["cut"].append((r["encounter_id"], t[-50:]))
        if t.endswith("..."): trunc["ends with '...'"] += 1
        if re.search(r"\[[^\]]*\]", t): trunc["bracket inside turn"] += 1; ex["bracket"].append((r["encounter_id"], t[:80]))
        if len(t.split()) == 1: trunc["one-word turn"] += 1
        if re.match(r"^[,.?!]", t): trunc["starts with punctuation"] += 1; ex["startpunct"].append((r["encounter_id"], t[:50]))
        if re.search(r"\b(\w+)- \1", t): trunc["stutter 'w- we'"] += 1
print(" ", dict(trunc))
for k, v in ex.items(): print(f"   {k}: {v[:3]}")
one_word = Counter()
for r in recs:
    for sp, t, off in dialogue_lines(r["dialogue"]):
        if len(t.strip().split()) == 1: one_word[t.strip().lower()] += 1
print("  one-word turns (top):", one_word.most_common(15))
