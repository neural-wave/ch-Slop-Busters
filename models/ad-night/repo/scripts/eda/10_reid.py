"""Phase 10 - re-identification risk.

Per-encounter quasi-identifier profile (age bucket, sex, code set, rare
details), k-anonymity on several attribute combinations, rarest details per
encounter, and what an attacker holding the original text would use.
"""
from piicats import *

by_id = {r["encounter_id"]: r for r in recs}
QI_CATS = ["employer_occupation_school", "hobby", "sports_team", "location", "event", "pet", "vehicle", "other_quasi_identifier", "product_brand", "named_relative_or_friend", "doctor_staff_name"]
rx = {c: [re.compile(p, re.I) for p in CATEGORIES[c]] for c in QI_CATS}
IGN = {"dax", "dragon", "insurance", "car", "the gym", "high school", "college", "at work", "military", "walking", "exercise", "fast food", "the woods", "vacation", "trip", "birthday", "holidays", "holiday", "party", "parties"}
profile = {}
token_enc = defaultdict(set)
for r in recs:
    eid = r["encounter_id"]
    age = None
    m = re.search(r"(\d+)-(?:year|month)-old", r["note"]) or re.search(r"(\d+) ?-?(?:year|month) ?-?old", r["dialogue"])
    if m:
        age = int(m.group(1)); 
        if re.search(r"month", m.group(0)): age = 0
    sex = None
    m = re.search(r"\b(male|female)\b", r["note"]) or re.search(r"\b(male|female)\b", r["dialogue"])
    if m: sex = m.group(1)
    elif re.search(r"\b(she|her)\b", r["note"], re.I): sex = "female"
    elif re.search(r"\b(he|his)\b", r["note"], re.I): sex = "male"
    codes = tuple(sorted(c["code"] for c in r["icd10_codes"]))
    details = Counter()
    for c in QI_CATS:
        for f in ("dialogue", "note"):
            for x in rx[c]:
                for mm in x.finditer(r[f]):
                    if not accept(c, mm): continue
                    tok = re.sub(r"\s+", " ", mm.group(0).lower().strip())
                    tok = re.sub(r"^(?:my|his|her|our|the|a|an|at|in|to|from|with) ", "", tok)
                    if tok in IGN or len(tok) < 3: continue
                    details[(c, tok)] += 1
    for k in details: token_enc[k].add(eid)
    profile[eid] = {"dataset": r["dataset"], "split": r["split"], "age": age, "age5": None if age is None else (age // 5 * 5 if age < 90 else 90), "age10": None if age is None else age // 10 * 10,
                    "sex": sex, "codes": codes, "details": details}
# rarity
for eid, p in profile.items():
    p["rare_details"] = sorted(((len(token_enc[k]), k[0], k[1]) for k in p["details"]), key=lambda x: x[0])[:5]
    p["n_unique_details"] = sum(1 for k in p["details"] if len(token_enc[k]) == 1)

head("10.1 k-anonymity on attribute combinations (whole corpus, 207 encounters)")
def kanon(keyfn, label):
    groups = Counter(keyfn(p) for p in profile.values())
    sizes = [groups[keyfn(p)] for p in profile.values()]
    print(f"  {label:<52} unique={sum(1 for s in sizes if s == 1):>3} ({100*sum(1 for s in sizes if s == 1)//len(sizes)}%)  k<=2={sum(1 for s in sizes if s <= 2):>3}  k<=5={sum(1 for s in sizes if s <= 5):>3}  median k={sorted(sizes)[len(sizes)//2]}")
kanon(lambda p: (p["sex"],), "(sex)")
kanon(lambda p: (p["age10"], p["sex"]), "(age decade, sex)")
kanon(lambda p: (p["age5"], p["sex"]), "(age 5y bucket, sex)")
kanon(lambda p: (p["age"], p["sex"]), "(exact age, sex)")
kanon(lambda p: (p["codes"],), "(code set)")
kanon(lambda p: (frozenset(p["codes"]) & {"I10", "E11.9", "F32.A", "K21.9", "I50.9", "E78.5"},), "(only the 6 most common codes)")
kanon(lambda p: (p["age10"], p["sex"], p["codes"]), "(age decade, sex, code set)")
kanon(lambda p: (p["age5"], p["sex"], p["codes"]), "(age 5y, sex, code set)")
kanon(lambda p: (p["age"], p["sex"], p["codes"]), "(exact age, sex, code set)")
occ = lambda p: tuple(sorted(t for c, t in p["details"] if c == "employer_occupation_school"))
hob = lambda p: tuple(sorted(t for c, t in p["details"] if c in ("hobby", "sports_team")))
kanon(lambda p: (p["age5"], p["sex"], p["codes"], occ(p)), "(age 5y, sex, codes, occupation)")
kanon(lambda p: (p["age5"], p["sex"], p["codes"], occ(p), hob(p)), "(age 5y, sex, codes, occupation, hobbies)")
kanon(lambda p: (p["sex"], p["codes"], occ(p), hob(p)), "(sex, codes, occupation, hobbies) — no age at all")
print("  note: k is computed within this 207-encounter corpus; the real population is unknown, so 'unique' = unique among the released encounters.")

head("10.2 Rarest details per encounter (corpus frequency 1 = appears in this encounter only)")
print(f"  encounters with >=1 corpus-unique detail: {sum(1 for p in profile.values() if p['n_unique_details'] > 0)} / {len(profile)}; median unique details per encounter: {sorted(p['n_unique_details'] for p in profile.values())[len(profile)//2]}")
print("  by dataset (mean unique details):", {ds: round(sum(p['n_unique_details'] for p in profile.values() if p['dataset']==ds) / len([p for p in profile.values() if p['dataset']==ds]), 1) for ds in DATASETS})
for eid, p in sorted(profile.items(), key=lambda kv: -kv[1]["n_unique_details"])[:25]:
    print(f"   {eid} {p['dataset'][:4]} age={p['age']} {p['sex']} codes={list(p['codes'])[:3]} unique={p['n_unique_details']} rarest={[(c[1][:14], c[2][:22]) for c in p['rare_details'][:4]]}")
all_unique = Counter(k[0] for k, v in token_enc.items() if len(v) == 1)
print("  corpus-unique detail tokens by category:", dict(all_unique))

head("10.3 What an attacker with the original text would link on")
print("  1. exact age + sex + full code set: see k-anonymity above (codes are preserved by design, so they are always available to the attacker).")
print("  2. rare free-text details that survive a name/date-only anonymizer: occupations, employers, hobbies, teams, places, pets, relatives, events (corpus-unique tokens:", sum(all_unique.values()), ").")
print("  3. clinical fingerprints that must be kept: exact lab values (A1c 7.4, creatinine 0.7, EF 35%), medication lists with doses, procedure history (RCA stent 2018), weights/heights.")
print("  4. structural fingerprints: number of turns, section order, boilerplate variant, dataset style — trivially re-linkable to the original encounter by text similarity even after perfect PII removal (the eval set is public).")
ex = {}
for eid in ("D2N138", "D2N058", "D2N105", "D2N029", "D2N189", "D2N157", "D2N076"):
    if eid in profile: ex[eid] = [c[2] for c in profile[eid]["rare_details"][:5]]
print("  examples of rare detail bundles:", ex)
save_json("reid_profiles.json", {eid: {k: (list(v) if isinstance(v, tuple) else (dict((f"{a}|{b}", n) for (a, b), n in v.items()) if isinstance(v, Counter) else v)) for k, v in p.items()} for eid, p in profile.items()})
