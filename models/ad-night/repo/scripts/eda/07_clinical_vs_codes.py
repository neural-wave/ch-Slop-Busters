"""Phase 7 - clinical content vs ICD-10 codes.

Lexical overlap between each code description and the two texts (verbatim /
paraphrased / symptom-only), where the evidence lives (note section), and an
inventory of sensitive-but-code-relevant content (mental health, substance
use, HIV/hepatitis, pregnancy, sexual history).
"""
from namelib import *

recs = load_all()
DESC_STOP = {"unspecified", "of", "the", "and", "or", "with", "without", "other", "in", "to", "initial", "subsequent", "encounter", "specified",
             "not", "due", "as", "site", "parts", "part", "level", "for", "on", "at", "by", "from", "status", "current", "personal", "history",
             "disease", "disorder", "disorders", "type", "primary", "essential", "mellitus", "syndrome", "native", "mention", "uncomplicated",
             "intractable", "undetermined", "intent", "same", "elsewhere", "classified", "nec", "nos", "a", "an", "right", "left", "bilateral",
             "unilateral", "acute", "chronic", "mild", "moderate", "severe", "activity", "place", "occurrence", "person", "injured"}
SYN = {"hypertension": ["high blood pressure", "blood pressure", "hypertensive"], "diabetes": ["diabetic", "sugar", "a1c"], "depression": ["depressed", "mood", "antidepressant"],
       "dysphagia": ["swallow", "swallowing"], "dorsalgia": ["back pain"], "gastro-esophageal": ["reflux", "gerd", "heartburn"], "reflux": ["gerd", "heartburn"],
       "hyperlipidemia": ["cholesterol", "lipid", "statin"], "hypothyroidism": ["thyroid", "levothyroxine", "synthroid"], "osteoarthritis": ["arthritis", "oa"],
       "epicondylitis": ["tennis elbow", "golfer's elbow", "elbow"], "sprain": ["sprained", "twisted"], "fracture": ["broken", "break", "fractured"],
       "contusion": ["bruise", "bruised"], "melanoma": ["skin cancer", "lesion", "mole"], "calculus": ["stone", "stones"], "cholelithiasis": ["gallstone", "gallstones"],
       "anemia": ["anemic", "hemoglobin", "iron"], "asthma": ["inhaler", "wheez", "albuterol"], "nicotine": ["smok", "cigarette", "tobacco"], "dependence": ["quit", "addict"],
       "alcohol": ["drink", "drinks", "beer", "wine", "liquor"], "insomnia": ["sleep", "sleeping"], "murmur": ["heart sound"], "heart failure": ["chf", "ejection fraction", "congestive"],
       "atrial": ["afib", "a-fib", "fibrillation"], "kidney": ["renal", "creatinine", "gfr"], "migraine": ["headache", "headaches"], "hematuria": ["blood in urine", "blood in the urine"],
       "lyme": ["tick", "lyme"], "rhinitis": ["allergies", "allergy", "zyrtec", "claritin"], "anaphylactic": ["allergic reaction", "epipen", "hives"], "peanuts": ["peanut"],
       "osteoporosis": ["bone density", "dexa", "fosamax"], "cognitive": ["memory"], "impairment": ["memory"], "hypogonadism": ["testosterone"], "transplant": ["transplant"],
       "angiopathy": ["circulation", "pulses", "claudication"], "ulcer": ["wound", "sore"], "coronary": ["cad", "stent", "heart attack", "stemi"], "atherosclerotic": ["cad", "stent", "blockage"],
       "mitral": ["valve"], "insufficiency": ["regurgitation", "leak"], "ventricular": ["pumping", "ejection"], "pain": ["hurts", "hurt", "sore", "ache", "aching"], "knee": ["knee"]}


def stem(w):
    w = w.lower().strip(".,;:()'\"")
    return w[:6] if len(w) > 6 else w


def overlap(desc, text):
    toks = [t for t in re.findall(r"[a-z][a-z\-']+", desc.lower()) if t not in DESC_STOP]
    if not toks:
        return 1.0, [], []
    tl = text.lower()
    found, missing = [], []
    for t in toks:
        if re.search(r"\b" + re.escape(stem(t)), tl) or any(s in tl for s in SYN.get(t, [])):
            found.append(t)
        else:
            missing.append(t)
    return len(found) / len(toks), found, missing


rows = []
for r in recs:
    for c in r["icd10_codes"]:
        d = c["description"]
        key = re.sub(r"\s*\(.*?\)", "", d.split(",")[0]).lower()
        od, fd, md = overlap(d, r["dialogue"]); on, fn, mn = overlap(d, r["note"])
        verb_note = key in r["note"].lower(); verb_dlg = key in r["dialogue"].lower()
        # which note section holds the best evidence
        best_sec, best = None, -1
        for h, body, _ in note_sections(r["note"]):
            o, _, _ = overlap(d, body)
            if o > best: best, best_sec = o, h
        rows.append({"eid": r["encounter_id"], "ds": r["dataset"], "split": r["split"], "code": c["code"], "desc": d, "ov_dlg": od, "ov_note": on,
                     "verbatim_note": verb_note, "verbatim_dlg": verb_dlg, "best_section": best_sec, "missing_note": mn, "missing_dlg": md})

head("7.1 Lexical overlap description <-> text")
def band(x): return "verbatim/≥0.99" if x >= 0.99 else ("paraphrase 0.5-0.99" if x >= 0.5 else ("weak 0.25-0.5" if x >= 0.25 else "symptom/lab-only <0.25"))
for field in ("ov_note", "ov_dlg"):
    print(f"-- {field}")
    for ds in DATASETS + ("all",):
        sub = [x for x in rows if ds == "all" or x["ds"] == ds]
        b = Counter(band(x[field]) for x in sub)
        print(f"   {ds:<11} n={len(sub):<4} " + "  ".join(f"{k}: {v} ({100*v//len(sub)}%)" for k, v in sorted(b.items())))
print("  code key phrase appears verbatim: note", sum(x["verbatim_note"] for x in rows), "/", len(rows), "; dialogue", sum(x["verbatim_dlg"] for x in rows))
print("  note overlap > dialogue overlap:", sum(x["ov_note"] > x["ov_dlg"] for x in rows), "; equal:", sum(x["ov_note"] == x["ov_dlg"] for x in rows), "; dialogue higher:", sum(x["ov_note"] < x["ov_dlg"] for x in rows))
print("  best note section for the code evidence:", Counter(x["best_section"] for x in rows).most_common(10))
print("  codes with weak/no lexical evidence in BOTH fields (<0.5):")
for x in sorted(rows, key=lambda x: x["ov_note"] + x["ov_dlg"])[:25]:
    print(f"     {x['eid']} {x['ds'][:4]} {x['code']:<9} {x['desc'][:60]:<62} note={x['ov_note']:.2f} dlg={x['ov_dlg']:.2f} missing={x['missing_note'][:4]}")
# external cause codes: are the words in the text?
print("  external-cause/activity codes (V/W/X/Y): evidence words present?")
for x in rows:
    if re.match(r"[VWXY]\d", x["code"]):
        print(f"     {x['eid']} {x['code']:<9} {x['desc'][:55]:<57} note={x['ov_note']:.2f} dlg={x['ov_dlg']:.2f} missing_note={x['missing_note']}")

head("7.2 Sensitive-but-code-relevant content")
SENS = {
    "mental_health": r"\b(depress\w*|anxiet\w*|anxious|panic|bipolar|schizo\w*|psychiatr\w*|psycholog\w*|therapist|therapy for|counsel\w*|suicid\w*|self-harm|ptsd|adhd|insomnia|antidepressant|prozac|zoloft|lexapro|sertraline|fluoxetine|buspar|wellbutrin|trazodone|xanax|ativan|klonopin|mood|dementia|memory loss|cognitive)\b",
    "substance_use": r"\b(smok\w*|cigarette\w*|tobacco|nicotine|vap\w*|alcohol\w*|drink(?:s|ing)?|beers?|wine|liquor|drunk|marijuana|cannabis|weed|thc|cocaine|heroin|opioid\w*|opiate\w*|narcotic\w*|methadone|suboxone|iv drug|drug use|substance|sober|rehab\b|aa meetings|overdos\w*)\b",
    "hiv_sti_hepatitis": r"\b(hiv|aids|hepatitis [abc]|hep [abc]|hcv|hbv|std|sti\b|sexually transmitted|syphilis|gonorrhea|chlamydia|herpes|hpv|prep\b|viral load|cd4)\b",
    "sexual_reproductive": r"\b(sexual\w*|sex\b|intercourse|erectile|libido|contracepti\w*|birth control|iud|condom\w*|pregnan\w*|miscarriage|abortion|menstrua\w*|menses|period|periods|menopaus\w*|fertility|infertil\w*|partner\w*|testosterone|hypogonadism|vaginal|penile|prostate)\b",
    "legal_social": r"\b(disability|lawsuit|lawyer|attorney|court|police|arrest\w*|jail|prison|custody|divorce\w*|homeless|shelter|abuse|domestic violence|neglect|food stamps|welfare|unemploy\w*|laid off|immigra\w*|undocumented)\b",
    "genetic_family_disease": r"\b(family history|runs in (?:my|the|her|his) family|genetic|hereditary|brca|huntington|sickle cell|cystic fibrosis|down syndrome|autism|smith-magenis|smith magenis)\b",
}
for k, pat in SENS.items():
    rx = re.compile(pat, re.I)
    enc = defaultdict(set); forms = Counter(); coded = set()
    for r in recs:
        for f in ("dialogue", "note"):
            for m in rx.finditer(r[f]):
                enc[r["dataset"]].add(r["encounter_id"]); forms[m.group(0).lower()] += 1
        cds = " ".join(c["code"] + " " + c["description"].lower() for c in r["icd10_codes"])
        if (k == "mental_health" and re.search(r"\bF[2-4]\d|G47|G31|R41", cds)) or (k == "substance_use" and re.search(r"\bF1[0-9]|Z72\.0|Z87\.891|Z71\.6|T40", cds)) \
           or (k == "hiv_sti_hepatitis" and re.search(r"\bB2[0-4]|B1[5-9]|Z21|A5[0-9]|A6[0-4]", cds)) or (k == "sexual_reproductive" and re.search(r"\bN(?:4[0-9]|5\d|6\d|7\d|8\d|9\d)|E29|O\d|Z3[0-9]|N91|N92|N94", cds)):
            coded.add(r["encounter_id"])
    total = sum(len(v) for v in enc.values())
    print(f"  {k:<24} encounters={total:<4} by_ds={ {d: len(v) for d, v in enc.items()} } coded_in_icd={len(coded):<3} top forms={forms.most_common(8)}")
# explicit examples worth a policy
print("  examples:")
for eid, pat in [("D2N023", r"hiv positive[^\n]{0,60}"), ("D2N113", r"[^\n]{0,60}iv drug[^\n]{0,60}"), ("D2N038", r"[^\n]{0,50}(?:iv drugs|partners)[^\n]{0,60}"), ("D2N142", r"[^\n]{0,40}(?:drinks|alcohol)[^\n]{0,60}"), ("D2N172", r"[^\n]{0,60}self-harm[^\n]{0,40}"), ("D2N046", r"[^\n]{0,30}adopted[^\n]{0,40}")]:
    r = next(x for x in recs if x["encounter_id"] == eid)
    for f in ("dialogue", "note"):
        m = re.search(pat, r[f], re.I)
        if m: print(f"     {eid} {f[:4]}: {m.group(0)[:130]!r}"); break
