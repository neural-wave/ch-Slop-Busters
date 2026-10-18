"""Phase 6 - temporal information and code dependencies.

Inventory of temporal expressions by type and format; list of ICD-10 codes in
the data whose correctness depends on age, sex, laterality, encounter type or a
numeric threshold, and where that information lives in the text.
Writes outputs/eda/temporal_expressions.json.
"""
from namelib import *
from numwords import spans as num_spans

recs = load_all()
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"
NUMW = r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|several|few|couple(?: of)?|many|\d+(?:\.\d+)?|\d+ ?(?:-|to|or) ?\d+|\d+ or \d+|a couple of|a few|half a|one and a half|two and a half|\w+ and a half)"
TYPES = {
    "absolute_date_numeric": [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", r"\b\d{1,2}/(?:19|20)\d{2}\b"],
    "absolute_date_month_day_year": [r"\b" + MONTHS + r"\.? \d{1,2}(?:st|nd|rd|th)? ?,? ?(?:19|20)\d{2}\b", r"\b" + MONTHS + r"\.? \d{1,2}(?:st|nd|rd|th)? ?,? ?'\d{2}\b"],
    "absolute_month_day": [r"\b" + MONTHS + r"\.? \d{1,2}(?:st|nd|rd|th)?\b(?! ?,? ?(?:19|20)\d{2})(?! ?(?:mg|%))"],
    "absolute_month_year": [r"\b" + MONTHS + r"\.? ?,? ?(?:of )?(?:19|20)\d{2}\b"],
    "year_only": [r"\b(?:in|since|of|back in|until|from|through|by|around|about|early|late|spring of|summer of|fall of|winter of) (?:19|20)\d{2}\b", r"(?<![/\d-])\b(?:19|20)\d{2}\b(?![/\d-])(?! ?(?:mg|mcg|ml|units?|calories|steps|pounds|lbs|dollars))", r"\b(?:nineteen|two thousand|twenty) (?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)(?: [a-z]+)?\b"],
    "month_only": [r"\b(?:in|since|last|this|next|by|until|through|from|early|late|mid|around|of|during) " + MONTHS + r"\b(?! \d)"],
    "season_holiday": [r"\b(?:spring|summer|winter|autumn)\b|\b(?:in the|this|last|next|every|the|early|late) fall\b|\bfall of\b", r"\b(?:christmas|thanksgiving|easter|halloween|new year'?s?|fourth of july|memorial day|labor day|mother's day|father's day|valentine'?s|super bowl|spring break|the holidays|holiday weekend)\b"],
    "weekday": [r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b"],
    "clock_time": [r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[ap]\.?m\.?)?\b", r"\b\d{1,2}\s?[ap]\.?m\.?\b"],
    "time_of_day": [r"\b(?:noon|midnight|at night|in the morning|in the afternoon|in the evening|bedtime|at bedtime|supper time|dinner ?time|lunch ?time|breakfast|overnight|tonight|this morning|this afternoon|this evening|last night|first thing)\b"],
    "relative_ago": [r"\b(?:" + NUMW + r" )?(?:days?|weeks?|months?|years?|decades?|hours?|minutes?|nights?) ago\b", r"\b(?:a while|a long time|some time|a bit) ago\b"],
    "relative_last_next": [r"\b(?:last|this past|past|next|this coming|the other|previous|prior|following|coming|upcoming|earlier this|later this) (?:week|month|year|spring|summer|fall|winter|weekend|visit|appointment|time|night|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday|" + MONTHS + r"|few (?:days|weeks|months|years)|couple (?:of )?(?:days|weeks|months|years)|\d+ (?:days|weeks|months|years))\b", r"\b(?:yesterday|tomorrow|today|the day before|the other day|the weekend|over the weekend|recently|lately|currently|now|these days|nowadays|at this time|at this point)\b"],
    "duration": [r"\b(?:for|over|in|within|after|about|around|almost|nearly|roughly|the last|the past|the next|another|at least|up to|every|each) (?:the (?:last|past|next) )?" + NUMW + r" ?(?:more )?(?:days?|weeks?|months?|years?|hours?|minutes?|nights?)\b", r"\b" + NUMW + r" (?:days?|weeks?|months?|years?|hours?|minutes?) (?:later|from now|prior|before|back|now|straight|in a row)\b", r"\b(?:x|times) ?\d+ (?:days?|weeks?|months?)\b"],
    "frequency": [r"\b(?:once|twice|three times|two times|four times|\d+ times|\d+x|every|each|daily|weekly|monthly|nightly|bid|tid|qid|q\.?\d+ ?h|q\d+h|prn|as needed) ?(?:a |per |an )?(?:day|week|month|year|hour|night|morning|evening|other day|\d+ hours?|\d+ weeks?)?\b"],
    "age_exact": [r"\b(?:\d{1,3}|[a-z]+(?:[ -][a-z]+)?)\s?-?\s?(?:year|month|week|day)s?\s?-?\s?old\b", r"\b(?:age|aged|at age|at the age of|turning|turned) (?:\d{1,3}|[a-z]+(?: [a-z]+)?)\b"],
    "age_bucket_or_lifestage": [r"\bin (?:his|her|their|my) (?:early |late |mid[- ])?(?:\d0s|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b", r"\b(?:as a (?:kid|child|teenager|teen|baby|young (?:man|woman|adult))|when (?:i|she|he) was (?:a )?(?:kid|child|teenager|teen|little|young|younger|in (?:high school|college|school|the military|the army))|since (?:childhood|college|high school|(?:my|his|her) (?:teens|twenties|thirties|forties))|(?:in|during) (?:my|his|her) (?:teens|twenties|thirties|forties|fifties|sixties)|elderly|middle-aged|young adult|teenager|adolescent|toddler|infant|newborn|pediatric|geriatric|premenopausal|postmenopausal|perimenopausal|menopause)\b"],
    "pregnancy_timing": [r"\b\d+ weeks? (?:pregnant|gestation|along)\b|\b(?:first|second|third) trimester\b|\bpostpartum\b|\bgestational\b|\b(?:weeks|months) postpartum\b"],
    "era_anchor": [r"\b(?:since|during|before|after|with|through|around|when|at the start of|at the beginning of|in the middle of|prior to|post) (?:the )?(?:covid(?:-19)?|pandemic|lockdown|quarantine|shutdown|the fires?|smoke season|the war|9/11|hurricane \w+|the storm|the flood|the election)\b"],
}
inv = {}
by_id = {r["encounter_id"]: r for r in recs}
for name, pats in TYPES.items():
    rxs = [re.compile(p, re.I) for p in pats]
    hits = []
    for r in recs:
        for field in ("dialogue", "note"):
            seen = set()
            for rx in rxs:
                for m in rx.finditer(r[field]):
                    if (m.start(), m.end()) in seen: continue
                    seen.add((m.start(), m.end()))
                    hits.append((r["encounter_id"], r["dataset"], field, m.group(0).lower(), ctx(r[field], m.start(), m.end(), 25)))
    forms = Counter(h[3] for h in hits)
    inv[name] = {"hits": len(hits), "dialogue": sum(1 for h in hits if h[2] == "dialogue"), "note": sum(1 for h in hits if h[2] == "note"),
                 "encounters": len({h[0] for h in hits}), "by_dataset": dict(Counter(h[1] for h in hits)), "distinct_forms": len(forms),
                 "top_forms": forms.most_common(12), "examples": [(h[0], h[2], h[4]) for h in hits[:4]]}
head("6.1 Temporal expressions inventory")
print(f"{'type':<30}{'hits':>6}{'dlg':>6}{'note':>6}{'enc':>5}  top forms")
for name, d in inv.items():
    print(f"{name:<30}{d['hits']:>6}{d['dialogue']:>6}{d['note']:>6}{d['encounters']:>5}  {d['top_forms'][:6]}")
save_json("temporal_expressions.json", inv)

head("6.2 ICD-10 codes whose correctness depends on age / sex / laterality / encounter / threshold")
codes = {}
for r in recs:
    for c in r["icd10_codes"]:
        codes.setdefault(c["code"], {"description": c["description"], "encounters": []})["encounters"].append(r["encounter_id"])
dep = defaultdict(list)
for code, d in codes.items():
    desc = d["description"].lower()
    if re.search(r"\b(right|left|bilateral|unilateral)\b", desc): dep["laterality"].append(code)
    if re.search(r"\b(initial encounter|subsequent encounter|sequela)\b", desc): dep["encounter_type (initial/subsequent)"].append(code)
    if re.search(r"\b(with|without) (?:complication|hyperglycemia|hypoglycemia|angina|heart failure|obstruction|status|exacerbation|mention|esophagitis|bleeding)", desc) or re.search(r"\bwith (?:hyperglycemia|diabetic|foot ulcer)", desc): dep["with/without modifier (numeric or clinical threshold)"].append(code)
    if re.search(r"\b(type 1|type 2|type ii|type i)\b", desc): dep["diabetes type"].append(code)
    if re.search(r"\b(stage|class|grade|severity|mild|moderate|severe|chronic|acute|persistent|intermittent|uncomplicated|recurrent|in remission)\b", desc): dep["severity / stage / acuity (needs timing or numbers)"].append(code)
    if re.search(r"\b(pregnan|trimester|gestation|postpartum|newborn|neonat|infant|juvenile|child|adolescen|elderly|senile|age-related|presbyop|premature)\b", desc): dep["age-dependent"].append(code)
    if re.search(r"\b(male|female|prostat|ovar|uter|cervi|testic|menopaus|menstru|vagin|penile|erectile|breast|pregnan|hypogonadism)\b", desc): dep["sex-dependent"].append(code)
    if re.search(r"\b(obesity|overweight|bmi|body mass)\b", desc): dep["BMI threshold"].append(code)
    if re.search(r"\b(hypertensi|kidney disease|ckd|stage \d|renal failure|hyperglycemia|hypoglycemia|hyperlipid|hypercholesterol|anemia|hypothyroid|hyperthyroid|hyponatremia|hyperkalemia|hypokalemia)\b", desc): dep["lab/vital threshold"].append(code)
    if re.search(r"\b(fall|slip|trip|struck|accident|activity|place of occurrence|external cause|collision|sports|exercise|gymnastics|contact with)\b", desc) or re.match(r"[VWXY]\d", code): dep["external cause / activity / place (quasi-identifier!)"].append(code)
    if re.search(r"\b(tobacco|nicotine|alcohol|drug|opioid|cannabis|substance|dependence|abuse|use disorder)\b", desc): dep["substance use (sensitive)"].append(code)
    if re.search(r"\b(depress|anxiety|bipolar|schizo|psych|mental|adjustment disorder|ptsd|insomnia|adhd|dementia|cognitive)\b", desc): dep["mental health (sensitive)"].append(code)
    if re.search(r"\b(hiv|hepatitis|sexually|syphilis|gonorr|chlamydia|herpes|hpv)\b", desc): dep["HIV/STI/hepatitis (sensitive)"].append(code)
    if re.search(r"\b(unspecified)\b", desc): dep["unspecified (laterality/severity not needed)"].append(code)
for k, v in dep.items():
    print(f"  {k}: {len(v)} codes, {sum(len(codes[c]['encounters']) for c in v)} code slots")
    for c in v[:14]:
        print(f"      {c:<10} {codes[c]['description'][:70]:<72} enc={codes[c]['encounters'][:3]}")
save_json("temporal_expressions.json", {"inventory": inv, "code_dependencies": {k: [(c, codes[c]["description"], codes[c]["encounters"]) for c in v] for k, v in dep.items()}})

head("6.3 Where the code-relevant demographics live in the text")
lat = Counter(); sex = Counter(); age_loc = Counter(); enc_type = Counter()
for r in recs:
    for field in ("dialogue", "note"):
        lat[field] += len(re.findall(r"\b(right|left|bilateral)\b", r[field], re.I))
        sex[field] += len(re.findall(r"\b(male|female|man|woman|he|she|his|her|him|mr\.?|mrs\.?|ms\.?|mister|miss|gentleman|lady|boy|girl|son|daughter)\b", r[field], re.I))
    for h, body, _ in note_sections(r["note"]):
        if re.search(r"\b\d+[- ](?:year|month)s?[- ]old\b", body): age_loc[h or "(no header)"] += 1
        if re.search(r"\b(male|female)\b", body): sex[f"note:{h}"] += 1
    enc_type["initial words"] += len(re.findall(r"\b(new patient|first time|initial|new injury|just happened|this happened|yesterday|last week|two days ago)\b", r["dialogue"]))
    enc_type["subsequent words"] += len(re.findall(r"\b(follow[- ]up|recheck|re-check|since (?:the|your|my) (?:surgery|injury|last visit)|post-?op|doing since|how have you been doing)\b", r["dialogue"]))
print("  laterality words (right/left/bilateral):", dict(lat))
print("  sex cues (pronouns/titles/male/female):", {k: v for k, v in sex.items() if not k.startswith('note:')})
print("  'male/female' by note section:", sorted(((k, v) for k, v in sex.items() if k.startswith('note:')), key=lambda kv: -kv[1])[:8])
print("  note sections carrying the explicit age:", age_loc.most_common(8))
print("  encounter-type cues in dialogue:", dict(enc_type))
print("  encounters with 'female/male' in note:", sum(1 for r in recs if re.search(r"\b(male|female)\b", r["note"])), "/", len(recs),
      "; only pronouns:", sum(1 for r in recs if not re.search(r"\b(male|female)\b", r["note"]) and re.search(r"\b(he|she|his|her)\b", r["note"], re.I)))
