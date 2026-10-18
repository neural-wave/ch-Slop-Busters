"""Phase 5 - ambiguous and adversarial tokens.

Numeric inventory with left/right context, classified into semantic classes
and labelled must-keep / privacy / neutral; eponyms; drug/brand names that look
like person or place names; lowercase entity tokens.
Writes outputs/eda/numeric_patterns.json.
"""
import random

from namelib import *
from numwords import spans as num_spans

recs = load_all()
rnd = random.Random(0)

LABS = r"(?:creatinine|potassium|sodium|hemoglobin|hgb|wbc|white blood cell|platelets?|ldl|hdl|cholesterol|triglycerides?|tsh|t4|psa|bun|egfr|gfr|ast|alt|bilirubin|albumin|inr|bnp|troponin|ferritin|iron|b12|vitamin d|calcium|magnesium|lipase|crp|esr|protein|titer|ph\b|lipid|lipids|labs?|blood work|hematocrit|mcv|glucose|sugar|sugars|a1c|hba1c|hemoglobin a1c|ejection fraction|ef\b|lvef|bsa|schnur|urobilinogen|ketones|nitrites?|specific gravity|cbc|cmp|bmp|uric acid|cortisol|prolactin|testosterone|estrogen|fsh|lh|hcg|ammonia|lactate|d-dimer|pt\b|ptt|aptt|igg|ige|rheumatoid factor|ana\b|anti-ccp|hepatitis|viral load|cd4|spirometry|fev1|fvc|dlco|pft)"
VITALS = r"(?:heart rate|pulse|respiratory rate|respirations?|temperature|temp|oxygen saturation|o2 sat|sat|spo2|bpm|degrees|fever|weight|weighs?|weighed|pounds|lbs|kg|kilos|height|tall|feet|foot|inches|bmi|body mass index|blood pressure|bp\b|pressure|systolic|diastolic|rr\b|hr\b)"
MEDS_UNITS = r"(?:mg|milligrams?|mcg|micrograms?|units?|iu|ml|milliliters?|cc|puffs?|tablets?|tabs?|pills?|capsules?|drops?|sprays?|patch|patches|grams?|g\b|meq|mls|teaspoons?|tablespoons?|ounces?|oz)"
DATE_TOK = re.compile(r"^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$|^\d{1,2}/\d{4}$")
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"


def classify(val, raw, left, right, kind):
    L, R = left.lower(), right.lower()
    rawl = raw.lower()
    if DATE_TOK.match(raw) and not re.search(r"\b(bp|blood pressure|pressure|grade|murmur|out of|strength|pain|scale|score|vision|acuity|reflex)\b", L[-45:] + " " + R[:20]):
        if re.search(r"date of birth|dob|birth ?date|born", L[-40:]): return "dob"
        if re.search(r"date of service|dos\b", L[-30:]): return "date_of_service"
        return "date_absolute"
    if re.search(r"\b" + MONTHS + r"\.?$", L[-12:].strip()) or re.match(r"^\s?(?:st|nd|rd|th)?,? ?" + MONTHS, R[:12]) or re.match(r"^(?:st|nd|rd|th)", R[:2]) and re.search(MONTHS, L[-15:]):
        return "date_absolute"
    if re.search(r"\b(?:mrn\w*|medical record number|record number|member id|id number|chart number)\b ?:?\s?(?:is )?$", L[-40:].strip() + "") or re.search(r"mrn", L[-12:]):
        return "mrn_id"
    if re.match(r"^\d{3}[-. ]\d{3}[-. ]\d{4}$", raw): return "phone"
    if re.match(r"^\d{1,2}:\d{2}", raw) or re.match(r"^ ?[ap]\.?m\b", R[:5]) or re.match(r"^ ?o'?clock", R[:9]): return "clock_time"
    if re.match(r"^ ?-? ?(?:year|month|week|day)s? ?-? ?old", R[:16]) or re.match(r"^ ?(?:years|year) of age", R[:14]) or re.search(r"\b(?:age|aged|at age|turning|turned|you are|he is|she is|i'm|i am) ?,?$", L[-14:].strip()) and int(val) < 110 and not re.match(r"^ ?(?:%|percent|mg|pounds)", R[:9]):
        return "age"
    if re.search(r"\$ ?$", L[-3:]) or re.match(r"^ ?(?:dollars|bucks)", R[:9]): return "money"
    if re.match(r"^ ?(?:%|percent)", R[:9]):
        if re.search(r"ejection fraction|ef\b|lvef|saturation|sat\b|spo2|o2|a1c|stenosis|blockage|occlusion|disability|chance|risk", L[-50:] + R[:30]): return "percent_clinical"
        return "percent_other"
    if re.match(r"^ ?(?:/|over) ?\d", R[:7]) or re.match(r"^\d{2,3}/\d{2,3}$", raw) or re.search(r"\d (?:/|over) $", L[-7:]) or re.search(r"\d/$", L[-4:]):
        if re.search(r"\b(?:out of|pain|scale|score|grade|murmur|strength|reflex|vision|acuity|\d/10|/10|/6|/5|/28|/16|/30)\b", L[-45:] + " " + R[:12]) or re.match(r"^\d{1,2}/(?:10|5|6|4|28|30|16|20)$", raw) or re.match(r"^\d/\d$", raw): return "score_or_grade"
        if re.match(r"^\d+/\d+$", raw) and int(raw.split("/")[0]) >= 70 and int(raw.split("/")[1]) <= 130 or re.search(r"blood pressure|bp\b|pressure|systolic|diastolic|\d{2,3} over", L[-50:] + R[:12]): return "blood_pressure"
        if re.search(r"\b(?:dose|mcg|mg|wixela|advair|symbicort|/50|/25|/500)\b", L[-30:] + R[:10]): return "dose"
        return "ratio_other"
    if re.match(r"^ ?" + MEDS_UNITS, R[:14]) or re.match(r"^ ?(?:of|more|less) ", R[:6]) and re.search(MEDS_UNITS + r"\b", R[:25]): return "dose"
    if re.search(r"\b(?:a1c|hba1c|hemoglobin a1c|a-one-c|a one c)\b", L[-45:] + R[:25]): return "hba1c"
    if re.search(r"\b(?:sugar|sugars|glucose)\b", L[-45:] + R[:20]): return "glucose"
    if re.search(r"\b(?:temperature|temp|fever)\b", L[-30:]) or re.match(r"^ ?(?:degrees|°)", R[:10]): return "vital_temperature"
    if re.match(r"^ ?(?:pounds|lbs|lb|kilos|kg|kilograms)\b", R[:11]) or re.search(r"\b(?:weight|weighs?|weighed|weight is|weight of|lost|gained|gain|loss)\b ?(?:about |around |of |to |at )?$", L[-20:]): return "vital_weight"
    if re.match(r"^ ?(?:feet|foot|inches|inch|'|\")", R[:8]) or re.search(r"\b(?:height|tall)\b", L[-15:]): return "vital_height"
    if re.search(r"\bbmi\b", L[-20:] + R[:10]): return "bmi"
    if re.search(r"\b(?:heart rate|pulse|hr\b|bpm)\b", L[-30:] + R[:8]): return "vital_heart_rate"
    if re.search(r"\b(?:respiratory rate|respirations?|rr\b)\b", L[-30:] + R[:8]): return "vital_resp_rate"
    if re.search(r"\b(?:saturation|sat\b|sats|spo2|o2|oxygen)\b", L[-35:] + R[:10]): return "vital_spo2"
    if re.search(r"\b(?:systolic|diastolic|blood pressure|bp\b|pressures?|top number|bottom number)\b", L[-45:] + R[:10]) and int(val) >= 40: return "blood_pressure"
    if re.match(r"^ ?(?:mm|millimeters?|cm|centimeters?|meters?|mm hg|mmhg)\b", R[:14]): return "measurement"
    if re.search(r"\b(?:stage|grade|class|level|type|clark level|breslow|gleason|phase|trimester|category)\b ?$", L[-16:].strip()) or re.search(r"\btype$", L[-6:]): return "stage_grade_type"
    if re.search(r"\b(?:score|scale|phq|gad|act\b|mmse|moca|apgar|asa|nyha|ldh|pain level|rated?|rating)\b", L[-30:] + R[:10]) or re.match(r"^ ?out of (?:ten|10|five|5|six|6)", R[:14]): return "score_or_grade"
    if re.search(LABS + r"\b", L[-45:] + " " + R[:25]): return "lab_value"
    if re.match(r"^ ?(?:times?|x)\b ?(?:a|per|each|every)? ?(?:day|week|month|year|daily|weekly)", R[:22]) or re.match(r"^ ?(?:times|x) ", R[:8]) or re.search(r"\b(?:every|each)$", L[-6:]) and re.match(r"^ ?(?:hours?|days?|weeks?|months?|years?|minutes?)", R[:9]) or re.match(r"^ ?(?:to|or|-) ?(?:\d+|\w+) times", R[:16]): return "frequency"
    if re.match(r"^ ?(?:ago)", R[:5]) or re.match(r"^ ?(?:days?|weeks?|months?|years?|hours?|minutes?|nights?|decades?)\b", R[:10]) and not re.match(r"^ ?(?:years?|months?) ?-? ?old", R[:14]):
        if re.search(r"\b(?:since|in|for|about|around|over|last|past|next|within|after|every|the last|the past|the next|another|almost|nearly|at least|up to|only|about|like|maybe|probably)\b ?(?:the )?(?:last |past |next )?$", L[-14:]) or re.match(r"^ ?(?:days?|weeks?|months?|years?) (?:ago|now|later|from now|prior|before|back)", R[:22]) or re.match(r"^ ?(?:days?|weeks?|months?|years?|hours?|minutes?)\b", R[:9]):
            return "duration_or_relative_time"
    if re.match(r"^(?:19|20)\d{2}$", raw) or kind == "year-ish" and 1900 < int(val) < 2030 or re.match(r"^\d{4}$", raw) and 1900 < int(raw) < 2030: return "year"
    if re.match(r"^ ?(?:refills?|pills?|tablets?|cigarettes?|packs?|pack|drinks?|beers?|glasses?|cups?|bottles?|miles?|steps?|flights?|stairs|pillows?|kids?|children|babies|sons?|daughters?|grandchildren|siblings?|brothers?|sisters?|dogs?|cats?|views?|episodes?|visits?|attempts?|rounds?|sets?|reps?|minutes? (?:walk|of)|hours? (?:of|a night)|liters?|cups|servings?|doses?|courses?|injections?|shots?|sessions?|more)\b", R[:16]): return "count"
    if int(val) >= 10000: return "large_number_id_like"
    if re.search(r"\b(?:number|problem|diagnosis|question|option|step|first|second|third)\b ?$", L[-12:]): return "ordinal_enumeration"
    return "other"


LABEL = {"dob": "privacy", "date_of_service": "privacy", "date_absolute": "privacy (generalize to year / shift)", "mrn_id": "privacy", "phone": "privacy", "clock_time": "privacy (low)",
         "age": "privacy (bucket) — code-relevant for age-specific codes", "year": "privacy (low) / clinical timeline", "money": "neutral",
         "percent_clinical": "must-keep", "percent_other": "neutral", "score_or_grade": "must-keep", "blood_pressure": "must-keep", "dose": "must-keep", "hba1c": "must-keep",
         "glucose": "must-keep", "vital_temperature": "must-keep", "vital_weight": "must-keep (BMI/obesity codes) — fingerprint risk if extreme", "vital_height": "must-keep (BMI)", "bmi": "must-keep",
         "vital_heart_rate": "must-keep", "vital_resp_rate": "must-keep", "vital_spo2": "must-keep", "measurement": "must-keep", "stage_grade_type": "must-keep",
         "lab_value": "must-keep", "frequency": "must-keep", "duration_or_relative_time": "must-keep (relative timeline)", "count": "neutral / must-keep (pack-years, drinks)",
         "large_number_id_like": "check (lab or id)", "ordinal_enumeration": "neutral", "ratio_other": "check", "other": "check"}

head("5.1 Numeric inventory (digits + spelled-out numbers) by semantic class")
classes = defaultdict(lambda: {"n": 0, "dialogue": 0, "note": 0, "by_dataset": Counter(), "spoken": 0, "examples": []})
allhits = []
for r in recs:
    for field in ("dialogue", "note"):
        t = r[field]
        for s, e, val, kind in num_spans(t.lower() if field == "dialogue" else t):
            raw = t[s:e]
            if not re.search(r"\d", val):
                continue
            try:
                int(float(val.replace(",", "").split("/")[0]))
            except ValueError:
                continue
            c = classify(val.replace(",", "").split("/")[0].split(".")[0] if "/" in raw else val.replace(",", "").split(".")[0], raw, t[max(0, s - 60):s], t[e:e + 40], kind)
            d = classes[c]; d["n"] += 1; d[field] += 1; d["by_dataset"][r["dataset"]] += 1
            if kind != "digits": d["spoken"] += 1
            allhits.append((c, r["encounter_id"], field, ctx(t, s, e, 28)))
for c, d in classes.items():
    ex = [h for h in allhits if h[0] == c]
    d["examples"] = [(h[1], h[2], h[3]) for h in rnd.sample(ex, min(4, len(ex)))]
    d["by_dataset"] = dict(d["by_dataset"]); d["label"] = LABEL.get(c, "check")
print(f"{'class':<28}{'n':>6}{'dlg':>6}{'note':>6}{'spoken':>7}  label")
for c, d in sorted(classes.items(), key=lambda kv: -kv[1]["n"]):
    print(f"{c:<28}{d['n']:>6}{d['dialogue']:>6}{d['note']:>6}{d['spoken']:>7}  {d['label']}")
    for ex in d["examples"][:3]:
        print(f"      {ex[0]} {ex[1][:4]}: {ex[2]}")
save_json("numeric_patterns.json", {"classes": classes, "total_numeric_tokens": len(allhits)})

head("5.2 Dates: clinically anchored vs purely identifying")
anch = Counter(); anch_ex = defaultdict(list)
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}/\d{4}|" + MONTHS + r"\.? \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?|" + MONTHS + r"\.? ?,? ?(?:of )?(?:19|20)\d{2}|(?:19|20)\d{2})\b", re.I)
CLIN = r"(?:surgery|surgical|procedure|stent|repair|replacement|colonoscopy|mammogram|biopsy|ecg|ekg|echo|mri|ct|x-ray|ultrasound|labs?|blood work|a1c|hospitaliz\w+|admitted|discharged|diagnos\w+|started|stopped|last seen|last visit|seen|visit|appointment|follow[- ]up|vaccin\w+|dose|injection|infusion|transplant|fracture|accident|injur\w+|fell|fall|pregnan\w+|delivered|stress test|cath\w*|pap|screening|obtained|performed|completed|underwent|placed|resected|removed|since|in)"
for r in recs:
    for field in ("dialogue", "note"):
        for m in DATE_RE.finditer(r[field]):
            L = r[field][max(0, m.start() - 70):m.start()].lower(); R = r[field][m.end():m.end() + 40].lower()
            if re.search(r"date of birth|dob|birth ?date|born", L[-35:]): k = "DOB"
            elif re.search(r"date of service|today is|today's date", L[-30:]): k = "date of service"
            elif re.search(r"\b" + CLIN + r"\b", L) or re.search(r"\b" + CLIN + r"\b", R): k = "clinically anchored (procedure/test/visit)"
            else: k = "unanchored"
            anch[(k, field)] += 1
            if len(anch_ex[k]) < 4: anch_ex[k].append((r["encounter_id"], ctx(r[field], m.start(), m.end(), 35)))
for k, n in sorted(anch.items()): print(f"  {k}: {n}")
for k, ex in anch_ex.items(): print(f"  {k}: {ex[:3]}")

head("5.3 Drug / brand / product names that look like person or place names")
drugs = Counter()
for r in recs:
    for m in re.finditer(r"\b([A-Z][a-z]{3,}(?:-[A-Z]?[a-z]+)?)\b(?=[ ,]+(?:\d+(?:\.\d+)? ?(?:mg|mcg|units?|ml|%|iu|meq)|\d+ ?(?:mg|mcg)|daily|twice|once|as needed|prn|nightly|tablets?|capsules?|cream|gel|ointment|inhaler|injection|patch))", r["note"]):
        drugs[m.group(1)] += 1
print("  capitalized tokens before a dose/route in notes (drug-name proxy):", len(drugs), "distinct;", drugs.most_common(30))
look_like = {d: c for d, c in drugs.items() if d.lower() in FIRST_NAMES | SURNAMES | PLACES}
print("  drug tokens that are also first/last/place names:", look_like)
lower_drugs = Counter()
for r in recs:
    for d in drugs:
        lower_drugs[d] += len(re.findall(r"\b" + re.escape(d.lower()) + r"\b", r["dialogue"]))
print("  same drug names, lowercase in dialogue (top):", lower_drugs.most_common(20))
print("  known brand/product words that collide with names/places (from reading):",
      {"brooks": "running-shoe brand + surname (Brooks)", "enterprise": "rent-a-car employer", "kaiser": "insurer + 'kaiser permanente broadway' (street)",
       "nike/asics/new balance": "shoe brands in HPI", "dax": "scribe product (72, virtassist)", "dragon": "voice assistant (280, virtassist)",
       "head & shoulders / selsun blue / nizoral": "shampoos", "honey nut cheerios": "food", "mountain dew": "soda (matches 'mountain')", "friday's": "restaurant (weekday)",
       "zoom": "telehealth platform", "mount baker": "place (surname Baker)", "rex center": "ASR for rec center"})
dd = Counter(); dx = Counter()
for r in recs:
    dd[r["dataset"]] += len(re.findall(r"\bdragon\b", r["dialogue"])); dx[r["dataset"]] += len(re.findall(r"\bdax\b", r["dialogue"]))
print("  dragon per dataset:", dict(dd), " dax per dataset:", dict(dx), " in notes:", sum(len(re.findall(r"\b(?:dragon|dax)\b", r["note"], re.I)) for r in recs))
print("  dragon command verbs:", Counter(m.group(1) for r in recs for m in re.finditer(r"dragon ?[,.?]? (\w+)", r["dialogue"])).most_common(8))

head("5.4 Lowercase tokens that would be entities if cased")
cased_note = Counter(); lower_dlg = Counter()
for r in recs:
    for m in re.finditer(r"(?<![.!?•\-]\s)(?<!\n)(?<!: )\b([A-Z][a-z]+)\b", r["note"]):
        w = m.group(1).lower()
        if w in FIRST_NAMES | SURNAMES: cased_note["name"] += 1
        elif w in PLACES: cased_note["place"] += 1
        elif w in BRANDS | set(drugs.keys()) | {d.lower() for d in drugs}: cased_note["brand/drug"] += 1
        else: cased_note["other_capitalized"] += 1
    for m in re.finditer(r"\b([a-z]+)\b", r["dialogue"]):
        w = m.group(1)
        if w in {"ray", "may", "hope", "summer", "april", "june", "will", "grace", "candy", "rich", "nick", "sunny", "dawn"}: continue
        if w in FIRST_NAMES | SURNAMES: lower_dlg["name-gazetteer"] += 1
        elif w in PLACES: lower_dlg["place"] += 1
        elif w in BRANDS: lower_dlg["brand"] += 1
print("  note: mid-sentence capitalized tokens by type:", dict(cased_note), "-> capitalization alone flags names but also ~", cased_note["other_capitalized"], "other tokens (drugs, tests, days, eponyms)")
print("  dialogue: lowercase tokens that are gazetteer names/places/brands:", dict(lower_dlg))
print("  share of ALL dialogue name tokens that are capitalized: 1/~900 (only 'Julie' in D2N114)")
