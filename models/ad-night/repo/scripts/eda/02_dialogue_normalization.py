"""Phase 2 - normalization quirks of `dialogue` vs `note`.

Titles, abbreviations, decimals/units, dates/times formats, hyphen/stutter
tokens, contractions, spelled-out numbers.  Prints counts per dataset and
a few verbatim examples with encounter_id.
"""
from common import *

recs = load_all()


def survey(name, pattern, fields=("dialogue", "note"), flags=re.I, n_ex=3, show=True, maxlen=70):
    """Count regex hits per field x dataset, print examples. Returns dict."""
    rx = re.compile(pattern, flags)
    res = {}
    for field in fields:
        cnt = Counter(); enc = defaultdict(set); ex = []
        for r in recs:
            for m in rx.finditer(r[field]):
                cnt[r["dataset"]] += 1; enc[r["dataset"]].add(r["encounter_id"])
                if len(ex) < n_ex:
                    ex.append((r["encounter_id"], ctx(r[field], m.start(), m.end(), 28)[:maxlen * 2]))
        res[field] = {"n": sum(cnt.values()), "by_dataset": dict(cnt), "encounters": sum(len(v) for v in enc.values()), "examples": ex}
    if show:
        d, n = res.get("dialogue"), res.get("note")
        line = f"{name:<32}"
        if d: line += f" dlg={d['n']:<5} {d['by_dataset']!s:<40} enc={d['encounters']:<4}"
        if n: line += f" | note={n['n']:<5} {n['by_dataset']!s:<38}"
        print(line)
        for f in fields:
            for e in res[f]["examples"][:2]:
                print(f"      {f[:4]} {e[0]}: {e[1]}")
    return res

head("2.0 Encounters without a [patient] line")
print([r["encounter_id"] + "/" + r["dataset"] for r in recs if "[patient]" not in r["dialogue"]])

head("2.1 Titles")
survey("mr/mrs/ms/dr + ' .' (split)", r"\b(mr|mrs|ms|dr) \. \w+")
survey("mr/mrs/ms/dr + '.' (unsplit)", r"\b(mr|mrs|ms|dr)\. \w+")
survey("mr/mrs/ms/dr no dot", r"\b(mr|mrs|ms|dr) (?!\.)[a-z]{2,}", fields=("dialogue",))
survey("miss/mister/missus", r"\b(miss|mister|missus|madam|ma'am|sir)\b")
survey("'doctor <name>'", r"\bdoctor [a-z]{3,}\b(?! (?:said|told|is|was|will|who|and|has|'s|wants|thinks|that|to|of|for|about|can|did|if|here))", fields=("dialogue",))
survey("'dr' in note (Dr. X)", r"\bDr\.? [A-Z][a-z]+", fields=("note",), flags=0)
survey("'nurse'/'pa' named", r"\b(nurse|physician assistant|pa|np) [a-z]{3,}", fields=("dialogue",))

head("2.2 Abbreviations")
for name, pat in [
    ("a.m./p.m.", r"\b[ap]\s?\.\s?m\s?\.?"), ("am/pm bare", r"\b\d{1,2}(?::\d\d)? ?[ap]m\b"),
    ("b.i.d/t.i.d/q.d", r"\b[bqt]\s?\.?\s?i?\s?\.?\s?d\s?\."), ("bid/tid/qd words", r"\b(bid|tid|qid|qd|prn|po|iv|im)\b"),
    ("twice a day/daily", r"\b(twice|once|three times|two times) (a|per) day\b"),
    ("x-ray variants", r"\bx[- ]?rays?\b"), ("copd/chf/cad/ckd/gerd/uti", r"\b(copd|chf|cad|ckd|gerd|uti|dvt|pe|mi|ra|oa|ibs|pcos|hiv|hcv|uc)\b"),
    ("ekg/ecg/echo/ct/mri/cbc/cmp/bmp", r"\b(ekg|ecg|echo|ct|mri|cbc|cmp|bmp|tsh|psa|ekgs|cts)\b"),
    ("a1c variants", r"\b(hemoglobin a1c|hba1c|a1c|a-1-c|a one c|a1 c)\b"), ("bp variants", r"\b(bp|blood pressure|b\.p\.)\b"),
    ("i.e./e.g./etc.", r"\b(i\.e\.|e\.g\.|etc\.?)"), ("letter-spelled (c-o-p-d)", r"\b([a-z]-){2,}[a-z]\b"),
    ("vs./vs", r"\bvs\.?\b"), ("st./ave./rd. (street)", r"\b(st|ave|rd|blvd)\s?\.")]:
    survey(name, pat)

head("2.3 Decimals, fractions, units")
for name, pat in [
    ("decimal 8.3", r"\b\d+\.\d+\b"), ("spaced decimal '8 . 3'", r"\b\d+ \. \d+\b"), ("spaced 'point' '8 point 3'", r"\b\d+ point \d+\b"),
    ("bp slash 140/90", r"\b\d{2,3}/\d{2,3}\b"), ("bp spaced '140 / 90'", r"\b\d{2,3} / \d{2,3}\b"), ("bp 'over' '140 over 90'", r"\b\d{2,3} over \d{2,3}\b"),
    ("mg/mcg/ml/units", r"\b\d+(\.\d+)? ?(mg|mcg|ml|mls|units?|iu|meq|g|grams?|milligrams?|micrograms?|cc)\b"),
    ("percent", r"\b\d+(\.\d+)? ?(%|percent)"), ("degrees/temp", r"\b\d{2,3}(\.\d)? ?(degrees|°|f\b)"),
    ("lbs/kg/feet/inches", r"\b\d+(\.\d+)? ?(lbs?|pounds?|kg|kilos?|kilograms?|feet|foot|ft|inches|inch|cm)\b"),
    ("ranges 2-3 / 2 to 3", r"\b\d+ ?(-|to) ?\d+\b"), ("comma thousands 1,000", r"\b\d{1,3},\d{3}\b"),
    ("x per week/day", r"\b(\d+|one|two|three|four|five|six) (times|x) (a|per) (day|week|month|year)\b"),
    ("vitals in note 'BP' line", r"(?:Blood pressure|BP)[:\s]+\d", ), ("spo2/o2 sat", r"\b(spo2|o2 sat(uration)?|oxygen saturation|sats?)\b"),
    ("number-letter glued 5mg", r"\b\d+(mg|mcg|ml|mm|cm|lbs|kg)\b")]:
    survey(name, pat)

head("2.4 Dates and times (all formats)")
MONTHS = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
for name, pat in [
    ("mm/dd/yyyy", r"\b\d{1,2}/\d{1,2}/\d{4}\b"), ("mm/dd/yy", r"\b\d{1,2}/\d{1,2}/\d{2}\b(?!\d)"), ("mm/dd no year", r"\b\d{1,2}/\d{1,2}\b(?!/)"),
    ("mm-dd-yyyy / yyyy-mm-dd", r"\b(\d{1,2}-\d{1,2}-\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
    ("month d(st) , yyyy", MONTHS + r"\.? \d{1,2}(st|nd|rd|th)? ?,? ?\d{4}\b"), ("month d(st) no year", MONTHS + r"\.? \d{1,2}(st|nd|rd|th)?\b(?! ?,? ?\d{4})"),
    ("month yyyy", MONTHS + r"\.? ,? ?(of )?\d{4}\b"), ("'d of month'", r"\b\d{1,2}(st|nd|rd|th)? of " + MONTHS),
    ("month alone", r"\b" + MONTHS + r"\b"), ("year 19xx/20xx", r"\b(19|20)\d{2}\b"), ("'in 2019' / 'since 2019'", r"\b(in|since|back in|of) (19|20)\d{2}\b"),
    ("ordinal day 'the 15th'", r"\bthe \d{1,2}(st|nd|rd|th)\b"), ("date of birth phrase", r"\b(date of birth|dob|birth ?date|born (on|in))\b"),
    ("date of service phrase", r"\b(date of service|today's date|today is)\b"),
    ("weekday", r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    ("clock hh:mm", r"\b\d{1,2}:\d{2}\b"), ("'o'clock'", r"\b\d{1,2} ?o ?' ?clock\b"), ("'in the morning/evening'", r"\b(in the (morning|afternoon|evening)|at night|bedtime|nighttime)\b"),
    ("season", r"\b(spring|summer|fall|autumn|winter)\b"), ("holiday", r"\b(christmas|thanksgiving|easter|halloween|new year'?s?|fourth of july|memorial day|labor day|mother's day|father's day|valentine'?s)\b"),
    ("'x ago'", r"\b(\w+[- ])?(\w+ )?(days?|weeks?|months?|years?|decades?) ago\b"), ("'last x'", r"\b(last|this past|past|next|this coming) (week|month|year|spring|summer|fall|winter|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday|" + MONTHS[1:-1] + r")\b"),
    ("age 'NN-year-old'", r"\b\d{1,3} ?-? ?year ?-? ?old\b"), ("age 'NN years old'", r"\b\d{1,3} years old\b"), ("age spelled 'fifty-nine-year-old'", r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(-| )?(one|two|three|four|five|six|seven|eight|nine)?(-| )?(year|years)(-| )?old\b"),
    ("age 'age NN'/'at NN'", r"\b(age|aged|at age) \d{1,3}\b"), ("'in his/her 50s'", r"\b(in (his|her|their|my) (early |late |mid[- ])?(\d0s|twenties|thirties|forties|fifties|sixties|seventies|eighties))\b"),
    ("since covid/pandemic", r"\b(since|during|before|after|with) (the )?(covid|pandemic|lockdown|quarantine)\b"), ("covid mention", r"\bcovid\b")]:
    survey(name, pat)

head("2.5 Hyphens, stutters, disfluencies")
for name, pat in [
    ("stutter 'w- we'", r"\b([a-z]{1,3})- \1", ), ("trailing hyphen token 'that-'", r"\b[a-z]{2,}- "), ("trailing hyphen before punct 'front- .'", r"\b[a-z]+- [\.,]"),
    ("'um'/'uh'", r"\b(um|uh|umm|uhh|hmm|mm-hmm|mm|mmhmm|hm|er|ah)\b"), ("'yeah'/'okay'/'alright'", r"\b(yeah|okay|ok|alright|all right|yep|nope)\b"),
    ("'you know'/'i mean'/'kind of'/'like ,'", r"\b(you know|i mean|kind of|sort of|like ,)\b"), ("hyphenated compound", r"\b[a-z]+-[a-z]+(-[a-z]+)*\b"),
    ("space-hyphen '56 -year-old'", r"\d -year"), ("double hyphen/dash", r"--|–|—"), ("ellipsis", r"\.\.\.|…"),
    ("repeated word 'the the'", r"\b(\w+) \1\b")]:
    survey(name, pat)

head("2.6 Contractions / tokenized forms")
for name, pat in [
    ("do n't / ca n't (split)", r"\b(do|does|did|ca|wo|is|are|was|were|have|has|had|should|could|would) n't\b"), ("don't (unsplit)", r"\b(don|doesn|didn|can|won|isn|aren|wasn|weren|haven|hasn|hadn|shouldn|couldn|wouldn)'t\b"),
    ("wan na / gon na / got ta", r"\b(wan na|gon na|got ta)\b"), ("wanna/gonna", r"\b(wanna|gonna|gotta)\b"),
    ("i'm / you're / it's / 's", r"\b\w+'(m|re|s|ll|ve|d)\b"), ("space before 's: 'patient 's'", r"\b\w+ '(s|ll|re|ve|m|d)\b"),
    ("apostrophe curly in dialogue", r"’", ), ("y'all / 'cause / 'em", r"\b(y'all|'cause|'em|'til)\b"),
    ("lowercase ' i '", r"\bi\b"), ("' i' capitalized", r"\bI\b")]:
    survey(name, pat, flags=0)

head("2.7 Numbers spelled out vs digits")
for name, pat in [
    ("digits", r"\b\d+\b"), ("number words", r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b"),
    ("compound 'fifty-nine'", r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(one|two|three|four|five|six|seven|eight|nine)\b"),
    ("'a couple'/'a few'", r"\b(a couple|a few|several|a handful|a number)\b"), ("'half'/'quarter'", r"\b(half|quarter|a third)\b"),
    ("ordinal words", r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"),
    ("year spelled '(nineteen|two thousand)'", r"\b(nineteen (\w+ )?(\w+)|two thousand( and)?( \w+)?( \w+)?)\b"),
    ("number + 'ish'", r"\b\d+-?ish\b|\b(\w+)ish\b")]:
    survey(name, pat)

head("2.8 Casing in note (truecase reference)")
survey("sentence starts lowercase in note", r"(?:^|\. )[a-z]\w+", fields=("note",), flags=re.M)
survey("ALL-CAPS words mid-sentence (note)", r"(?<![A-Z\n])\b[A-Z]{3,}\b(?![A-Z])", fields=("note",), flags=0)
survey("Mixed-case inside word (note)", r"\b[a-z]+[A-Z]\w*\b", fields=("note",), flags=0)
survey("Title-case word after lowercase (potential name)", r"(?<=[a-z] )[A-Z][a-z]{2,}\b", fields=("note",), flags=0, n_ex=6)
