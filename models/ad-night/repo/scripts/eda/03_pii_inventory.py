"""Phase 3 - PII inventory per category.

For each category: a first-pass regex/gazetteer, hits per field x dataset,
encounters affected, note section, dialogue line position, surface forms and
examples.  Overlapping hits inside a category are merged (longest wins) and
name-like captures are filtered against a corpus-derived stop list.
Writes outputs/eda/pii_inventory.json and outputs/eda/pii_samples.txt
(20 random hits per category for hand-checking precision).
"""
import random
import statistics


from piicats import *

rnd = random.Random(0)
inventory = {}
samples_out = []
by_id = {r["encounter_id"]: r for r in recs}
for cat, pats in CATEGORIES.items():
    rxs = [re.compile(p, re.I) for p in pats]
    hits = []  # (encounter_id, dataset, split, field, start, end, text, line_idx, section)
    for r in recs:
        for field in ("dialogue", "note"):
            t = r[field]
            if field == "dialogue":
                offs = [o for _, _, o in dialogue_lines(t)]
            else:
                secs = note_sections(t)
            for rx in rxs:
                for m in rx.finditer(t):
                    if m.start() == m.end() or not accept(cat, m):
                        continue
                    if field == "dialogue":
                        pos = max(i for i, o in enumerate(offs) if o <= m.start()); sec = None
                    else:
                        pos = None; sec = None
                        for h, body, off in secs:
                            if off <= m.start():
                                sec = h
                    hits.append((r["encounter_id"], r["dataset"], r["split"], field, m.start(), m.end(), m.group(0), pos, sec))
    hits = merge(hits)
    by_field = Counter(h[3] for h in hits); by_ds = Counter(h[1] for h in hits); by_split = Counter(h[2] for h in hits)
    enc = {h[0] for h in hits}
    enc_by_ds = {ds: len({h[0] for h in hits if h[1] == ds}) for ds in DATASETS}
    forms = Counter(h[6].lower() for h in hits)
    secs_c = Counter(h[8] for h in hits if h[3] == "note")
    dlg_lines = [h[7] for h in hits if h[3] == "dialogue"]
    first3 = sum(1 for x in dlg_lines if x <= 2)
    inventory[cat] = {
        "patterns": pats, "hits": len(hits), "by_field": dict(by_field), "by_dataset": dict(by_ds), "by_split": dict(by_split),
        "encounters": len(enc), "pct_encounters": round(100 * len(enc) / N, 1),
        "encounters_by_dataset": {ds: f"{enc_by_ds[ds]}/{len([r for r in recs if r['dataset']==ds])}" for ds in DATASETS},
        "dialogue_hits_in_first3_lines": f"{first3}/{len(dlg_lines)}",
        "dialogue_line_median": statistics.median(dlg_lines) if dlg_lines else None,
        "note_sections_top": secs_c.most_common(6), "surface_forms_top": forms.most_common(20), "distinct_forms": len(forms),
        "examples": [(h[0], h[3], ctx(by_id[h[0]][h[3]], h[4], h[5], 30)) for h in hits[:3]],
    }
    samp = rnd.sample(hits, min(20, len(hits)))
    samples_out.append(f"\n##### {cat}  ({len(hits)} hits)\n" + "\n".join(
        f"  {h[0]} {h[1][:4]} {h[3][:4]} L{h[7] if h[7] is not None else '-'} [{(h[8] or '-')[:12]}]: {ctx(by_id[h[0]][h[3]], h[4], h[5], 45)}" for h in samp))
    inv = inventory[cat]
    print(f"{cat:<32} hits={inv['hits']:<5} dlg/note={by_field.get('dialogue',0)}/{by_field.get('note',0):<4} enc={inv['encounters']:>3} ({inv['pct_encounters']}%) "
          f"enc/ds={inv['encounters_by_dataset']} first3={inv['dialogue_hits_in_first3_lines']} forms={inv['distinct_forms']}")
    print(f"      top forms: {forms.most_common(10)}")
    print(f"      note sections: {secs_c.most_common(4)}")

(OUT_DIR / "pii_samples.txt").write_text("\n".join(samples_out), encoding="utf-8")
print("[saved] outputs/eda/pii_samples.txt")
save_json("pii_inventory.json", inventory)

# ---------------------------------------------------------------- recall probes
head("3.x Recall probes (encounters with NO hit for a category that should be near-universal)")
def hits_for(cat, field=None):
    rxs = [re.compile(p, re.I) for p in CATEGORIES[cat]]
    out = {}
    for r in recs:
        n = 0
        for f in ([field] if field else ("dialogue", "note")):
            n += sum(1 for rx in rxs for m in rx.finditer(r[f]) if accept(cat, m))
        out[r["encounter_id"]] = n
    return out
_g, _i, _t = hits_for("patient_first_name_greeting", "dialogue"), hits_for("patient_name_intro", "dialogue"), hits_for("title_plus_name", "dialogue")
name_dlg = {e: _g[e] + _i[e] + _t[e] for e in by_id}
no_name_dlg = [e for e, n in name_dlg.items() if n == 0]
print(f"dialogues with no patient-name hit: {len(no_name_dlg)} -> first 160 chars of each:")
for e in no_name_dlg:
    print(f"   {e} {by_id[e]['dataset']}: {by_id[e]['dialogue'][:160]!r}")
_i2, _t2 = hits_for("patient_name_intro", "note"), hits_for("title_plus_name", "note")
name_note = {e: _i2[e] + _t2[e] for e in by_id}
no_name_note = [e for e, n in name_note.items() if n == 0]
print(f"notes with no patient-name hit: {len(no_name_note)} -> capitalized tokens that are in the name gazetteers (likely missed names):")
truly_none = []
for e in no_name_note:
    body = "\n".join(l for l in by_id[e]["note"].split("\n") if not HEADER_RE.match(l.strip()))
    caps = [w for w in re.findall(r"(?<![.!?]\s)(?<!\n)\b[A-Z][a-z]+\b", body) if w.lower() in FIRST_NAMES | SURNAMES]
    caps = [w for w in caps if w.lower() not in {"may", "will", "mark", "bill", "ray", "pat", "sue", "hope", "grace", "june", "april", "rose", "lily", "holly", "summer", "autumn", "joy", "faith", "candy", "ginger", "sunny", "daisy", "ivy", "violet", "dawn", "jack", "bell", "wood", "woods", "rice", "price", "stone", "long", "little", "young", "white", "black", "green", "gray", "brown", "hill", "park", "lane", "fox", "wolf", "day", "west", "east", "north", "south", "cook", "baker", "fisher", "hunter", "carpenter", "mason", "walker", "ward", "wells", "banks", "fields", "bishop", "dean", "knight", "king", "joseph", "thomas", "martin", "george", "henry", "lee", "allen", "berry", "rush", "hall", "marshall", "chase", "lane"}]
    if caps:
        print(f"   {e} {by_id[e]['dataset']}: {Counter(caps).most_common(4)}")
    else:
        truly_none.append(e)
print(f"   notes with NO name token at all: {len(truly_none)} {Counter(by_id[e]['dataset'] for e in truly_none)} {truly_none}")
age = hits_for("exact_age")
no_age = [e for e, n in age.items() if n == 0]
print(f"encounters with no age hit: {len(no_age)}: {no_age}")
for e in no_age[:40]:
    m = re.search(r"(?:HISTORY OF PRESENT ILLNESS|HPI:?|SUBJECTIVE|CHIEF COMPLAINT)\n\n([^\n]{0,110})", by_id[e]["note"])
    print(f"   {e} {by_id[e]['dataset']}: {(m.group(1) if m else '')!r}")
print("--- all named relatives/friends/staff found:")
for cat in ("named_relative_or_friend",):
    rxs = [re.compile(p, re.I) for p in CATEGORIES[cat]]
    for r in recs:
        for f in ("dialogue", "note"):
            for rx in rxs:
                for m in rx.finditer(r[f]):
                    if accept(cat, m):
                        print(f"   {r['encounter_id']} {f[:4]}: {ctx(r[f], m.start(), m.end(), 35)}")
print("--- all clock times / MRN-like / insurer forms:")
for cat in ("clock_time", "phone_email_mrn_id_insurance"):
    rxs = [re.compile(p, re.I) for p in CATEGORIES[cat]]
    forms = Counter()
    for r in recs:
        for f in ("dialogue", "note"):
            for rx in rxs:
                for m in rx.finditer(r[f]):
                    if m.group(0).lower() != "insurance":
                        forms[(r["encounter_id"], m.group(0).lower())] += 1
    print(f"   {cat}: {sorted(forms)}")
