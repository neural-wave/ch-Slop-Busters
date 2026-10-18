"""Phase 11 - train vs eval shift.

Dataset mix, code distribution and chapters, lengths, PII density per 1k
words for the main categories, header sets, name overlap, structural anomalies.
Writes outputs/eda/train_eval_shift.json.
"""
import statistics

from piicats import *

S = by_split(recs)
out = {}
head("11.1 Dataset mix and lengths")
for sp in ("train", "eval"):
    rs = S[sp]
    out[sp] = {"n": len(rs), "dataset_mix": dict(Counter(r["dataset"] for r in rs)),
               "dialogue_chars_median": statistics.median(len(r["dialogue"]) for r in rs), "note_chars_median": statistics.median(len(r["note"]) for r in rs),
               "dialogue_words_median": statistics.median(len(r["dialogue"].split()) for r in rs), "note_words_median": statistics.median(len(r["note"].split()) for r in rs),
               "codes_per_encounter_mean": round(sum(len(r["icd10_codes"]) for r in rs) / len(rs), 2)}
    print(f"  {sp}: {out[sp]}")

head("11.2 Codes")
c_tr = Counter(c["code"] for r in S["train"] for c in r["icd10_codes"]); c_ev = Counter(c["code"] for r in S["eval"] for c in r["icd10_codes"])
ch_tr = Counter(c["code"][0] for r in S["train"] for c in r["icd10_codes"]); ch_ev = Counter(c["code"][0] for r in S["eval"] for c in r["icd10_codes"])
print("  chapter letters train:", dict(sorted(ch_tr.items())))
print("  chapter letters eval :", dict(sorted(ch_ev.items())))
unseen = {c: n for c, n in c_ev.items() if c not in c_tr}
print(f"  eval codes unseen in train: {len(unseen)}/{len(c_ev)} distinct, {sum(unseen.values())}/{sum(c_ev.values())} slots; encounters with >=1 unseen: {sum(1 for r in S['eval'] if any(c['code'] not in c_tr for c in r['icd10_codes']))}/{len(S['eval'])}")
print("  unseen examples:", list(unseen.items())[:12])
print("  top-10 train:", c_tr.most_common(10)); print("  top-10 eval :", c_ev.most_common(10))
print("  singletons (codes seen once) train:", sum(1 for v in c_tr.values() if v == 1), "eval:", sum(1 for v in c_ev.values() if v == 1))
out["codes"] = {"train_distinct": len(c_tr), "eval_distinct": len(c_ev), "eval_unseen_distinct": len(unseen), "eval_unseen_slots": sum(unseen.values()), "chapters_train": dict(ch_tr), "chapters_eval": dict(ch_ev)}

head("11.3 PII density per 1k words (regex categories from phase 3)")
KEY = ["patient_first_name_greeting", "patient_name_intro", "title_plus_name", "doctor_staff_name", "dob", "exact_age", "absolute_date", "location", "employer_occupation_school", "hobby", "phone_email_mrn_id_insurance", "named_relative_or_friend", "product_brand"]
dens = {}
for cat in KEY:
    rxs = [re.compile(p, re.I) for p in CATEGORIES[cat]]
    row = {}
    for sp in ("train", "eval"):
        for ds in DATASETS + ("all",):
            rs = [r for r in S[sp] if ds == "all" or r["dataset"] == ds]
            words = sum(len(r["dialogue"].split()) + len(r["note"].split()) for r in rs)
            hits = sum(1 for r in rs for f in ("dialogue", "note") for rx in rxs for m in rx.finditer(r[f]) if accept(cat, m))
            enc = sum(1 for r in rs if any(accept(cat, m) for f in ("dialogue", "note") for rx in rxs for m in rx.finditer(r[f])))
            row[f"{sp}/{ds}"] = {"per_1k_words": round(1000 * hits / max(words, 1), 2), "pct_encounters": round(100 * enc / max(len(rs), 1))}
    dens[cat] = row
    print(f"  {cat:<30} " + " ".join(f"{k}={v['per_1k_words']:>5}/{v['pct_encounters']:>3}%" for k, v in row.items() if k.endswith("/all")) + "   | per dataset eval: " + " ".join(f"{ds[:4]}={row[f'eval/{ds}']['per_1k_words']}" for ds in DATASETS))
out["pii_density"] = dens

head("11.4 Headers, speaker tags, structure")
for sp in ("train", "eval"):
    hs = Counter(h for r in S[sp] for h, _, _ in note_sections(r["note"]) if h)
    tags = Counter(sp_ for r in S[sp] for sp_, _, _ in dialogue_lines(r["dialogue"]))
    untagged = sum(1 for r in S[sp] for sp_, t, _ in dialogue_lines(r["dialogue"]) if sp_ is None and t.strip())
    print(f"  {sp}: {len(hs)} distinct headers; tags={dict(tags)}; untagged non-empty lines={untagged}; notes without any name token={sum(1 for r in S[sp] if not re.search(r'[A-Z][a-z]+ (?:is|was) an? (?:very )?(?:pleasant )?\d', r['note']) and not re.search(r'\b(?:Mr|Mrs|Ms)\.', r['note']))}")
    out[sp]["headers"] = dict(hs)
h_tr = set(out["train"]["headers"]); h_ev = set(out["eval"]["headers"])
print("  headers only in eval:", sorted(h_ev - h_tr)); print("  headers only in train:", sorted(h_tr - h_ev))
print("  aci dialogues without punctuation: train", sum(1 for r in S["train"] if r["dataset"] == "aci" and r["dialogue"].count(".") < 5), "/", sum(1 for r in S["train"] if r["dataset"] == "aci"),
      " eval", sum(1 for r in S["eval"] if r["dataset"] == "aci" and r["dialogue"].count(".") < 5), "/", sum(1 for r in S["eval"] if r["dataset"] == "aci"))
print("  dictated DOB header: train", sum(1 for r in S["train"] if re.search(r"date of birth|birth date", r["dialogue"][:300])), " eval", sum(1 for r in S["eval"] if re.search(r"date of birth|birth date", r["dialogue"][:300])))
print("  MRN: train", sum(1 for r in S["train"] if re.search(r"\bmrn|medical record number", r["dialogue"])), " eval", sum(1 for r in S["eval"] if re.search(r"\bmrn|medical record number", r["dialogue"])))
print("  [patient_guest]: train", sum(1 for r in S["train"] if "[patient_guest]" in r["dialogue"]), " eval", sum(1 for r in S["eval"] if "[patient_guest]" in r["dialogue"]))
print("  non-ASCII chars: train", sum(1 for r in S["train"] for ch in r["dialogue"] + r["note"] if ord(ch) > 126), " eval", sum(1 for r in S["eval"] for ch in r["dialogue"] + r["note"] if ord(ch) > 126))
save_json("train_eval_shift.json", out)
