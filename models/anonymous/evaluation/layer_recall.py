"""Dev tool: per-type recall/precision of detection layers against the eval PHI ground truth."""
import collections
import json
import sys

from common import find_all, load_jsonl


def score(recs, spans_by_id, ann_dir="annotations/eval", show=8):
    tot, hit = collections.Counter(), collections.Counter()
    misses = collections.defaultdict(list)
    fp, nspans = collections.Counter(), 0
    for r in recs:
        ann = json.load(open(f"{ann_dir}/{r['encounter_id']}.json"))
        spans = spans_by_id[r["encounter_id"]]
        nspans += len(spans)
        for it in ann["phi"]:
            occ = find_all(r[it["field"]], it["text"])
            tot[it["type"]] += 1
            if occ and all(any(s.field == it["field"] and s.start < b and a < s.end for s in spans) for a, b in occ):
                hit[it["type"]] += 1
            elif len(misses[it["type"]]) < show:
                misses[it["type"]].append(it["text"])
        gold = [(it["field"], *o) for it in ann["phi"] for o in find_all(r[it["field"]], it["text"])]
        for s in spans:
            if not any(s.field == f and s.start < b and a < s.end for f, a, b in gold):
                fp[(s.type, s.text.lower()[:30])] += 1
    for t in sorted(tot, key=lambda t: -tot[t]):
        print(f"  {t:16s} {hit[t]:4d}/{tot[t]:4d} = {hit[t] / tot[t]:.2f}   miss: {misses[t]}")
    print(f"  TOTAL recall {sum(hit.values()) / sum(tot.values()):.3f}  spans={nspans}  unmatched={sum(fp.values())}")
    print("  top unmatched:", fp.most_common(25))
