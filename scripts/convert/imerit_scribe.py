"""imeritinc/multilingual-ambient-scribe-dataset: 199 scripted doctor-patient encounters in 5 languages.

Closest to the challenge format: `dialogue` from the transcript, `note` rendered from the structured summary
(Patient Information block + Reason for Visit, HPI, history, medications, exam, tests, Assessment & Plan).
Transcripts have no speaker labels: each line is one turn and we tag them alternately `[doctor]`/`[patient]`,
starting with the doctor (`meta.speaker_tags`). The same scenarios recur across languages (`meta.scenario`),
so keep a scenario in a single split when using this for anything but evaluation.

PII gold is partial: the annotated patient name, consulting physician and DOB, matched (case-sensitive, whole
word, honorific stripped) in both fields; where a field has no exact match, accent/one-edit spelling variants are
used (`label` ends in "(variant)"). Other names, places or dates in the transcript are not labelled.
No ICD-10 codes (diagnoses are in Assessment & Plan).
"""

from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KAGGLE, Labels, clean, dialogue_text, encounter, parser, sample, write_dataset  # noqa: E402

LANGS = {"UK English": "en", "German": "de", "French": "fr", "Spanish": "es", "Dutch": "nl"}
HEADER = re.compile(r"^(transcripted|transcription|translated)( line)?$", re.I)
QUOTES = "\"'“”„‚‘’«»"
HONORIFICS = set("mr mrs ms miss mx dr dra prof sr sra srta señor señora señorita don doña herr frau m mme mlle monsieur madame "
                 "mademoiselle dhr mevr mevrouw meneer heer dokter doctor doctora docteur".split())  # fmt: skip
# Some French annotations are sentences ("La patiente est Mme Johnson."): drop their capitalized function words.
STOPWORDS = set("la le les l el los las der die das de het the".split())
SKIP_SECTIONS = {"Physical Examination (DO NOT USE - Depreciated)"}


def html_text(s: str) -> str:
    s = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", s)
    s = re.sub(r"(?i)<li[^>]*>", "- ", s)
    s = html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ")
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def name_tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[^\W\d_][\w'-]*", value)
            if t[0].isupper() and len(t) > 1 and t.lower().rstrip(".") not in HONORIFICS | STOPWORDS]  # fmt: skip


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def one_edit(a: str, b: str) -> bool:
    """Levenshtein distance <= 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :] or (len(a) == len(b) and a[i + 1 :] == b[i + 1 :])


def variants(token: str, text: str) -> set[str]:
    """Capitalized words of `text` that spell `token` differently: accents (Lopez/López) or, for tokens of 5+ letters,
    one edit (Bennet/Bennett, Rodriquez/Rodriguez)."""
    key = fold(token)
    return {w for w in set(re.findall(r"[^\W\d_][\w'-]+", text))
            if w != token and w[0].isupper() and (fold(w) == key or (len(token) >= 5 and one_edit(fold(w), key)))}  # fmt: skip


def render_note(classifications: list[dict]) -> tuple[str, dict]:
    info, parts = {}, []
    for c in classifications:
        title = c["title"]
        if title == "Patient Information":
            info = {k["title"]: k["answer"].strip() for k in c.get("classifications", []) if isinstance(k.get("answer"), str)}
            block = [f"{k}: {v}" for k, v in info.items() if v and re.search(r"\w", v)]
            parts.append("PATIENT INFORMATION\n" + "\n".join(block))
        elif title not in SKIP_SECTIONS:
            body = html_text(c["answer"]) if isinstance(c.get("answer"), str) else ""
            subs = [f"{k['title']}: {html_text(k['answer'])}" for k in c.get("classifications", [])
                    if isinstance(k.get("answer"), str) and html_text(k["answer"])]  # fmt: skip
            text = "\n".join(x for x in [body, *subs] if x)
            if text:
                parts.append(f"{title.upper()}\n{text}")
    return "\n\n".join(parts), info


def read_turns(path: Path) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    if lines and HEADER.match(lines[0]):
        lines = lines[1:]
    return [(("doctor", "patient")[i % 2], ln.strip(QUOTES + " ")) for i, ln in enumerate(lines)]


def main(argv: list[str] | None = None) -> None:
    p = parser(__doc__, KAGGLE / "imeritinc__multilingual-ambient-scribe-dataset" / "iMerit_Multilingual_Ambient_Scribe_Dataset")
    a = p.parse_args(argv)
    tasks = []
    for folder, lang in LANGS.items():
        for f in sorted((a.src / folder / "summaries").glob("*.json")):
            tasks += [(folder, lang, t) for t in json.loads(f.read_text(encoding="utf-8"))]
    unmatched: list[str] = []

    def items():
        for folder, lang, t in sample(tasks, a.limit, a.seed):
            fname = t["externalId"]
            scenario = int(re.search(r"Encounter (\d+)", fname).group(1))
            note, info = render_note(t["task"]["classifications"])
            dialogue = dialogue_text(read_turns(a.src / folder / "transcripts" / fname))
            enc = encounter(f"imerit-{lang}-{scenario:03d}", "imerit_scribe", dialogue=clean(dialogue), note=clean(note))
            lab = Labels(enc)
            dob = info.get("Age/DOB", "")
            if re.search(r"\d", dob) and not lab.match_pii(dob, "DOB", "Age/DOB"):
                unmatched.append(f"{enc['encounter_id']}: {dob}")
            for key, cat in (("Name", "PATIENT_NAME"), ("Consulting Physician", "PERSON_NAME")):
                for tok in name_tokens(info.get(key, "")):
                    found = 0
                    for f in ("dialogue", "note"):
                        n = lab.match_pii(tok, cat, key, fields=[f], ignore_case=False)
                        if not n:  # spelling differs from the annotation in this field: fall back to near matches
                            n = sum(lab.match_pii(v, cat, f"{key} (variant)", fields=[f], ignore_case=False) for v in variants(tok, enc[f]))
                        found += n
                    if not found:
                        unmatched.append(f"{enc['encounter_id']}: {tok}")
            meta = {"language": lang, "scenario": scenario, "source_file": f"{folder}/transcripts/{fname}", "speaker_tags": "inferred_alternating",
                    "gender": info.get("Gender"), "consultation_type": info.get("Consultation Type")}  # fmt: skip
            yield enc, lab.record("none", pii_gold="partial", meta=meta)

    recs = list(items())
    write_dataset("imerit_scribe", recs, {
        "source": "kaggle:imeritinc/multilingual-ambient-scribe-dataset",
        "language": sorted(LANGS.values()),
        "origin": "scripted/acted encounters (transcript + annotated structured summary)",
        "license": "CC BY-NC-SA 4.0 (non-commercial)",
        "icd10": "none",
        "pii_gold": {"labelled": "annotated patient name, consulting physician, DOB (PATIENT_NAME, PERSON_NAME, DOB)",
                     "notes": f"other names/places/dates are unlabelled; {len(unmatched)} annotated values matched nowhere: {unmatched[:20]}"},
    }, a.out)  # fmt: skip


if __name__ == "__main__":
    main()
