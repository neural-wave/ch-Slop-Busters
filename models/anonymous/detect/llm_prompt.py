"""Prompt contract shared by fine-tuning (train/) and inference (detect/llm_layer.py)."""
import json
import re

from common import TIER

SYSTEM = (
    "You are an on-premise clinical de-identification model. You read an excerpt of a patient-doctor encounter "
    "and list every span that could help re-identify a person. You never list medical content."
)

INSTRUCTION = """List the privacy-related spans in the excerpt below as a JSON array.
Each item: {{"text": exact substring, "type": TYPE, "generalized": short non-identifying replacement or ""}}.
TYPEs — direct: NAME_PATIENT, NAME_RELATIVE, NAME_CLINICIAN, NAME_OTHER, AGE, DATE, LOCATION, ORGANIZATION, CONTACT, ID;
quasi-identifiers: OCCUPATION, ACTIVITY, LIFE_EVENT, TEMPORAL, PERSONAL_DETAIL.
Rules: copy text exactly; list each distinct span once; no titles in names; never list diseases, symptoms, drugs,
tests, procedures, body parts, the scribe product "dax"/"dragon", generic relation words or clinically needed
relative times. Disease eponyms (parkinson's disease) are medical. Output [] if there is nothing.

Excerpt ({field}):
<<<
{text}
>>>"""

TYPES = set(TIER)


def messages(field: str, text: str):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.format(field=field, text=text)}]


def target(items):
    return json.dumps([{"text": i["text"], "type": i["type"], "generalized": i.get("generalized", "")} for i in items],
                      ensure_ascii=False)


def parse(output: str):
    """Tolerant JSON-array parser for model output."""
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.S).strip()
    m = re.search(r"\[.*\]", output, flags=re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        items = []
        for obj in re.findall(r"\{[^{}]*\}", m.group(0)):
            try:
                items.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    return [i for i in items if isinstance(i, dict) and isinstance(i.get("text"), str) and i.get("type") in TYPES]
