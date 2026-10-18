Code-Anchored, Verify-and-Repair Anonymization of Patient–Doctor Encounters
Proposal for the Bosch "Anonymization for Patient Data" challenge @ Neural Wave Hackathon.

1. What the data looks like
140 training / 67 evaluation encounters from three sources (aci, virtassist, virtscribe).
Dialogues: ~6.5k characters, lowercase ASR-style transcripts with [doctor] / [patient] turns.
Notes: ~2.7k characters, structured sections (Chief Complaint, HPI, Assessment & Plan, …).
~2 ICD-10 codes per encounter; codes and descriptions are available at inference and must be preserved.
Identifiers observed:

Category	Examples in the data	Notes
Names	first name in dialogue ("andrew") → full name in note ("Andrew Campbell")	Cross-field consistency is essential
Honorifics	Ms./Mr./Mrs./Dr.	Frequent; strong name cue
Ages	"59-year-old"	Exact age in almost every encounter
Dates	"june 2020", "3/6", "2019"	Absolute, linkable to visit date
Relatives	wife, son Sean, daughter Theresa, mother in hospital	Often named
Soft quasi-identifiers	occupation, employer, hobbies ("barre classes", "landscaping"), vacations	Rare words → re-identification risk
Non-PHI traps	"dax" (ambient-scribe product)	Naive NER strips it as a name
2. Core idea
Standard de-identification detects PHI and redacts. Here the ICD-10 codes are available at inference, so they are used as a protection specification: anonymize everything else aggressively, then verify that code-relevant information survived and roll back any edit that destroyed it.

2.1 Medical anchor extraction (what must survive)
Expand each assigned code's description through the ICD-10 hierarchy and UMLS/SNOMED synonyms (scispaCy entity linker, CPU).
Link every disorder, finding, drug, procedure and lab mention in dialogue and note to a concept.
Lock all mentions that support an assigned code or one of its siblings/parents (e.g. E11.9 vs E11.65 depends on HbA1c).
Derive code-aware perturbation bands: lab values, BMI, BP, eGFR may only be jittered inside the band that maps to the same code (HbA1c 8.3 → 8.1 is fine; 6.6 → 6.4 crosses the diagnostic threshold and is forbidden).
2.2 Layered PHI detection
Layer	Target	Resource
Regex	HIPAA Safe Harbor structured identifiers: dates, phones, MRNs, emails, IDs	CPU, negligible
De-id transformer (RoBERTa-class, i2b2-trained)	names, locations, ages, professions	CPU or <1 GB VRAM
Zero-shot span model (GLiNER-class)	occupation, employer, hobbies, relatives, life events, institutions	~1 GB
Gated ~3B instruct LLM at Q4	only sentences flagged as non-medical context or containing rare tokens (IDF against general + clinical corpus)	~2 GB VRAM, <10 % of sentences
Total peak VRAM stays below the 4 GB limit; NER and linking run on CPU.

2.3 Eponym disambiguation (Parkinson the patient vs Parkinson's disease)
A span is treated as medical if it links to a UMLS disorder concept and appears in disease syntax (possessive + disease/dementia/syndrome, co-occurring symptoms). It is treated as a name if it follows an honorific, appears in a greeting or speaker slot, or co-refers with the patient-name slot of the note. Unresolved conflicts go to the LLM with the encounter's code list as context.

2.4 Action policy
One placeholder/surrogate map per encounter, shared between dialogue and note, with first-name ↔ full-name linking so the note never leaks what the dialogue hid.

Action	Applied to	Default output
Remove / replace	patient, relative and clinician names, contact info, institutions	typed placeholders [PATIENT], [RELATIVE], [CLINICIAN], [CLINIC] (realistic surrogates via config flag)
Generalize	exact age → 5-year band (unless an age-specific code applies); absolute dates → relative to visit ("~3 months ago"); cities → region type; hobbies/occupations → hypernym ("exercise class", "yard work") unless occupationally relevant to a code	
Perturb	non-coding numerics, within the same clinical class	
Retain	all anchored concepts from 2.1	
2.5 Utility guard (closed loop)
After editing: re-link the medical concept set and run a local ICD-10 predictor on original vs anonymized text. If an anchored concept disappears or a predicted code flips, revert that specific edit and log the reason. This optimizes the organizers' metrics (medical concept recall, ICD-10 F1) directly instead of hoping for them.

3. Bonus tasks (designed in, not bolted on)
Explainability — every edit emits {span, entity_type, action, rule_or_model, confidence, reason, code_relevance}; 100 % decision coverage by construction, plus a per-encounter diff view.
Adversarial privacy attacks — during development (larger compute allowed): an attacker LLM tries to recover name/age/date/occupation from anonymized text; a cross-field linkage attack between dialogue and note; a k-anonymity check over (age band, sex, condition set, relative mentions) across the dataset. Recovered items become new detection rules.
Efficiency — CPU NER/linking, LLM gated to a small fraction of sentences, llama.cpp/ONNX runtime; runtime, peak RAM and peak VRAM reported per encounter.
Without ICD-10 codes at inference — protect all UMLS disorder/procedure/drug/lab concepts and use the local ICD predictor's top-k as pseudo-codes for the anchor step.
Robustness — synthetic SOAP notes, discharge summaries and ER dialogues generated with a large LLM, with injected PHI as ground truth; same metrics reported.
4. Self-evaluation before the mid-term submission
Annotate PHI in the eval split with a large LLM plus manual spot-checks (permitted during development) → PHI recall/precision.
Measure medical concept recall via the UMLS linker and ICD-10 F1 via the local predictor, anonymized vs original.
Run three configurations — conservative, balanced, aggressive — and plot the privacy/utility curve the organizers ask for.
Spend the single mid-term evaluation on the balanced configuration.
5. Deliverables
anonymized_eval.jsonl — eval split with dialogue and note anonymized, icd10_codes untouched.
Repository: anchor/ (code → concept expansion), detect/ (regex, NER, GLiNER, gated LLM), policy/ (actions, surrogate map), guard/ (concept recall + ICD predictor rollback), explain/ (decision log + diff viewer), attack/ (red-team scripts), bench/ (runtime/RAM/VRAM), README.md.
