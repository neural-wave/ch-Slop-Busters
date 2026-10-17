"""ICD-10 coders that stand in for the organizers' coder in the utility metric (registry: eval/coders.toml).

base.py       Coder protocol, Prediction, CoderError
codes.py      canonical code spelling, 3-character categories
bootstrap.py  set scores, paired bootstrap over encounters
ollama.py     Ollama client and the LLM coders (llm_note, llm_both)
knn.py  lexical.py  command.py  text.py   the lexical stand-ins and external commands
"""
