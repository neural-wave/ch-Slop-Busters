# ch-Slop-Busters: unified-harness

<p align="center">
  <img src="image.png" alt="Slop Busters logo" width="200">
</p>

- **Hackathon:** Neural Wave 2026, challenge “Anonymization of patient data”
- **Team:** ch-Slop-Busters — Cristiano Colangelo, Alind Xhyra, Davide Molinelli, Ketai, Travaz
- **Project:** unified-harness, an offline anonymizer for patient–doctor encounters (rules + a small local
  LLM through Ollama, under 4 GB of GPU memory, no cloud)
- **License:** Apache 2.0

## Quickstart (Docker)

```bash
mkdir -p outputs
docker compose build harness
docker compose up ollama-pull
docker compose run --rm harness doctor
docker compose run --rm harness anonymize \
    --input /data/eval_patient_doctor_encounters.jsonl --out-root /outputs/docker
docker compose run --rm harness validate --original /data/eval_patient_doctor_encounters.jsonl \
    --anonymized /outputs/docker/<date>/anonymized-<date>-unified-harness.jsonl
docker compose down
```

`./data` is mounted at `/data` and `./outputs` at `/outputs`.

## Deliverables

- [`anonymized_eval_patient_doctor_encounters.jsonl`](anonymized_eval_patient_doctor_encounters.jsonl): the 67
  encounters of [`data/eval_patient_doctor_encounters.jsonl`](data/eval_patient_doctor_encounters.jsonl) anonymized
  by unified-harness in its default configuration (rules + local LLM, privacy preset)
