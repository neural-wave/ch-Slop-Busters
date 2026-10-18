# ch-Slop-Busters: Anonymization for Patient data
<p align="center">
  <img src="image.png" alt="Slop Busters logo" width="200">
</p>

- **Hackathon:** Neural Wave 2026, challenge “Anonymization for Patient data”
- **Team:** ch-Slop-Busters — Cristiano Colangelo · Ketai Qiu · Daniel Travaglia · Alind Xhyra · Davide Molinelli
- **Project:** Ensemble + Generalizer, an offline anonymizer for patient–doctor encounters (rules + a small local LLM through Ollama, under 4 GB of GPU memory, no cloud)
- **License:** Apache 2.0

## Quickstart (Docker)

[Install Docker](https://docs.docker.com/engine/install/)

```bash
mkdir -p outputs
docker compose build harness
docker compose up ollama-pull
docker compose run --rm harness doctor
docker compose run --rm harness anonymize \
    --input /data/eval_patient_doctor_encounters.jsonl --out-root /outputs/docker
docker compose down
```

`./data` is mounted at `/data` and `./outputs` at `/outputs`.

## Deliverables

- [`anonymized_eval_patient_doctor_encounters.jsonl`](deliverables/anonymized_eval_patient_doctor_encounters.jsonl): the 67 encounters of the evaluation dataset anonymized by unified-harness in its default configuration (rules + local LLM, privacy preset)
- [`report`](deliverables/Slop%20Busters%20Report.pdf)
- [`video presentation`](https://www.swisstransfer.com/dl/01a14db5-c1bd-70d3-8331-16470295c63a)