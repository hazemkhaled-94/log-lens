# log-lens

Tools for making sense of infrastructure logs. It masks noisy log lines with
Drain3, fine-tunes a transformer to predict log severity, serves predictions
over a FastAPI endpoint, and flags entries whose predicted severity disagrees
with the reported one as possible anomalies. There's also a clustering module
that groups the masked logs in the trained model's own embedding space — a way
to inspect what the classifier sees.

## Layout

- `src/api` — FastAPI service and model loading.
- `src/classifier` — training, inference, anomaly scoring, evaluation.
- `src/clusterer` — ModernBERT embeddings, GMM clustering, plotting.
- `src/data_manager/logs` — log parsing and dataset building.
- `src/data_manager/masker` — Drain3 masking and templates.

## Setup

Requires Python 3.12+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
cp .env.example .env   # then edit the paths in .env
```

## Usage

```bash
# Train on the logs in DATA_DIR
poetry run python -m classifier.main train

# Classify a single line
poetry run python -m classifier.main infer "[ERROR] failed to connect"

# Evaluate trained models against TEST_DATA_DIR
poetry run python -m classifier.main evaluate --model-dir models/<base>/<run>

# Serve the API
poetry run uvicorn api.main:app --host "$HOST" --port "$PORT"
```

Training loss can be switched with `--loss-function weighted_ce|gce`.
Configuration lives in `.env`; see `.env.example` for every variable.

## Notes

Model checkpoints, evaluation outputs, and tensorboard events are gitignored.

## License

MIT
