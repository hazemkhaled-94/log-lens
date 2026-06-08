# log-lens

Tools for making sense of infrastructure logs. It masks noisy log lines with
Drain3, fine-tunes a ModernBERT classifier to predict log severity, serves
predictions over a FastAPI endpoint, and flags entries whose predicted
severity disagrees with the emitted level as possible anomalies. A clustering
module groups the masked logs in the trained model's own embedding space — a
way to inspect what the classifier sees.

See [MODEL_CARD.md](MODEL_CARD.md) for the model, training data, metrics, and
limitations. Trained weights are on the Hugging Face Hub:
[modernlogbert-wce](https://huggingface.co/hazemkhaled-94/modernlogbert-wce)
and [modernlogbert-gce](https://huggingface.co/hazemkhaled-94/modernlogbert-gce).

## Layout

- `src/api` — FastAPI service and model loading.
- `src/classifier` — training, inference, anomaly scoring, evaluation.
- `src/clusterer` — ModernBERT embeddings, GMM clustering, plotting.
- `src/data_manager/logs` — log parsing, dataset building, corpus sampling.
- `src/data_manager/masker` — Drain3 masking and templates.
- `src/configs` — central, environment-driven configuration.

## Setup

Requires Python 3.12+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
cp .env.example .env          # then edit the paths in .env
poetry run pre-commit install # optional: enable lint/type hooks on commit
```

## Usage

`poetry install` registers three commands — `log-lens` (training + inference),
`log-lens-cluster` (visualization), and `log-lens-sample` (data prep):

```bash
# Training
log-lens train                       # train on DATA_DIR (--loss-function weighted_ce|gce)

# Inference
log-lens infer "failed to connect"   # classify a single line
log-lens evaluate --model-dir models/<base>/<run>   # batch-evaluate against TEST_DATA_DIR
log-lens serve                       # start the FastAPI server

# Visualization
log-lens-cluster                     # cluster logs in the model's embedding space

# Data prep: build a balanced training corpus from a raw log collection
log-lens-sample --src <raw_corpus> --out data/train/sample --total 200000
```

Prefix with `poetry run` if you haven't activated the virtualenv. Each command
is also runnable as `python -m <module>` (e.g. `python -m classifier.main`).

Inputs are Drain3-masked before the model sees them, identically across
training, inference, the API, and clustering. Configuration lives in `.env`;
see `.env.example` for every variable.

## Outputs

Inference artifacts are written under `INFERENCE_OUTPUT_DIR` (default
`output/`), organized per operation: `output/clusterer/` for clustering
reports/plots and `output/<model>/` for batch-evaluation exports. Trained
checkpoints, logs, and tensorboard events are gitignored.

## Datasets

The public log corpora used for training and evaluation (HDFS, Hadoop, Spark,
BGL, Thunderbird, OpenStack, Apache, Android, Zookeeper, Windows, etc.) come
from the **[loghub](https://github.com/logpai/loghub)** collection. They are
not committed to this repo — download them from loghub into your `DATA_DIR`,
or build a balanced subset with `log-lens-sample`.

## Docker

Two images: a CPU **inference** image with the model baked in, and a GPU
**training** image that trains on your own data.

### Inference (CPU) — use the published model directly

The image bakes in [modernlogbert-wce](https://huggingface.co/hazemkhaled-94/modernlogbert-wce)
by default, so it's ready to use with no mounts:

```bash
docker build -t log-lens .                # bakes in the default model
docker run -p 8000:8000 log-lens
```

Override `--build-arg MODEL_REPO=<hf_repo_id>` to bake in a different model
(private repos additionally need an `HF_TOKEN` build secret). To publish your
own checkpoint: `hf auth login` then
`make publish-model MODEL_REPO=<user>/<name> MODEL_DIR=models/<base>/<run>`.

### Training (GPU) — train on your own dataset

No model is baked in; mount your data and an output directory. Requires the
NVIDIA Container Toolkit; match the CUDA tag in `Dockerfile.train` to your
driver.

```bash
docker build -f Dockerfile.train -t log-lens-train .
docker run --gpus all \
  -v "$PWD/data:/app/data" -v "$PWD/output:/app/output" \
  -e DATA_DIR=data -e TEST_DATA_DIR=data \
  -e OUTPUT_DIR=output -e LOGGING_DIR=output/logs \
  log-lens-train
```

The trained checkpoint is written to the mounted `output/` volume.

## Development

```bash
poetry run pytest        # tests
poetry run flake8 src tests
poetry run mypy src
```

CI runs the same checks on every push and pull request
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## License

Apache-2.0 — see [LICENSE](LICENSE).
