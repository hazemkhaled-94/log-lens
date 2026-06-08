# Model Card — log-lens severity classifier

A fine-tuned [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
encoder that predicts the **severity level** of a single log line
(`TRACE / DEBUG / INFO / WARN / ERROR / FATAL`).

Inputs are Drain3-masked before the model sees them; this repository ships the
**Drain3 preprocessing pipeline** that produces that masked form (used
identically in training, inference, the API, and clustering).

## Model details

- **Architecture:** `ModernBERT-base` with a sequence-classification head, 6
  output classes.
- **Input:** one log message, Drain3-masked (variables replaced by
  placeholders such as `<IP>`, `<UUID>`, `<HEX>`, `<NUM>`, `<CLASS_NAME>`,
  `<HOST>`, `<PID>`, `<CONN_ID>`, `<COMPOSITE_ID>`, `<MASKED_INFO>`),
  tokenized to a maximum length of 512.
- **Output:** a severity label + confidence. The API additionally derives a
  semantic-severity *mismatch* anomaly score (observed vs predicted level).
- **Two checkpoints**, identical in every respect except the training loss:
  - **WCE** — Weighted Cross-Entropy →
    [modernlogbert-wce](https://huggingface.co/hazemkhaled-94/modernlogbert-wce).
  - **WGCE** — Weighted Generalized Cross-Entropy (noise-tolerant, `q = 0.7`) →
    [modernlogbert-gce](https://huggingface.co/hazemkhaled-94/modernlogbert-gce).

## Intended use

- **Intended:** triage/observability research — predicting or sanity-checking
  log severity, and flagging entries whose predicted severity disagrees with
  the emitted level as candidate anomalies.
- **Out of scope:** a sole source of truth for alerting or incident severity.
  Performance can vary on unfamiliar log formats, so validate on your own
  distribution and keep a human in the loop.

## Training data

- **In-distribution:** a publicly available collection of system log corpora
  (loghub-style), preprocessed into a level-balanced, stratified sample with
  [`DatasetPreprocessor`](src/data_manager/logs/dataset_preprocessor.py).
- **Out-of-distribution (evaluation only):** a single private industrial
  Kubernetes log deployment. It is **not** included in this repository and was
  used purely as an OOD generalization probe.

## Training procedure

Both checkpoints share these hyperparameters (see
[`trainer_args.py`](src/classifier/training/trainer_args.py)):

| Hyperparameter | Value |
|---|---|
| Backbone | ModernBERT-base |
| Epochs | 8 |
| Per-device batch size | 32 |
| Gradient accumulation | 4 (effective batch 128) |
| Learning rate | 1e-5 (separate LRs for head vs backbone) |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Max sequence length | 512 |
| Best-model metric | `eval_f1` |
| Seed | `RANDOM_STATE` (env, default 94) |

## Evaluation results

### In-distribution (held-out stratified slice)

| Metric | WCE | WGCE |
|---|---|---|
| Accuracy | **88.18%** | 87.37% |
| Macro precision | 0.7647 | 0.7368 |
| Macro recall | 0.8271 | 0.7977 |
| Macro F1 | **0.7813** | 0.7447 |
| Weighted F1 | 0.8947 | 0.8884 |
| Mean confidence (all) | 90.17% | 95.57% |
| Mean confidence (correct) | 95.31% | 97.85% |

On the curated in-distribution slice, **WCE is the stronger and
better-calibrated model**. WGCE is consistently more confident — the
calibration cost of a noise-tolerant objective.

### Out-of-distribution (industrial Kubernetes deployment)

Evaluated on a private industrial Kubernetes domain — a different log
distribution than training. Both checkpoints **degraded modestly but stayed
usable**, the expected cost of moving to unfamiliar formats and frameworks.
WGCE's noise-tolerant design shows up as **~21% fewer under-predictions** than
WCE on this domain. Behavior on other distributions is unverified.

## Limitations and biases

- **OOD generalization.** Only modest degradation was observed on a single
  private industrial domain; other distributions are unverified — validate on
  your own logs.
- **Confidence ≠ correctness**, particularly for WGCE (the more confident of
  the two). Treat scores as signals, not guarantees.
- **Preprocessing coupling.** Inputs must be Drain3-masked exactly as in
  training; raw text yields degraded predictions.

## How to use

```bash
log-lens infer "your log line here"
log-lens serve                 # FastAPI inference server
```

Set `MODEL_DIR` (in `.env`) to the checkpoint you want to serve — either a
local path or a published Hugging Face Hub repo id. The Docker image fetches
the published checkpoint at build time (see the README).
