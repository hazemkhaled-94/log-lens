"""CLI for the log classification pipeline (train / infer / evaluate)."""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelinePaths:
    """Filesystem and model settings used by the CLI pipelines."""

    root_dir: Path
    data_dir: Path
    model_dir: Path
    log_dir: Path
    output_dir: Path
    base_model: str
    test_data_dir: Path


def load_pipeline_paths() -> PipelinePaths:
    """Read pipeline paths from environment variables."""
    root_dir = Path(os.getcwd())
    return PipelinePaths(
        root_dir=root_dir,
        data_dir=root_dir / os.getenv("DATA_DIR", ""),
        model_dir=root_dir / os.getenv("MODEL_DIR", ""),
        log_dir=root_dir / os.getenv("LOG_DIR", ""),
        output_dir=root_dir / os.getenv("OUTPUT_DIR", ""),
        base_model=os.getenv("BASE_MODEL", ""),
        test_data_dir=root_dir / os.getenv("TEST_DATA_DIR", ""),
    )


def run_training(args: argparse.Namespace) -> None:
    """Orchestrate data preparation and model training."""

    from classifier.inference.batch_inference import ModelEvaluator
    from classifier.training.tokenizer import LogTokenizer
    from classifier.training.trainer_args import TrainerArgs
    from classifier.training.trainer import LogModelTrainer
    from data_manager.logs.log_dataset import LogDatasetBuilder
    from configs import seed_everything

    seed = seed_everything()
    logger.info("=== Starting Training Pipeline (seed=%d) ===", seed)

    paths: PipelinePaths = args.paths
    output_path = paths.output_dir / paths.base_model

    builder = LogDatasetBuilder()
    datasets = builder.create_splits(
        builder.build_base_dataset(paths.data_dir, max_logs=args.max_logs)
    )
    held_out_test = datasets["test"]

    tokenizer_handler = LogTokenizer(
        model_name=paths.base_model, local_cache=str(output_path)
    )
    tokenized_datasets = tokenizer_handler.tokenize_datasets(datasets)

    # held_out_test (text + label) is kept for post-training evaluation
    del datasets

    loss_functions = [args.loss_function]
    if args.train_both_losses:
        loss_functions = ["weighted_ce", "gce"]

    for loss_function in loss_functions:
        logger.info("=== Training model with loss: %s ===", loss_function)
        trainer = LogModelTrainer(
            config=TrainerArgs(
                output_dir=str(output_path),
                logging_dir=str(paths.log_dir),
                model_path=paths.base_model,
                loss_function=loss_function,
                gce_q=args.gce_q,
            )
        )
        _, save_dir = trainer.train(
            datasets=tokenized_datasets,
            tokenizer=tokenizer_handler.tokenizer,
        )

        logger.info(
            "=== Held-out test evaluation for %s (model=%s) ===",
            loss_function,
            save_dir,
        )
        ModelEvaluator(
            data_dir=paths.test_data_dir,  # unused by evaluate_from_dataset
            model_path=Path(save_dir),
            sample_size=len(held_out_test),
            batch_size=64,
        ).evaluate_from_dataset(held_out_test)


def run_inference(args: argparse.Namespace) -> None:
    """Run inference on a single log entry string."""

    from classifier.inference.inference import LogLevelPredictor
    from data_manager.logs.log_entry import LogEntry
    from data_manager.masker.pipeline import Drain3Pipeline

    logger.info("=== Starting Inference ===")
    paths: PipelinePaths = args.paths
    entry = LogEntry(raw_json_dict={"line": args.text})
    # Mask like the training pipeline so the model sees the same form.
    entry.message = Drain3Pipeline().mask(entry.message)
    LogLevelPredictor(model_path=paths.model_dir).predict(
        entry=entry, verbose=True
    )


def run_evaluation(args: argparse.Namespace) -> None:
    """Run mass evaluation of one or more models on test data.

    When ``--model-dir`` is repeated, each model is evaluated against the
    same test corpus with outputs namespaced under ``output/<model_name>/``.
    Without the flag, falls back to the ``MODEL_DIR`` env var.
    """

    from classifier.inference.batch_inference import ModelEvaluator
    from configs import seed_everything

    seed_everything()
    paths: PipelinePaths = args.paths
    model_dirs: list[Path] = (
        [Path(m) for m in args.model_dir]
        if args.model_dir
        else [paths.model_dir]
    )

    logger.info(
        "=== Starting Mass Evaluation on %d model(s) ===", len(model_dirs)
    )
    for idx, model_path in enumerate(model_dirs, start=1):
        logger.info(
            "--- [%d/%d] Evaluating %s ---",
            idx,
            len(model_dirs),
            model_path,
        )
        ModelEvaluator(
            data_dir=paths.test_data_dir,
            model_path=model_path,
            sample_size=150000,
            batch_size=64,
        ).evaluate()


def run_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI inference server."""
    from api.main import serve

    serve()


def run_cluster(args: argparse.Namespace) -> None:
    """Run the clustering pipeline over the configured data directory."""
    from clusterer.main import main as run_clusterer

    run_clusterer()


def main() -> None:
    """Parse CLI arguments and dispatch pipeline commands."""
    paths = load_pipeline_paths()

    parser = argparse.ArgumentParser(
        description="Log Classification Orchestrator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train")
    train_p.add_argument(
        "--loss-function",
        choices=["weighted_ce", "gce"],
        default="weighted_ce",
        help="Loss used for training when --train-both-losses is not set.",
    )
    train_p.add_argument(
        "--gce-q",
        type=float,
        default=0.7,
        help="Generalized cross entropy q parameter in range (0, 1].",
    )
    train_p.add_argument(
        "--train-both-losses",
        action="store_true",
        help=(
            "Train two models sequentially: "
            "one with weighted_ce and one with gce."
        ),
    )
    train_p.add_argument(
        "--max-logs",
        type=int,
        default=None,
        help=(
            "Cap the number of usable training entries pulled from "
            "DATA_DIR. Default: no cap (consume the entire corpus). "
            "Useful for fast smoke runs, e.g. --max-logs 10000."
        ),
    )
    train_p.set_defaults(func=run_training)

    infer_p = subparsers.add_parser("infer")
    infer_p.add_argument("text", help="The log line to classify")
    infer_p.set_defaults(func=run_inference)

    eval_p = subparsers.add_parser("evaluate")
    eval_p.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help=(
            "Path to a fine-tuned model directory. May be repeated to "
            "evaluate several models sequentially against the same "
            "TEST_DATA_DIR. When omitted, the MODEL_DIR env var is "
            "used. Per-model outputs are namespaced under "
            "output/<model_name>/."
        ),
    )
    eval_p.set_defaults(func=run_evaluation)

    serve_p = subparsers.add_parser("serve", help="Start the FastAPI server.")
    serve_p.set_defaults(func=run_serve)

    cluster_p = subparsers.add_parser(
        "cluster", help="Cluster logs in the model's embedding space."
    )
    cluster_p.set_defaults(func=run_cluster)

    args = parser.parse_args()
    args.paths = paths
    args.func(args)


if __name__ == "__main__":
    main()
