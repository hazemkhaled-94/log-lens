"""Main entry point for the log clustering pipeline."""

import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from clusterer.embedder import LogEmbedder
from clusterer.engine import LogEngine
from clusterer.evaluator import ClusterEvaluator
from clusterer.visualizer import ClusterVisualizer
from configs import INFERENCE_OUTPUT_DIR
from data_manager.logs.log_file import LogFile
from data_manager.masker.pipeline import Drain3Pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def _resolve_data_dir() -> Path:
    """Resolve the dataset directory.

    Checks CLUSTER_DATA_DIR, then DATA_DIR, then falls back to
    ``resources/datasets/mini`` relative to the project root.
    """
    project_root = Path(__file__).resolve().parents[2]
    explicit = os.getenv("CLUSTER_DATA_DIR") or os.getenv("DATA_DIR")
    if explicit:
        return Path(explicit) if Path(explicit).is_absolute() \
            else project_root / explicit
    return project_root / "resources" / "datasets" / "mini"


def _collect_embeddings(
    data_dir: Path,
    embedder: LogEmbedder,
    masker: Drain3Pipeline,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Embed all log files in ``data_dir``; return batches/levels/msgs.

    Messages are Drain3-masked first — the same preprocessing the
    classifier applies — so the clusters reflect what the model sees.
    """
    all_embeddings_list: list[np.ndarray] = []
    all_true_levels: list[str] = []
    all_messages: list[str] = []

    logger.info("Starting file-by-file embedding process...")

    for file_path in LogFile.iter_files(data_dir):
        logger.info("Processing file: %s", file_path.name)

        log_collection = LogFile.from_file(file_path)
        clean_msgs, true_levels = log_collection.get_filtered_data(
            include_unknown=False
        )

        if not clean_msgs:
            continue

        masked_msgs = [masker.mask(msg) for msg in clean_msgs]
        file_embeddings = embedder.embed_logs(masked_msgs)
        all_embeddings_list.append(file_embeddings)
        all_true_levels.extend(true_levels)
        all_messages.extend(masked_msgs)

        del log_collection
        del clean_msgs
        del true_levels
        gc.collect()

    return all_embeddings_list, all_true_levels, all_messages


def main() -> None:
    """Run the main log clustering and visualization pipeline."""
    data_dir = _resolve_data_dir()

    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(INFERENCE_OUTPUT_DIR) / "clusterer"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing models...")
    embedder = LogEmbedder()
    masker = Drain3Pipeline()
    clusterer = LogEngine()
    visualizer = ClusterVisualizer()
    evaluator = ClusterEvaluator()

    all_embeddings_list, all_true_levels, all_messages = _collect_embeddings(
        data_dir=data_dir,
        embedder=embedder,
        masker=masker,
    )

    if not all_embeddings_list:
        logger.warning("No valid log messages found. Exiting.")
        sys.exit(0)

    logger.info("Concatenating all batches into a single matrix...")
    embeddings = np.vstack(all_embeddings_list)
    logger.info(f"Final Embedding Matrix Shape: {embeddings.shape}")

    logger.info("Reducing dimensions to 100...")
    reduced_dims_embeddings = clusterer.reduce_dims(embeddings, n_dims=100)

    logger.info("Running clustering algorithm...")
    labels = clusterer.cluster(reduced_dims_embeddings)

    logger.info("Evaluating clusters...")
    evaluator.evaluate(
        all_messages, all_true_levels, labels, report_dir=str(output_dir)
    )

    logger.info("Reducing dimensions to 3 for plotting...")
    reduced_dims_plotting = clusterer.reduce_dims(embeddings, n_dims=3)

    logger.info("Generating 3D plot...")
    visualizer.plot_gmm_clusters_3d(
        reduced_dims_plotting,
        labels,
        save_path=str(output_dir / "gmm_plot_3d.png"),
        show=True,
    )


if __name__ == "__main__":
    main()
