"""Evaluation utilities for log clustering analysis."""

import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """Evaluates cluster purity against ground-truth log levels."""

    def evaluate(
        self,
        messages: list[str],
        true_levels: list[str],
        cluster_labels: np.ndarray,
        report_dir: str = "data",
    ) -> None:
        """Log per-cluster purity stats and write outliers to a report file."""
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        file_path = report_path / "outliers_report.txt"

        logger.info("\n" + "=" * 50)
        logger.info("                 CLUSTER EVALUATION")
        logger.info("=" * 50)

        unique_clusters = np.unique(cluster_labels)
        all_outliers = []

        for cluster_id in unique_clusters:
            indices = np.where(cluster_labels == cluster_id)[0]

            levels_in_cluster = [true_levels[i] for i in indices]
            msgs_in_cluster = [messages[i] for i in indices]

            counts = Counter(levels_in_cluster)

            if not counts:
                continue

            majority_label = counts.most_common(1)[0][0]
            total = len(levels_in_cluster)
            majority_count = counts[majority_label]

            accuracy = (majority_count / total) * 100

            logger.info(
                f"\n[ Cluster {cluster_id} ] - "
                f"Dominant Label: '{majority_label}'"
            )
            logger.info(
                f"Purity: {accuracy:.2f}% "
                f"({majority_count}/{total} logs match)"
            )

            for level, count in counts.items():
                logger.info(f"  -> Contains {count} '{level}' logs")

            outliers = [
                f"[Cluster {cluster_id}] Tagged as "
                f"[{levels_in_cluster[i]}]: {msgs_in_cluster[i]}"
                for i in range(total)
                if levels_in_cluster[i] != majority_label
            ]

            if outliers:
                logger.info(f"  [!] Found {len(outliers)} outliers.")
                all_outliers.extend(outliers)

        logger.info("\n" + "=" * 50)

        if all_outliers:
            file_path.write_text("\n".join(all_outliers), encoding="utf-8")
            logger.info(
                f"-> Full outlier report saved to: {file_path.resolve()}\n"
            )
        else:
            logger.info("-> Perfect clustering! No outliers found.\n")
