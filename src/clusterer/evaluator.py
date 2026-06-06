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
        """Log per-cluster purity stats and write anomalies to a report file."""
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        file_path = report_path / "anomalies_report.txt"

        logger.info("\n" + "=" * 50)
        logger.info("                 CLUSTER EVALUATION")
        logger.info("=" * 50)

        unique_clusters = np.unique(cluster_labels)
        all_mismatches = []

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

            mismatches = [
                f"[Cluster {cluster_id}] Tagged as "
                f"[{levels_in_cluster[i]}]: {msgs_in_cluster[i]}"
                for i in range(total)
                if levels_in_cluster[i] != majority_label
            ]

            if mismatches:
                logger.info(f"  [!] Found {len(mismatches)} anomalies.")
                all_mismatches.extend(mismatches)

        logger.info("\n" + "=" * 50)

        if all_mismatches:
            file_path.write_text("\n".join(all_mismatches), encoding="utf-8")
            logger.info(
                f"-> Full anomaly report saved to: "
                f"{file_path.resolve()}\n"
            )
        else:
            logger.info("-> Perfect clustering! No anomalies found.\n")
