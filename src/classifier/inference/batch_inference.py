"""Mass evaluation module for the trained log classification model."""
# type: ignore
# The transformers library has incomplete type stubs, causing false positives.
# These are suppressed to allow the module to type-check cleanly while the
# runtime behavior is correct.

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional

from sklearn.metrics import (  # type: ignore
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from configs import INFERENCE_OUTPUT_DIR
from classifier.anomaly.calibration import PercentileThresholdCalibrator
from classifier.anomaly.scorer import (
    SeverityMismatchResult,
    SeverityMismatchScorer,
)
from classifier.inference.inference import LogLevelPredictor
from data_manager.logs.log_entry import LogEntry
from data_manager.logs.log_file import LogFile
from data_manager.logs.log_labels import LogLevel
from data_manager.masker.pipeline import Drain3Pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates a trained log classification model on a dataset."""

    PredictionTuple = Tuple[LogEntry, str, str]

    def __init__(
        self,
        data_dir: Path,
        model_path: Path,
        sample_size: int = 10000,
        batch_size: int = 128,
    ) -> None:
        """Initialize the evaluator."""
        self.data_dir = data_dir
        self.model_path = model_path
        self.sample_size = sample_size
        self.batch_size = batch_size

        self.predictor: Optional[LogLevelPredictor] = None
        self.drain3: Drain3Pipeline = Drain3Pipeline()

        self.entries: List[LogEntry] = []
        self.true_labels: List[str] = []
        self.predictions_log: List[
            Tuple[ModelEvaluator.PredictionTuple, float]
        ] = []
        self.mismatch_results: List[SeverityMismatchResult] = []

        base_threshold = float(os.getenv("ANOMALY_SCORE_THRESHOLD", "20.0"))
        under_weight = float(
            os.getenv("ANOMALY_UNDER_PREDICTION_WEIGHT", "1.5")
        )
        over_weight = float(os.getenv("ANOMALY_OVER_PREDICTION_WEIGHT", "1.0"))
        calibration_percentile = float(
            os.getenv("ANOMALY_CALIBRATION_PERCENTILE", "95.0")
        )

        self.mismatch_scorer = SeverityMismatchScorer(
            threshold=base_threshold,
            under_prediction_weight=under_weight,
            over_prediction_weight=over_weight,
        )
        self.threshold_calibrator = PercentileThresholdCalibrator(
            percentile=calibration_percentile,
            min_threshold=base_threshold,
        )
        self.correct_predictions = 0
        self.run_id: str = ""
        self._file_handler: Optional[logging.FileHandler] = None

    def evaluate(self) -> None:
        """Run the full evaluation pipeline against ``self.data_dir``."""
        self._init_run_artifacts()
        try:
            logger.info("Starting full evaluation pipeline...")
            self._load_data()
            self._sanitize_entries()
            self._load_model()
            self._log_sample_inputs()
            self._run_inference()
            self._generate_report()
            self._generate_anomaly_report()
            self._generate_error_analysis()
            self._export_auto_labeled_logs()
            self._export_incorrect_predictions()
        finally:
            self._teardown_run_artifacts()

    def evaluate_from_dataset(
        self,
        dataset,
        sample_size: Optional[int] = None,
    ) -> None:
        """Run the full evaluation pipeline on an in-memory dataset.

        Use this when the test split already lives in memory (e.g. the
        held-out split produced right after training). Skips disk load
        and Drain3 sanitization; all other steps run identically to
        :meth:`evaluate`.

        Args:
            dataset: HF Dataset with ``text`` (Drain3-masked) and
                ``label`` (integer ``LogLevel`` id) columns.
            sample_size: Optional cap on rows used (random subsample).
        """
        from data_manager.logs.log_entry import LogEntry
        from data_manager.logs.log_labels import LogLevel

        self._init_run_artifacts()
        try:
            logger.info(
                "Starting in-memory evaluation pipeline on the held-out "
                "test split (model=%s)...",
                self.model_path,
            )

            n = len(dataset)
            if sample_size is not None and sample_size < n:
                sampled_indices = random.sample(range(n), sample_size)
                dataset = dataset.select(sampled_indices)

            self.entries = []
            self.true_labels = []
            for row in dataset:
                entry = LogEntry(raw_line=row["text"])
                # Override the regex-derived fields. Drain3 masking strips
                # the level token from the message before training, so the
                # parser would re-classify these as UNKNOWN; the integer
                # label is the source of truth here.
                entry.message = row["text"]
                try:
                    level_name = LogLevel(int(row["label"])).name
                except ValueError:
                    level_name = "UNKNOWN"
                entry.line_level = level_name
                self.entries.append(entry)
                self.true_labels.append(level_name)

            logger.info(
                "Loaded %d entries from in-memory test split.",
                len(self.entries),
            )

            self._load_model()
            self._log_sample_inputs()
            self._run_inference()
            self._generate_report()
            self._generate_anomaly_report()
            self._generate_error_analysis()
            self._export_auto_labeled_logs()
            self._export_incorrect_predictions()
        finally:
            self._teardown_run_artifacts()

    def _init_run_artifacts(self) -> None:
        """Stamp the run with a unique id and attach a per-run log handler."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.model_path.name}_{timestamp}"

        out_dir = self._default_export_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{self.run_id}_report.txt"

        handler = logging.FileHandler(report_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(handler)
        self._file_handler = handler

        logger.info("Persisting evaluation report to %s", report_path)

    def _teardown_run_artifacts(self) -> None:
        """Detach this run's report file handler and flush its buffer."""
        if self._file_handler is None:
            return
        logging.getLogger().removeHandler(self._file_handler)
        self._file_handler.close()
        self._file_handler = None

    def _load_data(self) -> None:
        """Load and sanitize log data from disk."""

        logger.info(f"Scanning directory {self.data_dir} for logs...")

        aggregate_log_file = LogFile.from_directory(self.data_dir)

        if not aggregate_log_file.entries:
            logger.error("No valid logs found in the directory.")
            return

        logger.info(
            f"Found {len(aggregate_log_file.entries)} "
            "total log entries across all files."
        )

        actual_sample_size = min(
            self.sample_size, len(aggregate_log_file.entries)
        )
        sampled_entries = random.sample(
            aggregate_log_file.entries, actual_sample_size
        )
        self.entries = sampled_entries

        for entry in sampled_entries:
            self.true_labels.append(LogLevel.canonicalize(entry.line_level))

        logger.info(
            f"Successfully sampled {actual_sample_size} clean log entries."
        )

    def _sanitize_entries(self) -> None:
        """Mask each entry's message via Drain3; skips empty messages."""
        logger.info("Sanitizing messages via Drain3 masker...")
        sanitized = 0
        for entry in self.entries:
            if not entry.message:
                continue
            result = self.drain3.test(entry.message)
            entry.message = result.masked_message
            sanitized += 1
        logger.info(f"Sanitized {sanitized} log messages.")

    def _load_model(self) -> None:
        """Load model and tokenizer onto device."""
        logger.info("Initializing LogLevelPredictor engine...")
        self.predictor = LogLevelPredictor(self.model_path)

    def _log_sample_inputs(self, num_samples: int = 30) -> None:
        """Log a few samples to show exactly what the model sees."""
        logger.info("\n" + "=" * 70)
        logger.info(" SAMPLE MODEL INPUTS (MASKED TEXT SEEN BY TOKENIZER)")
        logger.info("=" * 70)

        if not self.entries:
            return

        samples = random.sample(
            self.entries, min(num_samples, len(self.entries))
        )
        for i, entry in enumerate(samples, 1):
            logger.info(f"Sample {i}: '{entry.message}'\n")

        logger.info("=" * 70 + "\n")

    def _run_inference(self) -> None:
        """Run batched inference on all texts."""
        logger.info("Starting mass inference...")
        self.correct_predictions = 0
        self.predictions_log = []
        self.mismatch_results = []

        if not self.predictor:
            raise RuntimeError("Predictor engine not loaded!")

        predicted_labels, confidence_intervals = self.predictor.predict_batch(
            self.entries, batch_size=self.batch_size
        )

        for entry, predicted_label, true_label, confidence in zip(
            self.entries,
            predicted_labels,
            self.true_labels,
            confidence_intervals,
        ):
            self.predictions_log.append(
                ((entry, true_label, predicted_label), confidence)
            )

            mismatch = self.mismatch_scorer.score(
                observed_severity=true_label,
                predicted_severity=predicted_label,
                confidence=confidence,
            )
            self.mismatch_results.append(mismatch)

            if predicted_label.upper() == true_label.upper():
                self.correct_predictions += 1

        self._calibrate_anomaly_threshold()

        logger.info(
            "Inference complete. "
            f"{self.correct_predictions}/{len(self.entries)} "
            "correct predictions."
        )

        all_confidences = [conf for _, conf in self.predictions_log]
        if all_confidences:
            avg_conf = sum(all_confidences) / len(all_confidences)
            correct_confs = [
                conf
                for (_, true_l, pred_l), conf in self.predictions_log
                if pred_l.upper() == true_l.upper()
            ]
            wrong_confs = [
                conf
                for (_, true_l, pred_l), conf in self.predictions_log
                if pred_l.upper() != true_l.upper() and true_l != "UNKNOWN"
            ]
            avg_correct = (
                sum(correct_confs) / len(correct_confs)
                if correct_confs
                else 0.0
            )
            avg_wrong = (
                sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0.0
            )
            logger.info(
                f"Avg Confidence (all):     {avg_conf:.2f}%\n"
                f"Avg Confidence (correct): {avg_correct:.2f}%\n"
                f"Avg Confidence (wrong):   {avg_wrong:.2f}%"
            )

    def _calibrate_anomaly_threshold(self) -> None:
        """Calibrate and apply threshold to mismatch results.

        The calibration uses a percentile of the observed score
        distribution so anomaly volume can be controlled without
        requiring externally labeled anomaly data.
        """
        if not self.mismatch_results:
            return

        scores = [result.anomaly_score for result in self.mismatch_results]
        calibrated_threshold = self.threshold_calibrator.calibrate(scores)
        self.mismatch_scorer.update_threshold(calibrated_threshold)

        self.mismatch_results = [
            self.mismatch_scorer.apply_threshold(
                result=result,
                threshold=calibrated_threshold,
            )
            for result in self.mismatch_results
        ]

    @staticmethod
    def _precision_at_k(
        y_true: list[int],
        y_score: list[float],
        k: int,
    ) -> float:
        """Compute precision at top-k scored events."""
        if k <= 0 or not y_true or not y_score:
            return 0.0

        ranked = sorted(
            zip(y_true, y_score),
            key=lambda item: item[1],
            reverse=True,
        )
        top_k = ranked[:k]
        positives = sum(label for label, _ in top_k)
        return positives / len(top_k)

    def _generate_anomaly_report(self) -> None:
        """Generate anomaly-focused metrics for mismatch scoring."""
        if not self.mismatch_results:
            return

        logger.info("\n" + "=" * 70)
        logger.info(" SEMANTIC-SEVERITY MISMATCH ANOMALY REPORT")
        logger.info("=" * 70)

        threshold = self.mismatch_scorer.threshold
        anomaly_count = sum(
            1 for result in self.mismatch_results if result.is_anomaly
        )
        anomaly_rate = (
            (anomaly_count / len(self.mismatch_results)) * 100
            if self.mismatch_results
            else 0.0
        )

        logger.info(f"Calibrated Threshold: {threshold:.4f}")
        logger.info(
            "Flagged Anomalies:    %d/%d",
            anomaly_count,
            len(self.mismatch_results),
        )
        logger.info(f"Anomaly Rate:         {anomaly_rate:.2f}%")

        y_true = [
            int(
                self.mismatch_scorer.scale.is_critical_miss(
                    result.observed_severity,
                    result.predicted_severity,
                )
            )
            for result in self.mismatch_results
        ]
        y_score = [result.anomaly_score for result in self.mismatch_results]
        positive_count = sum(y_true)

        logger.info(
            "Target Event: critical misses (ERROR/FATAL under-prediction)"
        )
        logger.info(f"Critical Misses:       {positive_count}")

        if positive_count > 0 and positive_count < len(y_true):
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            p_at_k = self._precision_at_k(y_true, y_score, positive_count)
            logger.info(f"AUROC:                {auroc:.4f}")
            logger.info(f"AUPRC:                {auprc:.4f}")
            logger.info(f"Precision@K:          {p_at_k:.4f}")
        else:
            logger.info(
                "AUROC/AUPRC skipped: dataset lacks both positive and "
                "negative critical-miss examples."
            )

        direction_counts = Counter(
            result.mismatch_direction for result in self.mismatch_results
        )
        logger.info("Mismatch Direction Counts:")
        for direction, count in sorted(direction_counts.items()):
            logger.info(f"  {direction:<17} {count}")

        logger.info("=" * 70)

    def _generate_report(self) -> None:
        """Generate and log the evaluation report."""

        known_predictions = [
            p for p in self.predictions_log if p[0][1] != "UNKNOWN"
        ]
        actual_sample_size = len(known_predictions)

        accuracy = 0.0
        if actual_sample_size > 0:
            accuracy = (self.correct_predictions / actual_sample_size) * 100

        logger.info("\n" + "=" * 70)
        logger.info(" MASS EVALUATION REPORT")
        logger.info("=" * 70)
        known_confidences = [conf for _, conf in known_predictions]
        avg_conf = (
            sum(known_confidences) / len(known_confidences)
            if known_confidences
            else 0.0
        )

        logger.info(f"Total Logs Evaluated: {len(self.entries)}")
        logger.info(f"Known Logs Evaluated: {actual_sample_size}")
        logger.info(f"Correct Predictions:  {self.correct_predictions}")
        logger.info(f"Overall Accuracy:     {accuracy:.2f}%")
        logger.info(f"Avg Confidence:       {avg_conf:.2f}%\n")

        y_true = [true_l for (_, true_l, _), _ in known_predictions]
        y_pred = [pred_l for (_, _, pred_l), _ in known_predictions]

        if y_true:
            metrics_dict = classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            )

            logger.info("--- Accuracy (Recall) and Precision per Class ---")

            class_labels = sorted(list(set(y_true)))

            # Build per-class confidence buckets (keyed by true label)
            class_confidences: dict = {lbl: [] for lbl in class_labels}
            for (_, true_l, _), conf in known_predictions:
                if true_l in class_confidences:
                    class_confidences[true_l].append(conf)

            for label in class_labels:
                class_metrics = metrics_dict[label]  # type: ignore
                class_acc = class_metrics["recall"] * 100  # type: ignore
                class_prec = class_metrics["precision"] * 100  # type: ignore
                support = class_metrics["support"]  # type: ignore
                confs = class_confidences[label]
                avg_class_conf = sum(confs) / len(confs) if confs else 0.0

                logger.info(
                    f"{label:<8}: "
                    f"Accuracy: {class_acc:>6.2f}% | "
                    f"Precision: {class_prec:>6.2f}% | "
                    f"Avg Confidence: {avg_class_conf:>6.2f}% | "
                    f"Support: {support}"
                )

        logger.info("=" * 70)

        unknown_predictions = [
            pred_l
            for (_, true_l, pred_l), _ in self.predictions_log
            if (true_l == "UNKNOWN")
        ]

        if unknown_predictions:
            logger.info("\n--- UNKNOWN Logs Distribution ---")
            logger.info("Here is how the model categorized your UNKNOWN logs:")
            u_counts = Counter(unknown_predictions)
            for label, count in u_counts.items():
                logger.info(f"{label:<8}: {count} logs")
            logger.info("=" * 70)

        if y_true:
            logger.info(
                "\n--- Detailed Classification Report (Known Labels Only) ---"
            )
            report_str = classification_report(
                y_true, y_pred, digits=4, zero_division=0
            )
            logger.info(f"\n{report_str}")

    def _generate_error_analysis(self) -> None:
        """Generate and log error analysis."""
        logger.info("\n" + "-" * 70)
        logger.info(" ERROR ANALYSIS: INCORRECT PREDICTIONS")
        logger.info("-" * 70)

        # Ignore UNKNOWN labels in the error analysis
        incorrect_predictions = [
            (entry, true_l, pred_l, conf)
            for (entry, true_l, pred_l), conf in self.predictions_log
            if true_l != pred_l and true_l != "UNKNOWN"
        ]

        if not incorrect_predictions:
            logger.info("🎉 INCREDIBLE! Zero incorrect predictions found.")
        else:
            logger.info(
                f"Found {len(incorrect_predictions)} "
                "total incorrect predictions."
            )
            logger.info(
                "Here is a random sample of where the model got confused:\n"
            )

            num_errors = min(10, len(incorrect_predictions))
            samples = random.sample(incorrect_predictions, num_errors)

            for entry, true_label, pred_label, confidence in samples:
                logger.info(f"LOG:  '{entry.raw_line}'")
                logger.info(
                    f"TRUE: {true_label:<7} | "
                    f"PRED: {pred_label:<7} | "
                    f"CONF: {confidence:.2f}% "
                )

        logger.info("=" * 70)

    def _default_export_dir(self) -> Path:
        """Per-model output dir under the inference output dir."""
        return Path(INFERENCE_OUTPUT_DIR) / self.model_path.name

    def _export_auto_labeled_logs(
        self,
        output_file: Optional[str] = None,
    ) -> None:
        """Export pseudo-labeled UNKNOWN logs to JSON for future training."""
        if output_file is None:
            output_file = str(
                self._default_export_dir()
                / f"{self.run_id}_auto_labeled_logs.json"
            )
        logger.info(f"Exporting auto-labeled logs to {output_file}...")

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        new_dataset = []
        for (
            entry,
            true_label,
            pred_label,
        ), confidence in self.predictions_log:
            if true_label == "UNKNOWN":
                new_dataset.append(
                    {
                        "text": entry.raw_line,
                        "masked_text": entry.message,
                        "raw_json": entry.raw_json_dict,
                        "label": pred_label,
                        "confidence": round(confidence, 4),
                    }
                )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, indent=4, ensure_ascii=False)

        logger.info(f"Successfully saved {len(new_dataset)} logs! 🚀")

    def _export_incorrect_predictions(
        self,
        output_file: Optional[str] = None,
    ) -> None:
        """Export wrongly classified logs to a JSON file for auditing.

        Args:
            output_file: Target JSON path for exported misclassifications.
                Defaults to ``output/<model_name>/incorrect_predictions.json``.
        """
        if output_file is None:
            output_file = str(
                self._default_export_dir()
                / f"{self.run_id}_incorrect_predictions.json"
            )
        logger.info(f"Exporting incorrect predictions to {output_file}...")

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        incorrect_dataset = []
        for ((entry, true_label, pred_label), confidence), mismatch in zip(
            self.predictions_log,
            self.mismatch_results,
        ):
            if true_label != pred_label and true_label != "UNKNOWN":
                incorrect_dataset.append(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": round(confidence, 4),
                        "mismatch_direction": mismatch.mismatch_direction,
                        "mismatch_distance": mismatch.severity_distance,
                        "anomaly_score": round(mismatch.anomaly_score, 4),
                        "anomaly_threshold": round(
                            mismatch.anomaly_threshold,
                            4,
                        ),
                        "is_anomaly": mismatch.is_anomaly,
                        "raw_text": entry.raw_line,
                        "masked_text_seen_by_model": entry.message,
                        "raw_json": entry.raw_json_dict,
                    }
                )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(incorrect_dataset, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Successfully saved {len(incorrect_dataset)} "
            "incorrect predictions for audit!"
        )
