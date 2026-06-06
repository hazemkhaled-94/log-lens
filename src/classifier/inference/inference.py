"""Inference module for log classification predictions."""
# type: ignore
# The transformers library has incomplete type stubs, causing false positives.
# These are suppressed to allow the module to type-check cleanly while the
# runtime behavior is correct.

from pathlib import Path
from typing import List, Optional, Tuple
import logging
import torch
from tqdm.auto import tqdm  # type: ignore
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from data_manager.logs.log_entry import LogEntry

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class LogLevelPredictor:
    """Loads a trained log classification model and performs predictions."""

    def __init__(self, model_path: Path) -> None:
        """Initialize the predictor with a trained model."""
        self.model_path = model_path
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")
        self._load_model()
        self.id2label = self.model.config.id2label  # type: ignore

    def _load_model(self) -> None:
        """Load the tokenizer and model from disk."""
        logger.info(f"Loading model from {self.model_path}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore
                self.model_path,
                trust_remote_code=True
            )
            self.model = (  # type: ignore
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            )
            assert self.model is not None, "Model failed to instantiate."
            self.model.to(device=self.device)  # type: ignore
            self.model.eval()  # type: ignore
            logger.info("Model loaded successfully.")
        except Exception as e:
            msg = f"Failed to load model from {self.model_path}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    @torch.no_grad()
    def predict(
        self,
        entry: LogEntry,
        verbose: bool = True
    ) -> Tuple[str, float, int]:
        """Predict the log level for the given log text.

        Args:
            entry: The log entry to classify.
            verbose: If True, log the prediction details.

        Returns:
            A tuple of (predicted_label, confidence, predicted_class_id).
        """
        inputs = self.tokenizer(  # type: ignore
            entry.message,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)  # type: ignore
        logits = outputs.logits
        predicted_class_id = int(torch.argmax(logits, dim=-1).item())
        predicted_label = self.id2label.get(
            predicted_class_id, f"CLASS_ID_{predicted_class_id}"
        ).upper()

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = (
            probabilities[0][predicted_class_id].item() * 100
        )

        if verbose:
            self._log_prediction_details(
                entry.message,
                logits[0].tolist(),
                predicted_class_id,
                predicted_label,
                confidence
            )

        return predicted_label, confidence, predicted_class_id

    @torch.no_grad()
    def predict_batch(
        self,
        entries: List[LogEntry],
        batch_size: int
    ) -> Tuple[List[str], List[float]]:
        """Run optimized batch inference for many log entries.

        Args:
            entries: Log entries to classify.
            batch_size: Number of records processed per forward pass.

        Returns:
            Tuple of predicted labels and confidence percentages.
        """
        predictions = []
        confidence_levels = []
        logger.info(
            f"Starting batch inference on {len(entries)} "
            f"entries (batch_size={batch_size})..."
        )

        total_batches = (len(entries) + batch_size - 1) // batch_size

        progress = tqdm(
            range(0, len(entries), batch_size),
            total=total_batches,
            desc="Batch inference",
            unit="batch",
        )
        for i in progress:
            batch = entries[i: i + batch_size]
            batch_texts = [entry.message for entry in batch]

            inputs = self.tokenizer(  # type: ignore
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)  # type: ignore
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidences, pred_ids = torch.max(probs, dim=-1)

            for confidence, pred_id in zip(
                confidences.tolist(),
                pred_ids.tolist()
            ):
                predictions.append(
                    self.id2label.get(
                        pred_id, "UNKNOWN"
                    ).upper()
                )
                confidence_levels.append(confidence * 100)

            del inputs, outputs, probs, confidences, pred_ids

            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info(
            "Batch inference complete. "
            f"{len(predictions)} predictions made."
        )
        return predictions, confidence_levels

    @staticmethod
    def _log_prediction_details(
        log_text: str,
        logits: list,
        predicted_class_id: int,
        predicted_label: str,
        confidence: float
    ) -> None:
        """Log prediction details in a formatted way."""
        logger.info("="*60)
        logger.info(f"INPUT LOG:   '{log_text}'")
        logger.info(f"RAW LOGITS:  {logits}")
        logger.info(f"PREDICTED ID:{predicted_class_id}")
        logger.info(
            f"FINAL LABEL: {predicted_label} (Confidence: {confidence:.2f}%)"
        )
        logger.info("="*60)
