"""Benchmark custom fine-tuned models against open-source baselines."""
# type: ignore  # transformers stubs are incomplete

import logging
import random
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm  # type: ignore

from data_manager.logs.log_file import LogFile

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmarks and compares log classification models."""

    def __init__(
        self,
        data_dir: Path,
        custom_model_path: str,
        hf_model_path: str = "byviz/bylastic_classification_logs",
        sample_size: int = 10000,
        batch_size: int = 64
    ) -> None:
        """Initialize the benchmark.

        Args:
            data_dir: Path to directory containing log JSON files.
            custom_model_path: Path or HF model ID for custom model.
            hf_model_path: HF model ID for reference model.
            sample_size: Number of log entries to sample.
            batch_size: Batch size for inference.
        """
        self.data_dir = data_dir
        self.custom_model_path = custom_model_path
        self.hf_model_path = hf_model_path
        self.sample_size = sample_size
        self.batch_size = batch_size

        self.texts: list[str] = []
        self.true_labels: list[str] = []
        self.actual_sample_size = 0
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.results: dict[str, float] = {}

    def run(self) -> None:
        """Run the full benchmark."""
        self._load_data()
        self._benchmark()
        self._print_results()

    def _load_data(self) -> None:
        """Load and filter log entries."""
        logger.info(f"Scanning directory {self.data_dir} for logs...")
        aggregate_log_file = LogFile.from_directory(self.data_dir)

        valid_entries = [
            entry for entry in aggregate_log_file.entries
            if entry.line_level.upper() != "UNKNOWN"
        ]

        # Sample from valid entries for consistent comparison
        self.actual_sample_size = min(
            self.sample_size, len(valid_entries)
        )
        sampled_entries = random.sample(
            valid_entries, self.actual_sample_size
        )

        # Extract texts and labels for model evaluation
        self.texts = [entry.message for entry in sampled_entries]
        self.true_labels = [
            entry.line_level.upper() for entry in sampled_entries
        ]

    def _benchmark(self) -> None:
        """Evaluate both models on the same dataset."""
        models_to_test = {
            "Custom Fine-Tuned Model": self.custom_model_path,
            "Byviz Open-Source Model": self.hf_model_path
        }

        for model_name, model_path in models_to_test.items():
            logger.info("\n" + "="*70)
            logger.info(f" INITIALIZING: {model_name}")
            logger.info(f" Path: {model_path}")
            logger.info("="*70)

            accuracy = self._evaluate_model(model_name, model_path)
            self.results[model_name] = accuracy

    def _evaluate_model(self, model_name: str, model_path: str) -> float:
        """Evaluate a single model on the dataset.

        Args:
            model_name: Name of the model for display.
            model_path: Path or HF model ID.

        Returns:
            Accuracy percentage as a float.
        """
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True
            ).to(self.device)
        )
        model.eval()

        correct_predictions = 0

        # Process texts in batches
        for i in tqdm(
            range(0, len(self.texts), self.batch_size),
            desc=f"Scoring {model_name}"
        ):
            batch_texts = self.texts[i: i + self.batch_size]
            batch_true_labels = (
                self.true_labels[i: i + self.batch_size]
            )

            # Tokenize batch input
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding="max_length"
            ).to(self.device)

            # Forward pass without gradient computation
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            predicted_class_ids = (
                torch.argmax(logits, dim=-1).tolist()
            )

            # Compare predictions with ground truth labels
            for pred_id, true_label in zip(
                predicted_class_ids, batch_true_labels
            ):
                predicted_label = (
                    model.config.id2label.get(
                        pred_id, f"ID_{pred_id}"
                    ).upper()
                )

                if predicted_label == true_label:
                    correct_predictions += 1

        # Calculate accuracy percentage
        accuracy = (correct_predictions / self.actual_sample_size) * 100

        # Clean up GPU/RAM memory
        del model
        del tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return accuracy

    def _print_results(self) -> None:
        """Print the benchmark results."""
        logger.info("\n" + "🏆"*25)
        logger.info(" BENCHMARK SHOWDOWN RESULTS")
        logger.info("🏆"*25)
        logger.info(
            f"Dataset Size: {self.actual_sample_size} exact same logs\n"
        )

        for model_name, acc in self.results.items():
            logger.info(f"{model_name:<30}: {acc:.2f}% Accuracy")
        logger.info("="*70)
