# mypy: disable-error-code=import-untyped

"""Dataset creation utilities for log classification.

This module provides classes to convert streams of LogEntry objects
into structured Hugging Face Dataset objects and manage dataset splits.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from datasets import ClassLabel, Dataset, DatasetDict  # type: ignore

from configs import RANDOM_STATE
from data_manager.logs.log_labels import LogLevel

# Both stratified splits (80/20, then 50/50 on the 20%) need each
# class to have at least one example on either side of the cut.
# Empirically, classes with fewer than four rows in the source set
# can land with a single member in the temp partition and crash the
# second split; four is the conservative floor that keeps the 3-way
# split well-defined.
_MIN_EXAMPLES_PER_CLASS_FOR_STRATIFIED_SPLIT = 4


load_dotenv()
logger = logging.getLogger(__name__)


class LogDatasetBuilder:
    """Creates Hugging Face datasets from log entries with label maps."""

    def build_base_dataset(
        self,
        data_dir: Path,
        max_logs: Optional[int] = None,
    ) -> Dataset:
        """Converts LogEntry objects into a Hugging Face Dataset.

        Entries whose level doesn't resolve to a LogLevel (e.g. UNKNOWN)
        are skipped.

        Args:
            data_dir: Directory containing input log files (.json/.log).
            max_logs: Optional cap on entries emitted; None consumes all.

        Returns:
            A Hugging Face Dataset with 'text' and 'label' columns.
        """
        logger.info(
            "Creating base Hugging Face Dataset from generator (cap=%s)",
            max_logs if max_logs is not None else "no cap",
        )

        def gen(directory_path: str, max_logs: Optional[int]):
            """Yield {text, label} dicts; self-contained for pickling."""
            from pathlib import Path
            from data_manager.logs.log_file import LogFile
            from data_manager.logs.log_labels import LogLevel
            from configs import Drain3Config
            from data_manager.masker.pipeline import Drain3Pipeline

            iterator = LogFile.yield_from_directory(Path(directory_path))

            pipeline = Drain3Pipeline(
                config=Drain3Config()
            )
            processed = 0

            for entry in iterator:
                if max_logs is not None and processed >= max_logs:
                    break

                lvl = entry.line_level.upper()

                entry.message = pipeline.mask(entry.message)

                try:
                    numeric_label = LogLevel[lvl].value
                    yield {
                        "text": entry.message,
                        "label": numeric_label,
                    }
                    processed += 1
                except KeyError as err:
                    logger.debug(f"Skipping entry with unmapped label: {err}")
                    continue

        return Dataset.from_generator(
            gen,
            gen_kwargs={"directory_path": str(data_dir), "max_logs": max_logs}
        )

    def create_splits(
        self,
        dataset: Dataset,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = RANDOM_STATE,
    ) -> DatasetDict:
        """Splits a dataset into train, validation, and test sets.

        The label column is cast to ClassLabel over the full LogLevel
        namespace so integer IDs are stable even when some classes are
        absent. Classes with fewer than
        _MIN_EXAMPLES_PER_CLASS_FOR_STRATIFIED_SPLIT rows are dropped
        (with a warning) to keep the stratified 3-way split well-defined.

        Args:
            dataset: The Hugging Face Dataset to split.
            train_size: Fraction for training (default: 0.8).
            val_size: Fraction for validation (default: 0.1).
            test_size: Fraction for testing (default: 0.1).
            seed: Random seed for reproducibility.

        Returns:
            A DatasetDict with 'train', 'validation', and 'test' keys.

        Raises:
            ValueError: If split ratios do not sum to 1.0.
        """

        logger.info("Casting label column to ClassLabel...")
        class_label = ClassLabel(
            names=[name for _, name in sorted(LogLevel.id2label().items())]
        )
        dataset = dataset.cast_column("label", class_label)

        label_counts = Counter(dataset["label"])
        too_rare = {
            label_id: count
            for label_id, count in label_counts.items()
            if count < _MIN_EXAMPLES_PER_CLASS_FOR_STRATIFIED_SPLIT
        }
        if too_rare:
            dropped_summary = {
                class_label.int2str(lid): cnt
                for lid, cnt in too_rare.items()
            }
            logger.warning(
                "Dropping %d row(s) from %d class(es) with fewer than "
                "%d examples — too rare to stratify across a 3-way "
                "split: %s",
                sum(too_rare.values()),
                len(too_rare),
                _MIN_EXAMPLES_PER_CLASS_FOR_STRATIFIED_SPLIT,
                dropped_summary,
            )
            keep_ids = set(label_counts) - set(too_rare)
            dataset = dataset.filter(lambda ex: ex["label"] in keep_ids)

        if abs(train_size + val_size + test_size - 1.0) > 1e-5:
            logger.error("Split ratios must sum to 1.0.")
            raise ValueError("Split ratios must sum to 1.0.")

        logger.info("Applying STRATIFIED split into train/temp...")

        test_val_ratio = val_size + test_size
        train_test_split = dataset.train_test_split(
            test_size=test_val_ratio,
            seed=seed,
            stratify_by_column="label"
        )

        logger.info(
            "Applying STRATIFIED split of temp set into validation/test..."
        )

        val_ratio_adjusted = val_size / test_val_ratio
        val_test_split = train_test_split["test"].train_test_split(
            test_size=1.0 - val_ratio_adjusted,
            seed=seed,
            stratify_by_column="label"
        )

        logger.info("Stratified dataset splitting complete.")
        return DatasetDict(
            {
                "train": train_test_split["train"],
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            }
        )
