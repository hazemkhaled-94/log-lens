# mypy: disable-error-code=import-untyped

"""High-level training and inference pipeline for Drain3 masking."""

from __future__ import annotations

import logging
from typing import Union

from data_manager.logs.log_entry import LogEntry
from data_manager.logs.log_file import LogFile
from configs import Drain3Config
from data_manager.masker.parser import Drain3Parser
from data_manager.masker.template_results import TestResult, TrainResult

logger = logging.getLogger(__name__)

_EntryLike = Union[str, LogEntry]


class Drain3Pipeline:
    """High-level Drain3 training and inference pipeline.

    Accepts strings or LogEntry objects and returns typed result objects.
    """

    def __init__(self, config: Drain3Config | None = None) -> None:
        """Initialize with an optional config; defaults to env vars."""
        self._config = config or Drain3Config()
        self._parser = Drain3Parser(self._config)

    def train(self, entry: _EntryLike) -> TrainResult:
        """Train on a single log entry (str or LogEntry)."""
        return self._parser.train(self._to_message(entry))

    def test(self, entry: _EntryLike) -> TestResult:
        """Test a single log entry against the frozen parse tree."""
        return self._parser.match(self._to_message(entry))

    def mask(self, entry: _EntryLike) -> str:
        """Return the masked form of an entry (the text the model sees)."""
        return self.test(entry).masked_message

    def train_batch(self, entries: list[_EntryLike]) -> list[TrainResult]:
        """Train on a batch of entries; empty messages are skipped."""
        return [
            self._parser.train(msg)
            for e in entries
            if (msg := self._to_message(e))
        ]

    def test_batch(self, entries: list[_EntryLike]) -> list[TestResult]:
        """Test a batch of entries; empty messages are skipped."""
        return [
            self._parser.match(msg)
            for e in entries
            if (msg := self._to_message(e))
        ]

    def train_from_directory(self) -> int:
        """Train on all log files under DRAIN3_TRAIN_DIR.

        Returns:
            Number of messages processed.

        Raises:
            FileNotFoundError: If DRAIN3_TRAIN_DIR does not exist.
        """
        processed = 0
        for entry in LogFile.yield_from_directory(self._config.train_dir):
            if entry.message:
                self._parser.train(entry.message)
                processed += 1
        logger.info(
            "Training complete. Messages: %d, Clusters: %d",
            processed,
            self._parser.cluster_count,
        )
        return processed

    def test_from_directory(self) -> tuple[int, int]:
        """Test all log files under DRAIN3_TEST_DIR.

        Returns:
            Tuple of (matched_count, unmatched_count).

        Raises:
            FileNotFoundError: If DRAIN3_TEST_DIR does not exist.
        """
        matched = 0
        unmatched = 0
        for entry in LogFile.yield_from_directory(self._config.test_dir):
            if not entry.message:
                continue
            result = self._parser.match(entry.message)
            if result.matched:
                matched += 1
            else:
                unmatched += 1
                logger.warning(
                    "Unmatched log: %s\nMasked form: %s",
                    entry.message,
                    result.masked_message,
                )
        logger.info(
            "Testing complete. Matched: %d, Unmatched: %d",
            matched,
            unmatched,
        )
        return matched, unmatched

    @staticmethod
    def _to_message(entry: _EntryLike) -> str:
        """Return the plain-text message from a str or LogEntry."""
        if isinstance(entry, str):
            return entry
        return entry.message or ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pipeline = Drain3Pipeline()
    pipeline.train_from_directory()
    pipeline.test_from_directory()
