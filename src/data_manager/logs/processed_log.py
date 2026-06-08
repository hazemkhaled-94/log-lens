# mypy: disable-error-code=import-untyped

"""Typed object that binds raw logs with Drain3 masking output."""

from dataclasses import dataclass

from data_manager.logs.log_entry import LogEntry
from data_manager.masker.template_results import TestResult


@dataclass
class ProcessedLog:
    """Binds a raw log entry to its Drain3 analysis for ML inference."""

    original_entry: LogEntry
    drain3_result: TestResult

    @property
    def ml_input_text(self) -> str:
        """Template text when matched, otherwise the masked message."""
        if self.drain3_result.matched:
            return self.drain3_result.template
        return self.drain3_result.masked_message
