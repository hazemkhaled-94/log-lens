
"""Configuration utilities for Drain3-based log masking."""

import os
from pathlib import Path

from dotenv import load_dotenv
from drain3.template_miner_config import (  # type: ignore
    MaskingInstruction,
    TemplateMinerConfig,
)

load_dotenv()


class Drain3Config:
    """Drain3 configuration resolved from environment variables.

    Environment variables consumed:
        DRAIN3_STATE_FILE: Path to the binary persistence file.
        DRAIN3_TRAIN_DIR: Directory scanned during training.
        DRAIN3_TEST_DIR: Directory scanned during testing.
    """

    def __init__(self) -> None:
        """Load paths from environment variables."""
        self.state_file: str = os.getenv(
            "DRAIN3_STATE_FILE", ""
        )
        self.train_dir: Path = Path(
            os.getenv("DRAIN3_TRAIN_DIR", "")
        )
        self.test_dir: Path = Path(
            os.getenv("DRAIN3_TEST_DIR", "")
        )

    def build_miner_config(self) -> TemplateMinerConfig:
        """Build a TemplateMinerConfig with ordered masking rules."""
        cfg = TemplateMinerConfig()
        cfg.masking_instructions = self._masking_rules()
        return cfg

    @staticmethod
    def _masking_rules() -> list[MaskingInstruction]:
        """Return ordered masking instructions applied before tokenisation."""
        return [
            MaskingInstruction(
                r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", ""
            ),
            MaskingInstruction(r"\[[^\]]+\]", "<MASKED_INFO>"),
            MaskingInstruction(
                r"\b(?:[a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]+\b",
                "<CLASS_NAME>",
            ),
            MaskingInstruction(
                r"\b\d+(?:_\d+)+\b", "<COMPOSITE_ID>"
            ),
            MaskingInstruction(
                r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>"
            ),
            MaskingInstruction(
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
                "<UUID>",
            ),
            MaskingInstruction(r"\b[0-9a-fA-F]{10,}\b", "<HEX>"),
            MaskingInstruction(
                r'(host:\s*")[^"]+(\")', r"\1<HOST>\2"
            ),
            MaskingInstruction(r"\b\d+#\d+:", "<PID>:"),
            MaskingInstruction(r"\*\d+\b", "<CONN_ID>"),
            MaskingInstruction(r"\b\d+\b", "<NUM>"),
        ]
