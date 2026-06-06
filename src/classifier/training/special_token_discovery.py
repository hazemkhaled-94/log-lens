"""Special token discovery strategies for tokenizer preparation."""

import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SpecialTokenDiscoverer(Protocol):
    """Contract for discovering candidate special tokens from datasets."""

    def discover(self, datasets: Any) -> list[str]:
        """Return a sorted list of candidate special tokens."""
        ...


class RegexSpecialTokenDiscoverer:
    """Extract placeholder-like tokens using a configurable regex pattern."""

    def __init__(self, pattern: str = r"<[^<>\s]+>") -> None:
        """Compile the regex pattern used to discover placeholder-like tokens."""
        self._pattern = re.compile(pattern)

    def discover(self, datasets: Any) -> list[str]:
        """Discover placeholder-like tokens across dataset splits."""
        discovered: set[str] = set()

        for split_name in datasets.keys():
            split_dataset = datasets[split_name]
            if "text" not in split_dataset.column_names:
                logger.warning(
                    (
                        "Split '%s' has no 'text' column; "
                        "skipping token discovery."
                    ),
                    split_name,
                )
                continue

            for text in split_dataset["text"]:
                if not text:
                    continue
                discovered.update(self._pattern.findall(text))

        return sorted(discovered)
