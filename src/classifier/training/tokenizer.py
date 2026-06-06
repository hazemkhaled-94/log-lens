"""Tokenization utilities for log classification."""

import logging
from typing import Any

from .special_token_discovery import (  # type: ignore[import-untyped]
    RegexSpecialTokenDiscoverer,
    SpecialTokenDiscoverer,
)
from transformers import (  # type: ignore[import-untyped]
    AutoTokenizer,
    BatchEncoding,
)

logger = logging.getLogger(__name__)


class LogTokenizer:
    """Handles batched tokenization of log datasets."""

    def __init__(
        self,
        model_name: str,
        local_cache: str | None = None,
        max_length: int = 512,
        special_token_discoverer: SpecialTokenDiscoverer | None = None,
    ) -> None:
        """Initialize tokenizer and discovery strategy.

        Args:
            model_name: Hugging Face model id or local tokenizer path.
            local_cache: Optional directory for downloaded tokenizer files.
            max_length: Maximum sequence length used during tokenization.
            special_token_discoverer: Optional discovery strategy for dynamic
                special tokens.
        """
        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=local_cache
            )
        )
        self.max_length = max_length
        self.added_special_tokens: list[str] = []
        self._special_token_discoverer = (
            special_token_discoverer or RegexSpecialTokenDiscoverer()
        )

    def tokenize_datasets(
        self,
        datasets: Any
    ) -> Any:
        """Tokenize all splits, discover special tokens, and drop the raw text column.

        Args:
            datasets: DatasetDict containing train/val/test splits.

        Returns:
            DatasetDict with encoded tensor arrays.
        """
        logger.info("Applying batched tokenization via map method...")

        discovered_tokens = self._special_token_discoverer.discover(
            datasets
        )
        added_count = self._register_special_tokens(discovered_tokens)
        logger.info(
            "Dynamic placeholder token discovery found %d candidates; "
            "added %d new special tokens.",
            len(discovered_tokens),
            added_count,
        )

        tokenized_datasets = datasets.map(
            self.tokenize_batch,
            batched=True,
            remove_columns=["text"],
        )

        logger.info("Dataset tokenization complete.")
        return tokenized_datasets

    def tokenize_batch(
        self,
        batch: dict[str, list[str]]
    ) -> BatchEncoding:
        """Tokenize a single batch of input text rows.

        Args:
            batch: Mapping containing a "text" key with raw log strings.

        Returns:
            Hugging Face BatchEncoding with tokenized fields.
        """

        encoded = self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.max_length,
        )
        return encoded

    def _register_special_tokens(self, tokens: list[str]) -> int:
        """Register newly discovered placeholder tokens in the tokenizer vocabulary."""
        existing_tokens = set(self.tokenizer.all_special_tokens)
        new_tokens = [tok for tok in tokens if tok not in existing_tokens]

        if not new_tokens:
            self.added_special_tokens = []
            return 0

        added_count = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )

        self.added_special_tokens = new_tokens if added_count else []
        return added_count
