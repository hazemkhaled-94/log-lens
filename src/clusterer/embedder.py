"""Embedding utilities for log clustering.

Embeds log messages with the same fine-tuned ModernBERT checkpoint the
classifier serves, so clustering reflects the representation space the
model actually operates in.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore

from configs import MODEL_DIR


class LogEmbedder:
    """Generate dense embeddings from the trained classifier encoder."""

    def __init__(self) -> None:
        """Load the tokenizer and encoder from the trained MODEL_DIR."""
        model_path = Path(MODEL_DIR)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[2] / model_path

        self.device = self._select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_logs(
        self, logs: list[str], batch_size: int = 64
    ) -> np.ndarray:
        """Encode masked log messages into mean-pooled hidden-state vectors."""
        vectors: list[np.ndarray] = []
        for start in range(0, len(logs), batch_size):
            batch = logs[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            pooled = self._mean_pool(hidden, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors)

    @staticmethod
    def _select_device() -> torch.device:
        """Pick MPS, then CUDA, then CPU."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _mean_pool(
        hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool token embeddings, ignoring padding via the mask."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
