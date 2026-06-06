
"""Embedding utilities for log clustering."""

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from classifier.utils.config import MODEL_NAME, PREFIX_CLUSTERING


class LogEmbedder:
    """Generate dense embeddings for log messages."""

    def __init__(self):
        """Load the embedding model and save it to the local resources dir."""
        self.model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        project_root = Path(__file__).parent.parent.parent
        model_dir = MODEL_NAME.replace("/", "_")
        download_path = (
            project_root / "resources" / "models" / model_dir
        )
        self.model.save(str(download_path))

    def embed_logs(self, logs: List[str]) -> np.ndarray:
        """Encode log messages into vector embeddings."""
        prefixed_logs = [f"{PREFIX_CLUSTERING}{log}" for log in logs]

        embeddings = self.model.encode(
            prefixed_logs,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings
