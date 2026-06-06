"""Environment-based configuration for the log classifier."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable, raising ValueError if non-int."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as err:
        raise ValueError(f"Environment variable {name} must be int.") from err


@dataclass(frozen=True)
class ClassifierConfig:
    """Typed configuration values for classifier and clusterer modules."""

    model_name: str
    prefix_clustering: str
    n_clusters: int
    random_state: int


def load_config() -> ClassifierConfig:
    """Load classifier settings from environment variables."""
    return ClassifierConfig(
        model_name=os.getenv("MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"),
        prefix_clustering=os.getenv("PREFIX_CLUSTERING", "clustering: "),
        n_clusters=_get_int("N_CLUSTERS", 2),
        random_state=_get_int("RANDOM_STATE", 94),
    )


_CONFIG = load_config()

# Backward-compatible exports used throughout the codebase.
MODEL_NAME = _CONFIG.model_name
PREFIX_CLUSTERING = _CONFIG.prefix_clustering
N_CLUSTERS = _CONFIG.n_clusters
RANDOM_STATE = _CONFIG.random_state
