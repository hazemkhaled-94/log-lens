"""Central, environment-derived settings shared across the project."""

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
class Settings:
    """Typed configuration values used across classifier and clusterer."""

    model_dir: str
    inference_output_dir: str
    n_clusters: int
    random_state: int


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        model_dir=os.getenv("MODEL_DIR", ""),
        inference_output_dir=os.getenv("INFERENCE_OUTPUT_DIR", "output"),
        n_clusters=_get_int("N_CLUSTERS", 2),
        random_state=_get_int("RANDOM_STATE", 94),
    )


_SETTINGS = load_settings()

MODEL_DIR = _SETTINGS.model_dir
INFERENCE_OUTPUT_DIR = _SETTINGS.inference_output_dir
N_CLUSTERS = _SETTINGS.n_clusters
RANDOM_STATE = _SETTINGS.random_state
