"""Runtime dependency container for API inference services.

This module centralizes startup-time construction of heavy runtime
dependencies used by the API layer.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

from fastapi import FastAPI

from classifier.anomaly.scorer import SeverityMismatchScorer
from classifier.inference.inference import LogLevelPredictor
from data_manager.masker.pipeline import Drain3Pipeline
from dotenv import load_dotenv

load_dotenv()


class ModelContainer:
    """Manage lifecycle for API runtime dependencies.

    Attributes:
        predictor: Lazy-loaded log-level predictor singleton.
        mismatch_scorer: Lazy-loaded anomaly scorer singleton.
        inference_batch_size: Default batch size used for inference.
        model_path: Filesystem path used to load the predictor.
    """

    predictor: LogLevelPredictor | None = None
    mismatch_scorer: SeverityMismatchScorer | None = None
    masker: Drain3Pipeline | None = None
    inference_batch_size: int = 64
    model_path: str = ""

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        """
        Read a float env var; return ``default`` if missing/invalid.
        """
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        """
        Read an int env var; return ``default`` if missing/invalid.
        """
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _build_mismatch_scorer() -> SeverityMismatchScorer:
        """
        Build a SeverityMismatchScorer from environment configuration.
        """
        threshold = ModelContainer._read_float_env(
            "ANOMALY_SCORE_THRESHOLD", 20.0
        )
        under_weight = ModelContainer._read_float_env(
            "ANOMALY_UNDER_PREDICTION_WEIGHT", 1.5
        )
        over_weight = ModelContainer._read_float_env(
            "ANOMALY_OVER_PREDICTION_WEIGHT", 1.0
        )

        return SeverityMismatchScorer(
            threshold=threshold,
            under_prediction_weight=under_weight,
            over_prediction_weight=over_weight,
        )

    @classmethod
    def load(cls, model_path: str) -> LogLevelPredictor:
        """Load singleton runtime components from ``model_path``."""
        cls.model_path = model_path

        if cls.predictor is None:
            cls.predictor = LogLevelPredictor(Path(model_path))
        if cls.mismatch_scorer is None:
            cls.mismatch_scorer = cls._build_mismatch_scorer()
        if cls.masker is None:
            cls.masker = Drain3Pipeline()
        cls.inference_batch_size = cls._read_int_env(
            "INFERENCE_BATCH_SIZE", 64
        )
        return cls.predictor

    @classmethod
    def get_runtime_components(cls) -> "RuntimeComponents":
        """Return loaded runtime dependencies for request handlers.

        Raises:
            RuntimeError: If startup did not initialize dependencies.
        """
        if cls.predictor is None:
            raise RuntimeError("Model predictor is not initialized.")
        if cls.mismatch_scorer is None:
            raise RuntimeError("Mismatch scorer is not initialized.")
        if cls.masker is None:
            raise RuntimeError("Masker is not initialized.")
        return RuntimeComponents(
            predictor=cls.predictor,
            mismatch_scorer=cls.mismatch_scorer,
            masker=cls.masker,
            batch_size=cls.inference_batch_size,
        )

    @classmethod
    def get_runtime_snapshot(cls) -> dict[str, object]:
        """Return runtime configuration and readiness diagnostics."""
        return {
            "predictor_loaded": cls.predictor is not None,
            "mismatch_scorer_loaded": cls.mismatch_scorer is not None,
            "masker_loaded": cls.masker is not None,
            "model_path": cls.model_path,
            "inference_batch_size": cls.inference_batch_size,
            "anomaly_threshold": (
                cls.mismatch_scorer.threshold
                if cls.mismatch_scorer is not None
                else None
            ),
            "anomaly_under_prediction_weight": (
                cls.mismatch_scorer.under_prediction_weight
                if cls.mismatch_scorer is not None
                else None
            ),
            "anomaly_over_prediction_weight": (
                cls.mismatch_scorer.over_prediction_weight
                if cls.mismatch_scorer is not None
                else None
            ),
        }


class RuntimeComponents(NamedTuple):
    """Typed runtime bundle injected into API services."""

    predictor: LogLevelPredictor
    mismatch_scorer: SeverityMismatchScorer
    masker: Drain3Pipeline
    batch_size: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down API runtime dependencies."""

    model_path = os.getenv("MODEL_DIR", "")
    ModelContainer.load(model_path)
    yield

    ModelContainer.predictor = None
    ModelContainer.mismatch_scorer = None
    ModelContainer.masker = None
