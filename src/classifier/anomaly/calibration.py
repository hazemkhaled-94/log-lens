"""Threshold calibration strategies for anomaly scores."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ThresholdCalibrator(ABC):
    """Abstract contract for threshold calibration strategies."""

    @abstractmethod
    def calibrate(self, scores: list[float]) -> float:
        """Return a scalar anomaly threshold derived from ``scores``."""


class PercentileThresholdCalibrator(ThresholdCalibrator):
    """Calibrate threshold using a score percentile."""

    def __init__(
        self,
        percentile: float = 95.0,
        min_threshold: float = 0.0,
    ) -> None:
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError("percentile must be in [0, 100]")
        self.percentile = percentile
        self.min_threshold = min_threshold

    def calibrate(self, scores: list[float]) -> float:
        """Return the threshold at the configured percentile."""
        if not scores:
            return self.min_threshold
        values = np.asarray(scores, dtype=float)
        return float(np.percentile(values, self.percentile))
