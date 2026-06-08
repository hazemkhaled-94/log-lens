"""Scoring logic for runtime semantic-severity mismatch anomalies."""

from __future__ import annotations

from dataclasses import dataclass, replace

from classifier.anomaly.severity import SeverityScale


@dataclass(frozen=True)
class SeverityMismatchResult:
    """Mismatch scoring output for a single log event."""

    observed_severity: str
    predicted_severity: str
    severity_distance: int
    mismatch_direction: str
    confidence: float
    risk_weight: float
    anomaly_score: float
    anomaly_threshold: float
    is_anomaly: bool


class SeverityMismatchScorer:
    """Compute direction-aware severity mismatch scores."""

    def __init__(
        self,
        scale: SeverityScale | None = None,
        under_prediction_weight: float = 1.5,
        over_prediction_weight: float = 1.0,
        threshold: float = 20.0,
    ) -> None:
        self.scale = scale or SeverityScale()
        self.under_prediction_weight = under_prediction_weight
        self.over_prediction_weight = over_prediction_weight
        self.threshold = threshold

    def score(
        self,
        observed_severity: str,
        predicted_severity: str,
        confidence: float,
    ) -> SeverityMismatchResult:
        """Score one observed/predicted severity pair."""
        observed = self.scale.normalize(observed_severity)
        predicted = self.scale.normalize(predicted_severity)
        direction = self.scale.direction(observed, predicted)
        distance = self.scale.distance(observed, predicted)

        if direction == "under_prediction":
            risk_weight = self.under_prediction_weight
        elif direction == "over_prediction":
            risk_weight = self.over_prediction_weight
        else:
            risk_weight = 1.0

        normalized_distance = (
            distance / self.scale.max_distance
            if self.scale.max_distance
            else 0.0
        )
        anomaly_score = normalized_distance * confidence * risk_weight

        return SeverityMismatchResult(
            observed_severity=observed,
            predicted_severity=predicted,
            severity_distance=distance,
            mismatch_direction=direction,
            confidence=confidence,
            risk_weight=risk_weight,
            anomaly_score=anomaly_score,
            anomaly_threshold=self.threshold,
            is_anomaly=(distance > 0 and anomaly_score >= self.threshold),
        )

    def apply_threshold(
        self,
        result: SeverityMismatchResult,
        threshold: float,
    ) -> SeverityMismatchResult:
        """Return a copy of ``result`` with the given threshold applied."""
        is_anomaly = (
            result.severity_distance > 0 and result.anomaly_score >= threshold
        )
        return replace(
            result,
            anomaly_threshold=threshold,
            is_anomaly=is_anomaly,
        )

    def update_threshold(self, threshold: float) -> None:
        """Update the default threshold for future ``score()`` calls."""
        self.threshold = threshold
