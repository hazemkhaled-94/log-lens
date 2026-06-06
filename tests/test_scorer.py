"""Tests for the semantic-severity mismatch scorer."""

from classifier.anomaly.scorer import SeverityMismatchScorer


def test_match_is_not_an_anomaly():
    scorer = SeverityMismatchScorer()
    result = scorer.score("INFO", "INFO", confidence=100.0)
    assert result.severity_distance == 0
    assert result.mismatch_direction == "match"
    assert result.anomaly_score == 0.0
    assert result.is_anomaly is False


def test_under_prediction_uses_under_weight_and_flags_anomaly():
    scorer = SeverityMismatchScorer(
        under_prediction_weight=1.5, threshold=20.0
    )
    result = scorer.score("ERROR", "INFO", confidence=100.0)
    # distance 2/5 * confidence 100 * weight 1.5 = 60.
    assert result.mismatch_direction == "under_prediction"
    assert result.risk_weight == 1.5
    assert result.anomaly_score == 60.0
    assert result.is_anomaly is True


def test_over_prediction_uses_over_weight():
    scorer = SeverityMismatchScorer(over_prediction_weight=1.0)
    result = scorer.score("INFO", "ERROR", confidence=50.0)
    assert result.mismatch_direction == "over_prediction"
    assert result.risk_weight == 1.0


def test_apply_threshold_returns_updated_copy():
    scorer = SeverityMismatchScorer(threshold=20.0)
    result = scorer.score("ERROR", "INFO", confidence=100.0)
    relaxed = scorer.apply_threshold(result, threshold=100.0)
    assert relaxed.anomaly_threshold == 100.0
    assert relaxed.is_anomaly is False
    # Original is untouched (frozen dataclass copy).
    assert result.anomaly_threshold == 20.0
