"""Tests for the severity scale used in mismatch anomaly detection."""

from classifier.anomaly.severity import SeverityScale


def test_normalize_resolves_aliases_and_defaults_to_info():
    scale = SeverityScale()
    assert scale.normalize("warning") == "WARN"
    assert scale.normalize("CRITICAL") == "FATAL"
    assert scale.normalize(" error ") == "ERROR"
    # Unknown labels collapse to INFO.
    assert scale.normalize("nonsense") == "INFO"


def test_rank_and_distance():
    scale = SeverityScale()
    assert scale.rank("TRACE") == 0
    assert scale.rank("FATAL") == 5
    assert scale.max_distance == 5
    assert scale.distance("INFO", "ERROR") == 2
    assert scale.distance("ERROR", "INFO") == 2


def test_direction():
    scale = SeverityScale()
    assert scale.direction("INFO", "INFO") == "match"
    # Real event more severe than prediction -> under-prediction.
    assert scale.direction("ERROR", "INFO") == "under_prediction"
    assert scale.direction("INFO", "ERROR") == "over_prediction"


def test_critical_miss():
    scale = SeverityScale()
    assert scale.is_critical("ERROR") is True
    assert scale.is_critical("INFO") is False
    assert scale.is_critical_miss("ERROR", "WARN") is True
    assert scale.is_critical_miss("INFO", "DEBUG") is False
