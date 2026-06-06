"""Severity utilities used for mismatch-based anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeverityScale:
    """Ordered severity hierarchy with normalization and rank comparison."""

    order: tuple[str, ...] = (
        "TRACE",
        "DEBUG",
        "INFO",
        "WARN",
        "ERROR",
        "FATAL",
    )

    alias_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate default aliases when none are supplied."""
        if not self.alias_map:
            object.__setattr__(
                self,
                "alias_map",
                {
                    "WARNING": "WARN",
                    "SEVERE": "ERROR",
                    "CRITICAL": "FATAL",
                    "UNKNOWN": "INFO",
                },
            )

    @property
    def max_distance(self) -> int:
        """Maximum possible rank distance across the scale."""
        return len(self.order) - 1

    def normalize(self, severity: str) -> str:
        """Return the canonical label for ``severity``, defaulting to INFO."""
        normalized = severity.upper().strip()
        alias = self.alias_map.get(normalized, normalized)
        if alias in self.order:
            return alias
        return "INFO"

    def rank(self, severity: str) -> int:
        """Return the ordinal rank of ``severity`` (higher = more severe)."""
        normalized = self.normalize(severity)
        return self.order.index(normalized)

    def distance(self, observed: str, predicted: str) -> int:
        """Return the absolute rank difference between two severities."""
        return abs(self.rank(observed) - self.rank(predicted))

    def direction(self, observed: str, predicted: str) -> str:
        """Return ``match``, ``under_prediction``, or ``over_prediction``."""
        observed_rank = self.rank(observed)
        predicted_rank = self.rank(predicted)
        if observed_rank == predicted_rank:
            return "match"
        if predicted_rank < observed_rank:
            return "under_prediction"
        return "over_prediction"

    def is_critical(self, severity: str) -> bool:
        """Return True when severity is ERROR or FATAL."""
        return self.rank(severity) >= self.rank("ERROR")

    def is_critical_miss(self, observed: str, predicted: str) -> bool:
        """True when an ERROR/FATAL event is predicted at a lower severity."""
        if not self.is_critical(observed):
            return False
        return self.rank(predicted) < self.rank(observed)
