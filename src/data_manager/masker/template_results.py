"""Typed result models returned by Drain3 parser operations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Drain3Result:
    """Base result with the resolved template string."""

    template: str


@dataclass(frozen=True)
class TrainResult(Drain3Result):
    """Result produced after training on a single log message."""

    cluster_id: int
    is_new_cluster: bool


@dataclass(frozen=True)
class TestResult(Drain3Result):
    """Result produced after matching a single log message."""

    matched: bool
    masked_message: str
