"""Pydantic request and response schemas for the API layer."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class LogFieldsSchema(BaseModel):
    """Maps to LogMetadata in log_entry.py — captures the standard
    infrastructure metadata injected by the logging agent."""
    app: str = Field(
        default="",
        description="Application name"
    )
    container: str = Field(
        default="",
        description="Container ID or name"
    )
    detected_level: str = Field(
        default="unknown",
        description="Level detected by the log agent"
    )
    filename: str = Field(
        default="",
        description="Source log file path"
    )
    job: str = Field(
        default="",
        description="Loki/Promtail job label"
    )
    namespace: str = Field(
        default="",
        description="Kubernetes namespace"
    )
    node_name: str = Field(
        default="",
        description="Kubernetes node name"
    )
    pod: str = Field(
        default="",
        description="Kubernetes pod name"
    )
    service_name: str = Field(
        default="",
        description="Service name"
    )
    stream: str = Field(
        default="",
        description="Log stream (stdout/stderr)"
    )


class LogRequest(BaseModel):
    """Single log entry as stored in the dataset JSON files."""
    timestamp: str = Field(..., description="Timestamp string")
    line: str = Field(..., min_length=1, description="Raw log line")
    fields: LogFieldsSchema = Field(default_factory=LogFieldsSchema)


class BatchRequest(BaseModel):
    """Wrapper for a batch of log entries submitted to /predict/batch."""
    logs: List[LogRequest]


class PredictionResponse(BaseModel):
    """Result for a single log entry prediction."""
    observed_label: str = Field(
        description=(
            "Runtime developer-assigned severity extracted "
            "from the log"
        )
    )
    raw_line: str = Field(
        description="Original log line before sanitisation"
    )
    predicted_label: str = Field(
        description="Predicted log level (e.g. ERROR, WARN)"
    )
    confidence: float = Field(
        description="Model confidence score (0-100)"
    )
    mismatch_direction: str = Field(
        description="match, under_prediction, or over_prediction"
    )
    mismatch_distance: int = Field(
        description=(
            "Absolute severity-rank distance between "
            "observed and predicted labels"
        )
    )
    anomaly_score: float = Field(
        description="Direction-aware semantic-severity mismatch score"
    )
    anomaly_threshold: float = Field(
        description=(
            "Threshold used to convert anomaly score "
            "into a binary decision"
        )
    )
    is_anomaly: bool = Field(
        description=(
            "True when mismatch score exceeds threshold "
            "and severity differs"
        )
    )
    metadata: Dict[str, Any] = Field(
        description="Infrastructure metadata from the log fields"
    )


class BatchResponse(BaseModel):
    """Aggregated response for a batch prediction request."""
    total: int = Field(description="Number of log entries processed")
    predictions: List[PredictionResponse]


class ReadinessResponse(BaseModel):
    """Readiness and dependency status for the API runtime."""

    status: str = Field(
        description="Service readiness status: ready or not_ready"
    )
    predictor_loaded: bool = Field(
        description="Whether the predictor singleton is initialized"
    )
    mismatch_scorer_loaded: bool = Field(
        description="Whether the mismatch scorer singleton is initialized"
    )
    model_path: str = Field(description="Model path configured at startup")
    inference_batch_size: int = Field(
        description="Batch size used by runtime inference"
    )


class RuntimeConfigResponse(BaseModel):
    """Current runtime knobs used by anomaly and inference logic."""

    model_path: str = Field(description="Filesystem model path")
    inference_batch_size: int = Field(
        description="Batch size used for predictor batch inference"
    )
    anomaly_threshold: float = Field(
        description="Anomaly score threshold for flagging events"
    )
    anomaly_under_prediction_weight: float = Field(
        description="Risk multiplier when severity is under-predicted"
    )
    anomaly_over_prediction_weight: float = Field(
        description="Risk multiplier when severity is over-predicted"
    )


class AnomalyScoreRequest(BaseModel):
    """Request payload for standalone anomaly scoring."""

    observed_severity: str = Field(
        description="Observed runtime severity label"
    )
    predicted_severity: str = Field(
        description="Predicted model severity label"
    )
    confidence: float = Field(
        ge=0.0,
        le=100.0,
        description="Prediction confidence percentage in range [0, 100]",
    )


class AnomalyScoreResponse(BaseModel):
    """Standalone semantic-severity mismatch scoring response."""

    observed_severity: str = Field(
        description="Canonical observed severity"
    )
    predicted_severity: str = Field(
        description="Canonical predicted severity"
    )
    mismatch_direction: str = Field(
        description="match, under_prediction, or over_prediction"
    )
    mismatch_distance: int = Field(
        description="Absolute rank distance on severity scale"
    )
    confidence: float = Field(
        description="Input confidence percentage"
    )
    anomaly_score: float = Field(
        description="Computed mismatch anomaly score"
    )
    anomaly_threshold: float = Field(
        description="Threshold used for anomaly decision"
    )
    is_anomaly: bool = Field(
        description="True when event crosses anomaly threshold"
    )
