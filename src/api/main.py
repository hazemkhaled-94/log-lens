"""FastAPI entrypoint for batch log-level inference.

This module defines HTTP endpoints and wires request handling to an
inference service that operates on pre-loaded runtime dependencies.
"""

import logging
import os
from typing import List

import uvicorn  # type: ignore
from fastapi import FastAPI, HTTPException

from api.schemas import (
    AnomalyScoreRequest,
    AnomalyScoreResponse,
    BatchResponse,
    LogRequest,
    PredictionResponse,
    ReadinessResponse,
    RuntimeConfigResponse,
)
from api.services import ModelContainer, RuntimeComponents, lifespan
from data_manager.logs.log_entry import LogEntry

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Log Classification Service",
    description="""
    Fine-tuned ModernBert API for infrastructure log leveling.

    ## Capabilities
    * **Batch Prediction**: Process multiple logs in
    a single request for high-efficiency.
    * **Health Monitoring**: Check model status and system readiness.
    """,
    version="1.0.0",
    contact={
        "name": "Hazem Khaled",
        "email": "hazem.mkhaled@icloud.com",
    },
    lifespan=lifespan,
)


class InferenceService:
    """Handle core batch inference business logic."""

    def __init__(self, runtime: RuntimeComponents) -> None:
        self._predictor = runtime.predictor
        self._mismatch_scorer = runtime.mismatch_scorer
        self._masker = runtime.masker
        self._batch_size = runtime.batch_size

    def process_one(self, request: LogRequest) -> PredictionResponse:
        """Run prediction and anomaly scoring for a single log entry."""
        return self.process_batch([request])[0]

    def process_batch(
        self,
        requests: List[LogRequest],
    ) -> List[PredictionResponse]:
        """Run semantic severity prediction and mismatch scoring."""

        if not requests:
            return []

        logger.info("Processing batch of %d log entries", len(requests))

        entries = [
            LogEntry(raw_json_dict=req.model_dump()) for req in requests
        ]
        for entry in entries:
            entry.message = self._masker.mask(entry.message)

        labels, confidence_intervals = self._predictor.predict_batch(
            entries,
            batch_size=self._batch_size,
        )

        results: list[PredictionResponse] = []
        for entry, label, confidence in zip(
            entries,
            labels,
            confidence_intervals,
        ):
            mismatch = self._mismatch_scorer.score(
                observed_severity=entry.line_level,
                predicted_severity=label,
                confidence=confidence,
            )

            results.append(
                PredictionResponse(
                    observed_label=mismatch.observed_severity,
                    raw_line=entry.raw_line,
                    predicted_label=mismatch.predicted_severity,
                    confidence=round(confidence, 4),
                    mismatch_direction=mismatch.mismatch_direction,
                    mismatch_distance=mismatch.severity_distance,
                    anomaly_score=round(mismatch.anomaly_score, 4),
                    anomaly_threshold=round(mismatch.anomaly_threshold, 4),
                    is_anomaly=mismatch.is_anomaly,
                    metadata=entry.metadata.__dict__,
                )
            )

        logger.info("Batch complete — %d predictions returned", len(results))
        return results


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(logs: List[LogRequest]):
    """Predict severities for a batch of logs."""
    logger.info("POST /predict/batch — received %d entries", len(logs))
    try:
        runtime = ModelContainer.get_runtime_components()
        service = InferenceService(runtime)
        results = service.process_batch(logs)
        logger.info(
            "POST /predict/batch — responded with %d predictions", len(results)
        )
        return BatchResponse(total=len(results), predictions=results)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error during batch inference")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(log: LogRequest):
    """Predict severity for a single log."""
    logger.info("POST /predict - single prediction requested")
    try:
        runtime = ModelContainer.get_runtime_components()
        service = InferenceService(runtime)
        return service.process_one(log)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error during single inference")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Confirm service liveness and model availability."""
    if ModelContainer.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Inference attempted before model was loaded in lifespan.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/ready", response_model=ReadinessResponse, tags=["Monitoring"])
async def readiness_check() -> ReadinessResponse:
    """Return readiness diagnostics for runtime dependencies."""
    snapshot = ModelContainer.get_runtime_snapshot()
    predictor_loaded = bool(snapshot["predictor_loaded"])
    scorer_loaded = bool(snapshot["mismatch_scorer_loaded"])
    model_path_value = snapshot["model_path"]
    batch_size_value = snapshot["inference_batch_size"]

    if not isinstance(model_path_value, str):
        raise HTTPException(
            status_code=503,
            detail="Model path is not initialized.",
        )
    if not isinstance(batch_size_value, int):
        raise HTTPException(
            status_code=503,
            detail="Inference batch size is not initialized.",
        )

    return ReadinessResponse(
        status="ready" if predictor_loaded and scorer_loaded else "not_ready",
        predictor_loaded=predictor_loaded,
        mismatch_scorer_loaded=scorer_loaded,
        model_path=model_path_value,
        inference_batch_size=batch_size_value,
    )


@app.get("/model/info", tags=["Monitoring"])
async def get_model_info():
    """Return the base-model identifier and active checkpoint path."""
    return {
        "base_model": os.getenv("BASE_MODEL", ""),
        "checkpoint_path": ModelContainer.model_path,
    }


@app.get("/runtime/config", response_model=RuntimeConfigResponse)
async def get_runtime_config() -> RuntimeConfigResponse:
    """Return active runtime tuning parameters.

    Raises:
        HTTPException: If runtime components are not initialized.
    """
    runtime = ModelContainer.get_runtime_components()
    snapshot = ModelContainer.get_runtime_snapshot()

    anomaly_threshold = snapshot["anomaly_threshold"]
    under_weight = snapshot["anomaly_under_prediction_weight"]
    over_weight = snapshot["anomaly_over_prediction_weight"]

    if (
        anomaly_threshold is None
        or under_weight is None
        or over_weight is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Anomaly scorer configuration is not loaded.",
        )

    if not isinstance(anomaly_threshold, (int, float)):
        raise HTTPException(
            status_code=503,
            detail="Anomaly threshold has invalid type.",
        )
    if not isinstance(under_weight, (int, float)):
        raise HTTPException(
            status_code=503,
            detail="Under-prediction weight has invalid type.",
        )
    if not isinstance(over_weight, (int, float)):
        raise HTTPException(
            status_code=503,
            detail="Over-prediction weight has invalid type.",
        )

    return RuntimeConfigResponse(
        model_path=ModelContainer.model_path,
        inference_batch_size=runtime.batch_size,
        anomaly_threshold=float(anomaly_threshold),
        anomaly_under_prediction_weight=float(under_weight),
        anomaly_over_prediction_weight=float(over_weight),
    )


@app.get("/model/labels", tags=["Monitoring"])
async def get_model_labels():
    """Return configured label mappings from the loaded model.

    Raises:
        HTTPException: If the model is not initialized.
    """
    if ModelContainer.predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    model = ModelContainer.predictor.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    id2label = {
        int(k): str(v)
        for k, v in model.config.id2label.items()  # type: ignore[attr-defined]
    }
    label2id = {
        str(k): int(v)
        for k, v in model.config.label2id.items()  # type: ignore[attr-defined]
    }
    return {"id2label": id2label, "label2id": label2id}


@app.post("/anomaly/score", response_model=AnomalyScoreResponse)
async def score_anomaly(request: AnomalyScoreRequest) -> AnomalyScoreResponse:
    """Score one severity mismatch without running model inference.

    Raises:
        HTTPException: If runtime dependencies are unavailable.
    """
    runtime = ModelContainer.get_runtime_components()
    result = runtime.mismatch_scorer.score(
        observed_severity=request.observed_severity,
        predicted_severity=request.predicted_severity,
        confidence=request.confidence,
    )
    return AnomalyScoreResponse(
        observed_severity=result.observed_severity,
        predicted_severity=result.predicted_severity,
        mismatch_direction=result.mismatch_direction,
        mismatch_distance=result.severity_distance,
        confidence=result.confidence,
        anomaly_score=result.anomaly_score,
        anomaly_threshold=result.anomaly_threshold,
        is_anomaly=result.is_anomaly,
    )


def serve() -> None:
    """Start the API with uvicorn using HOST/PORT from the environment."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
