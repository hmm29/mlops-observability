"""Model-serving API with built-in monitoring.

Run with: uvicorn src.api.main:app
"""
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.api.middleware import build_metrics_middleware
from src.data_validation.drift import DriftDetector, DriftMonitor
from src.data_validation.schema import DataSchemaValidator
from src.model.training import MODEL_NAME, ModelBundle, default_bundle_dir, load_bundle
from src.monitoring.metrics import MLMetricsCollector

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature values for one prediction")
    request_id: Optional[str] = Field(None, description="Caller's request ID, echoed back")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="1 if the transaction is flagged, else 0")
    prediction_probability: float = Field(..., description="Probability of the flagged class")
    request_id: Optional[str] = None
    model_version: str
    processing_time_ms: float


def resolve_bundle(model_name: str) -> Tuple[Optional[ModelBundle], str]:
    """Find and load the model bundle. Returns (bundle or None, version).

    If MLFLOW_TRACKING_URI is set, the Production version is downloaded from
    the MLflow registry. Otherwise the bundle is read from MODEL_DIR, which
    defaults to models/<model name>.
    """
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if tracking_uri:
            from src.model_registry.client import ModelRegistry

            path, version = ModelRegistry(tracking_uri).download_bundle(model_name, tempfile.mkdtemp())
            return load_bundle(path), version
        bundle = load_bundle(os.getenv("MODEL_DIR", default_bundle_dir(model_name)))
        return bundle, str(bundle.metadata.get("version", "1"))
    except Exception as error:  # noqa: BLE001 - start anyway and report through /health
        logger.error("Could not load model '%s': %s", model_name, error)
        return None, "unknown"


def create_app(
    bundle: Optional[ModelBundle] = None,
    version: Optional[str] = None,
    model_name: Optional[str] = None,
    drift_window: int = 200,
    drift_min_samples: int = 50,
    drift_check_every: int = 25,
) -> FastAPI:
    """Build the app. Pass a bundle directly (tests) or let it be resolved."""
    model_name = model_name or os.getenv("MODEL_NAME", MODEL_NAME)
    if bundle is None:
        bundle, resolved_version = resolve_bundle(model_name)
        version = version or resolved_version
    version = version or str(bundle.metadata.get("version", "1"))

    registry = CollectorRegistry()
    metrics = MLMetricsCollector(model_name, version, registry=registry)

    validator: Optional[DataSchemaValidator] = None
    monitor: Optional[DriftMonitor] = None
    feature_names: List[str] = []
    if bundle is not None:
        validator = DataSchemaValidator(schema=bundle.schema)
        feature_names = list(bundle.schema["features"])
        monitor = DriftMonitor(
            DriftDetector(bundle.reference_data[feature_names]),
            window_size=drift_window,
            min_samples=drift_min_samples,
            check_every=drift_check_every,
        )

    app = FastAPI(
        title="MLOps Observability API",
        description="Serves an example model and exposes prediction, drift and API metrics.",
        version="0.2.0",
    )
    app.middleware("http")(build_metrics_middleware(registry))
    app.state.metrics_registry = registry
    app.state.drift_monitor = monitor

    @metrics.track_latency()
    def run_prediction(features: Dict[str, Any]) -> float:
        frame = pd.DataFrame([features])
        validation = validator.validate(frame)
        if not validation["valid"]:
            metrics.track_error("validation_error")
            raise HTTPException(status_code=422, detail={"errors": validation["errors"]})
        try:
            probability = float(bundle.model.predict_proba(frame[feature_names])[0, 1])
        except Exception as error:  # noqa: BLE001
            metrics.track_error("prediction_error")
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from error

        row = {name: features.get(name) for name in feature_names}
        for name, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics.track_feature_value(name, float(value))
        report = monitor.add(row)
        if report is not None:
            metrics.track_drift_report(report)
        metrics.track_prediction("success")
        return probability

    @app.get("/health")
    def health() -> dict:
        if bundle is None:
            return {"status": "degraded", "message": "Model not loaded", "model": model_name}
        return {"status": "ok", "model": model_name, "version": version}

    @app.get("/model")
    def model_info() -> dict:
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {**bundle.metadata, "version": version, "features": bundle.schema["features"]}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        start = time.perf_counter()
        if bundle is None:
            metrics.track_error("model_not_loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")
        probability = run_prediction(request.features)
        return PredictionResponse(
            prediction=int(probability >= 0.5),
            prediction_probability=round(probability, 6),
            request_id=request.request_id,
            model_version=version,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 3),
        )

    @app.get("/drift")
    def drift() -> dict:
        if monitor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "rows_in_window": monitor.rows_seen,
            "min_samples": monitor.min_samples,
            "report": monitor.last_report,
        }

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
