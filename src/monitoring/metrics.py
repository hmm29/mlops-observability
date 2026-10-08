"""Prometheus metrics for one served model."""
import inspect
import time
from functools import wraps
from typing import Optional

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram


class MLMetricsCollector:
    """Counters, gauges and a latency histogram labeled by model and version.

    Exposed metric names (Prometheus adds `_total` to counters):
      model_predictions_total{model_name, version, result}
      model_prediction_errors_total{model_name, version, error_type}
      model_prediction_latency_seconds{model_name, version}
      model_feature_value{model_name, version, feature_name}
      model_drift_score{model_name, version, feature_name, drift_method}
      model_drift_detected{model_name, version, feature_name}
    """

    def __init__(self, model_name: str, version: str, registry: Optional[CollectorRegistry] = None):
        self.model_name = model_name
        self.version = version
        registry = registry or REGISTRY

        self.prediction_count = Counter(
            "model_predictions",
            "Number of predictions made",
            ["model_name", "version", "result"],
            registry=registry,
        )
        self.prediction_errors = Counter(
            "model_prediction_errors",
            "Prediction requests that failed",
            ["model_name", "version", "error_type"],
            registry=registry,
        )
        self.prediction_latency = Histogram(
            "model_prediction_latency_seconds",
            "Time taken to validate and predict",
            ["model_name", "version"],
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=registry,
        )
        self.feature_values = Gauge(
            "model_feature_value",
            "Most recent value of each numeric feature",
            ["model_name", "version", "feature_name"],
            registry=registry,
        )
        self.drift_score = Gauge(
            "model_drift_score",
            "Drift statistic per feature (KS statistic or Jensen-Shannon divergence)",
            ["model_name", "version", "feature_name", "drift_method"],
            registry=registry,
        )
        self.drift_detected = Gauge(
            "model_drift_detected",
            "1 if the latest drift check flagged the feature, else 0",
            ["model_name", "version", "feature_name"],
            registry=registry,
        )

    def _labels(self, **extra):
        return dict(model_name=self.model_name, version=self.version, **extra)

    def track_prediction(self, result: str = "success") -> None:
        self.prediction_count.labels(**self._labels(result=result)).inc()

    def track_error(self, error_type: str) -> None:
        self.prediction_errors.labels(**self._labels(error_type=error_type)).inc()

    def observe_latency(self, seconds: float) -> None:
        self.prediction_latency.labels(**self._labels()).observe(seconds)

    def track_latency(self):
        """Decorator that records how long a sync or async function takes."""

        def decorator(func):
            if inspect.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        self.observe_latency(time.perf_counter() - start)

                return async_wrapper

            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    self.observe_latency(time.perf_counter() - start)

            return wrapper

        return decorator

    def track_feature_value(self, feature_name: str, value: float) -> None:
        self.feature_values.labels(**self._labels(feature_name=feature_name)).set(value)

    def track_drift_score(self, feature_name: str, score: float, method: str = "ks") -> None:
        self.drift_score.labels(**self._labels(feature_name=feature_name, drift_method=method)).set(score)

    def track_drift_report(self, report: dict) -> None:
        """Record every feature in a DriftDetector report."""
        for feature, drift in report.get("feature_drifts", {}).items():
            self.track_drift_score(feature, drift["statistic"], drift["test"])
            self.drift_detected.labels(**self._labels(feature_name=feature)).set(1 if drift["drift"] else 0)
