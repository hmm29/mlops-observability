import asyncio

import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry  # noqa: E402

from src.monitoring.metrics import MLMetricsCollector  # noqa: E402

LABELS = {"model_name": "m", "version": "1"}


@pytest.fixture()
def setup():
    registry = CollectorRegistry()
    return registry, MLMetricsCollector("m", "1", registry=registry)


def test_prediction_and_error_counters(setup):
    registry, metrics = setup
    metrics.track_prediction()
    metrics.track_prediction()
    metrics.track_error("validation_error")
    assert registry.get_sample_value("model_predictions_total", {**LABELS, "result": "success"}) == 2
    assert registry.get_sample_value("model_prediction_errors_total", {**LABELS, "error_type": "validation_error"}) == 1


def test_latency_decorator_on_sync_function(setup):
    registry, metrics = setup

    @metrics.track_latency()
    def work(value):
        return value * 2

    assert work(21) == 42
    assert registry.get_sample_value("model_prediction_latency_seconds_count", LABELS) == 1


def test_latency_decorator_on_async_function_times_the_await(setup):
    registry, metrics = setup

    @metrics.track_latency()
    async def work():
        await asyncio.sleep(0.05)
        return "done"

    assert asyncio.run(work()) == "done"
    assert registry.get_sample_value("model_prediction_latency_seconds_count", LABELS) == 1
    assert registry.get_sample_value("model_prediction_latency_seconds_sum", LABELS) >= 0.04


def test_latency_is_recorded_when_the_function_raises(setup):
    registry, metrics = setup

    @metrics.track_latency()
    def broken():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        broken()
    assert registry.get_sample_value("model_prediction_latency_seconds_count", LABELS) == 1


def test_drift_report_sets_score_and_flag(setup):
    registry, metrics = setup
    metrics.track_drift_report(
        {
            "feature_drifts": {
                "amount": {"test": "ks", "statistic": 0.42, "p_value": 0.001, "drift": True},
                "category": {"test": "jensen_shannon", "statistic": 0.03, "drift": False},
            }
        }
    )
    assert registry.get_sample_value("model_drift_score", {**LABELS, "feature_name": "amount", "drift_method": "ks"}) == 0.42
    assert registry.get_sample_value("model_drift_detected", {**LABELS, "feature_name": "amount"}) == 1
    assert registry.get_sample_value("model_drift_detected", {**LABELS, "feature_name": "category"}) == 0


def test_two_collectors_can_coexist_with_separate_registries():
    MLMetricsCollector("a", "1", registry=CollectorRegistry())
    MLMetricsCollector("b", "1", registry=CollectorRegistry())
