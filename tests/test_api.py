"""HTTP tests against the real app and the real trained example model."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import create_app  # noqa: E402
from src.model.training import FEATURES, generate_data  # noqa: E402
from tests.helpers import VALID_FEATURES, trained_bundle  # noqa: E402

LABELS = {"model_name": "example_model", "version": "1"}


@pytest.fixture()
def app():
    return create_app(bundle=trained_bundle(), drift_window=100, drift_min_samples=30, drift_check_every=10)


@pytest.fixture()
def client(app):
    return TestClient(app)


def sample(app, name, labels=None):
    return app.state.metrics_registry.get_sample_value(name, {**LABELS, **(labels or {})})


def records(rows, seed, drifted=False):
    return json.loads(generate_data(rows=rows, seed=seed, drifted=drifted)[FEATURES].to_json(orient="records"))


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model": "example_model", "version": "1"}
    info = client.get("/model").json()
    assert info["data"] == "synthetic"
    assert set(info["features"]) == set(FEATURES)


def test_predict_returns_a_real_model_score(client, app):
    low = client.post("/predict", json={"features": VALID_FEATURES, "request_id": "abc"})
    risky = {"amount": 4000.0, "account_age_days": 5, "transactions_last_24h": 12, "merchant_category": "electronics"}
    high = client.post("/predict", json={"features": risky})

    assert low.status_code == 200 and high.status_code == 200
    body = low.json()
    assert body["request_id"] == "abc"
    assert body["model_version"] == "1"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["prediction_probability"] <= 1.0
    assert high.json()["prediction_probability"] > body["prediction_probability"]
    assert sample(app, "model_predictions_total", {"result": "success"}) == 2
    assert sample(app, "model_prediction_latency_seconds_count") == 2


def test_extra_features_are_ignored(client):
    response = client.post("/predict", json={"features": {**VALID_FEATURES, "unused": "x"}})
    assert response.status_code == 200


def test_invalid_input_is_rejected_with_422(client, app):
    response = client.post("/predict", json={"features": {**VALID_FEATURES, "amount": "lots"}})
    assert response.status_code == 422
    assert "Column amount should be numeric" in response.json()["detail"]["errors"]
    assert client.post("/predict", json={"features": {"amount": 5}}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert sample(app, "model_prediction_errors_total", {"error_type": "validation_error"}) == 2
    assert sample(app, "model_predictions_total", {"result": "success"}) is None


def test_metrics_endpoint_exposes_model_and_api_metrics(client):
    client.post("/predict", json={"features": VALID_FEATURES})
    client.get("/does-not-exist")
    text = client.get("/metrics").text
    assert 'model_predictions_total{model_name="example_model",result="success",version="1"} 1.0' in text
    assert 'api_requests_total{endpoint="/predict",method="POST",status_code="200"} 1.0' in text
    assert 'api_requests_total{endpoint="unmatched",method="GET",status_code="404"} 1.0' in text
    assert "model_feature_value" in text


def test_drift_report_appears_after_enough_requests(client, app):
    assert client.get("/drift").json()["report"] is None
    for record in records(30, seed=11):
        assert client.post("/predict", json={"features": record}).status_code == 200

    body = client.get("/drift").json()
    assert body["rows_in_window"] == 30
    assert body["report"]["window_size"] == 30
    assert sample(app, "model_drift_score", {"feature_name": "amount", "drift_method": "ks"}) is not None


def test_drifted_traffic_is_detected_and_exported(client, app):
    for record in records(100, seed=12, drifted=True):
        client.post("/predict", json={"features": record})

    report = client.get("/drift").json()["report"]
    assert "amount" in report["flagged_features"]
    assert sample(app, "model_drift_score", {"feature_name": "amount", "drift_method": "ks"}) > 0.2
    assert sample(app, "model_drift_detected", {"feature_name": "amount"}) == 1
    assert sample(app, "model_drift_score", {"feature_name": "merchant_category", "drift_method": "jensen_shannon"}) > 0


def test_app_without_a_model_reports_degraded(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))
    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/predict", json={"features": VALID_FEATURES}).status_code == 503
    assert client.get("/drift").status_code == 503
    assert client.get("/metrics").status_code == 200
