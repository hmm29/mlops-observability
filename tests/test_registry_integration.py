"""Runs the registry client against a real MLflow registry in a local SQLite
file. No MLflow server is needed."""
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import create_app  # noqa: E402
from src.model.training import MODEL_NAME, load_bundle, train_bundle  # noqa: E402
from src.model_registry.client import ModelRegistry  # noqa: E402
from src.model_registry.version import compare_model_versions, find_best_model_version  # noqa: E402
from tests.helpers import VALID_FEATURES  # noqa: E402


@pytest.fixture()
def tracking_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts to ./mlruns
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_register_download_compare_and_serve(tracking_uri, tmp_path, monkeypatch):
    registry = ModelRegistry(tracking_uri)
    assert registry.get_model_versions(MODEL_NAME) == []

    first = train_bundle(str(tmp_path / "v1"), rows=400, seed=1)
    second = train_bundle(str(tmp_path / "v2"), rows=1200, seed=2)
    assert registry.log_bundle(str(tmp_path / "v1"), MODEL_NAME, metrics=first["metrics"]) == "1"
    assert registry.log_bundle(str(tmp_path / "v2"), MODEL_NAME, metrics=second["metrics"]) == "2"

    assert len(registry.get_model_versions(MODEL_NAME)) == 2
    assert registry.get_latest_model(MODEL_NAME).version in ("2", 2)

    comparison = compare_model_versions(registry, MODEL_NAME, "1", "2", metric="roc_auc")
    assert abs(comparison["difference"] - (first["metrics"]["roc_auc"] - second["metrics"]["roc_auc"])) < 1e-9
    best = find_best_model_version(registry, MODEL_NAME, metric="roc_auc")
    assert best["value"] == max(first["metrics"]["roc_auc"], second["metrics"]["roc_auc"])

    path, version = registry.download_bundle(MODEL_NAME, str(tmp_path / "download"))
    assert version == "2"
    assert load_bundle(path).metadata["training_rows"] == second["training_rows"]

    # The API serves the Production version when MLFLOW_TRACKING_URI is set.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok", "model": MODEL_NAME, "version": "2"}
    response = client.post("/predict", json={"features": VALID_FEATURES})
    assert response.status_code == 200
    assert response.json()["model_version"] == "2"


def test_download_without_a_production_version_fails_clearly(tracking_uri, tmp_path):
    with pytest.raises(Exception):
        ModelRegistry(tracking_uri).download_bundle("nothing_registered", str(tmp_path / "x"))
