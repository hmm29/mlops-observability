"""A thin client over the MLflow model registry."""
from datetime import datetime
from typing import Dict, Optional, Tuple

import mlflow


class ModelRegistry:
    """Register model bundles, look up versions and move them between stages.

    `tracking_uri` can point at an MLflow server (`http://localhost:5000`) or
    at a local database (`sqlite:///mlflow.db`), which needs no server.

    Uses MLflow's stage-based registry API, which MLflow 2.9+ marks as
    deprecated in favor of aliases. It works on MLflow 2.x; see requirements.
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.client = mlflow.tracking.MlflowClient(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def register_model(self, model_path: str, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Register an existing artifact URI as a new version of `name`."""
        mlflow.set_experiment(name)
        with mlflow.start_run():
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, value)
            mlflow.log_param("deployment_time", datetime.now().isoformat())
            mlflow.register_model(model_path, name)
            return f"models:/{name}/latest"

    def log_bundle(
        self,
        bundle_dir: str,
        name: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: Optional[str] = "Production",
    ) -> str:
        """Upload a model bundle folder, register it, and return the new version.

        The bundle's metrics are logged on the run, so versions can be compared
        with `compare_model_versions`. If `stage` is set the new version is
        moved to it.
        """
        mlflow.set_experiment(name)
        with mlflow.start_run() as run:
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, value)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.log_artifacts(bundle_dir, artifact_path="bundle")
            registered = mlflow.register_model(f"runs:/{run.info.run_id}/bundle", name)
        version = str(registered.version)
        if stage:
            self.transition_model_stage(name, version, stage)
        return version

    def download_bundle(self, name: str, destination: str) -> Tuple[str, str]:
        """Download the Production version's bundle. Returns (local path, version)."""
        latest = self.get_latest_model(name)
        if latest is None:
            raise LookupError(f"No Production version of '{name}' in the registry")
        path = mlflow.artifacts.download_artifacts(artifact_uri=latest.source, dst_path=destination)
        return path, str(latest.version)

    def get_latest_model(self, name: str):
        """The latest version in the Production stage, or None."""
        latest_version = self.client.get_latest_versions(name, stages=["Production"])
        if not latest_version:
            return None
        return latest_version[0]

    def get_model_versions(self, name: str):
        """All versions of a model."""
        return self.client.search_model_versions(f"name='{name}'")

    def transition_model_stage(self, name: str, version: str, stage: str):
        """Move a version to "Staging", "Production" or "Archived"."""
        return self.client.transition_model_version_stage(name=name, version=version, stage=stage)
