"""Check that the running Docker stack is wired together end to end.

Usage (after `docker compose up -d` and `scripts.simulate_traffic`):
    python scripts/check_stack.py

Uses only the standard library. Exits non-zero on the first failed check.
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

API = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
GRAFANA = "http://localhost:3000"
GRAFANA_AUTH = "Basic " + base64.b64encode(b"admin:admin").decode()


def annotate(title: str, details: str) -> None:
    """On GitHub Actions, show the message on the pull request."""
    if os.getenv("GITHUB_ACTIONS"):
        message = details[-3000:].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error title={title}::{message}", flush=True)


def get(url: str, auth: bool = False):
    request = urllib.request.Request(url, headers={"Authorization": GRAFANA_AUTH} if auth else {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def wait_for(description: str, check, timeout: int = 120):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            result = check()
            if result:
                print(f"ok    {description}")
                return result
        except Exception as error:  # noqa: BLE001 - keep retrying until the deadline
            last_error = error
        time.sleep(3)
    print(f"FAIL  {description} ({last_error})")
    annotate(f"Stack check failed: {description}", str(last_error))
    sys.exit(1)


def prometheus_value(query: str) -> float:
    data = get(f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": query}))
    results = data["data"]["result"]
    return float(results[0]["value"][1]) if results else 0.0


def main() -> None:
    if "--annotate-logs" in sys.argv:
        for service in ("api", "prometheus", "grafana"):
            logs = subprocess.run(
                ["docker", "compose", "logs", "--no-color", "--tail", "40", service],
                capture_output=True, text=True, check=False,
            )
            annotate(f"{service} logs", logs.stdout + logs.stderr)
        return
    wait_for("API is healthy with a model loaded", lambda: get(f"{API}/health")["status"] == "ok")
    wait_for(
        "API drift check flags the drifted feature",
        lambda: "amount" in (get(f"{API}/drift")["report"] or {}).get("flagged_features", []),
    )
    wait_for(
        "Prometheus is scraping the API",
        lambda: any(
            target["labels"].get("job") == "mlops-api" and target["health"] == "up"
            for target in get(f"{PROMETHEUS}/api/v1/targets")["data"]["activeTargets"]
        ),
    )
    wait_for("Prometheus has prediction counts", lambda: prometheus_value("sum(model_predictions_total)") > 0)
    wait_for(
        "Prometheus has a drift score above the alert threshold",
        lambda: prometheus_value('max(model_drift_score{drift_method="ks"})') > 0.2,
    )
    wait_for("Grafana is up", lambda: get(f"{GRAFANA}/api/health")["database"] == "ok")
    wait_for(
        "Grafana has the Prometheus data source",
        lambda: get(f"{GRAFANA}/api/datasources/uid/prometheus", auth=True)["type"] == "prometheus",
    )
    wait_for(
        "Grafana has the Model Monitoring dashboard",
        lambda: get(f"{GRAFANA}/api/dashboards/uid/model-monitoring", auth=True)["dashboard"]["title"] == "Model Monitoring",
    )
    wait_for(
        "Grafana has the three alert rules",
        lambda: {rule["uid"] for rule in get(f"{GRAFANA}/api/v1/provisioning/alert-rules", auth=True)}
        == {"mlops_feature_drift", "mlops_error_rate", "mlops_latency"},
    )
    print("All checks passed.")


if __name__ == "__main__":
    main()
