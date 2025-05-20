# MLOps Observability Platform - Setup Guide

This document provides detailed instructions for setting up and using the MLOps Observability Platform.

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Git

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hmm29/mlops-observability.git
cd mlops-observability
```

### 2. Set Up the Environment

#### Option 1: Using Docker Compose (Recommended)

This option starts all components, including the API server, Prometheus, and Grafana.

```bash
# Start the complete stack
docker-compose -f docker/docker-compose-grafana.yml up -d

# To stop the services
docker-compose -f docker/docker-compose-grafana.yml down
```

#### Option 2: Manual Setup

For development, you might want to run components separately:

```bash
# Set up a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m src.api.main

# In separate terminals, start Prometheus and Grafana
cd docker
docker-compose -f docker-compose-grafana.yml up prometheus grafana
```

## Accessing the Components

After starting the services, you can access them at:

- **API Server**: http://localhost:8000
  - API Documentation: http://localhost:8000/docs
  - Metrics Endpoint: http://localhost:8000/metrics
  
- **Prometheus**: http://localhost:9090
  - Query Interface: http://localhost:9090/graph
  
- **Grafana**: http://localhost:3000
  - Default login: admin/admin
  - Model Monitoring Dashboard: http://localhost:3000/d/model-monitoring

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/model_registry

# Run specific test file
pytest tests/model_registry/test_model_registry.py
```

### Integration Tests

```bash
# Make sure the API server is running
pytest tests/test_integration.py
```

### End-to-End Tests

```bash
# This will automatically start needed containers
pytest tests/test_e2e.py
```

## Monitoring Your Models

### 1. Instrumenting Your Code

Use our metrics collector to instrument your model code:

```python
from src.monitoring.metrics import MLMetricsCollector

# Initialize collector
metrics = MLMetricsCollector(
    model_name="your_model",
    model_version="1.0.0"
)

# Track prediction counts
with metrics.track_predictions():
    result = model.predict(features)

# Track prediction latency
with metrics.track_latency():
    result = model.predict(features)
```

### 2. Setting Up Drift Detection

```python
from src.data_validation.drift import DriftDetector
import pandas as pd

# Load reference dataset
reference_data = pd.read_csv("reference_data.csv")

# Initialize detector
detector = DriftDetector(reference_data)

# Check current data for drift
current_data = fetch_current_data()
drift_results = detector.detect_drift(current_data, threshold=0.05)

if drift_results["drift_detected"]:
    print(f"Drift detected in features: {drift_results['flagged_features']}")
```

### 3. Viewing Metrics and Dashboards

1. Open Grafana at http://localhost:3000
2. Navigate to the "Model Monitoring Dashboard"
3. Select your model and version from the dropdown menus

### 4. Setting Up Alerts

Alerts are configured in three places:

1. **Grafana Dashboards**: Dashboard-based alerts in the "Alert" tab
2. **Alert Rules**: YAML configuration in `grafana/alerts/drift_alert.yaml`
3. **Notification Channels**: Email and other notifications in `docker/grafana/provisioning/notifiers/`

To configure a new alert:

1. Edit or create a new YAML file in the `grafana/alerts/` directory
2. Restart Grafana to apply the changes:
   ```bash
   docker-compose -f docker/docker-compose-grafana.yml restart grafana
   ```

## Troubleshooting

### API Server Issues

If the API server doesn't start:

1. Check logs: `docker logs mlops-api`
2. Verify port availability: `netstat -an | grep 8000`
3. Ensure dependencies are installed: `pip install -r requirements.txt`

### Prometheus Issues

If metrics don't appear in Prometheus:

1. Check if the API endpoint is accessible: `curl http://localhost:8000/metrics`
2. Verify Prometheus configuration: `docker/prometheus/prometheus.yml`
3. Check Prometheus targets: http://localhost:9090/targets

### Grafana Issues

If dashboards don't display correctly:

1. Verify Prometheus data source is configured correctly
2. Check dashboard JSON for errors
3. Review Grafana logs: `docker logs mlops-grafana`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
