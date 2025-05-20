# MLOps Observability Testing Framework

This directory contains the testing infrastructure for the MLOps Observability platform. The tests are organized to validate different aspects of the system, from individual components to end-to-end functionality.

## Test Structure

- **Unit Tests**: Test individual components in isolation
  - `tests/model_registry/`: Tests for the model registry functionality
  
- **Integration Tests**: Test the interaction between components
  - `tests/test_integration.py`: Validates how different parts of the system work together
  
- **End-to-End Tests**: Test the complete system workflow
  - `tests/test_e2e.py`: Validates the entire platform working together in real-world scenarios

## Running Tests

### Prerequisites

- Docker and Docker Compose (for integration and E2E tests)
- Python 3.8+ with pytest installed
- The MLOps Observability platform code

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/model_registry

# Run specific test file
pytest tests/model_registry/test_model_registry.py

# Run with more detailed output
pytest -v tests/model_registry
```

### Running Integration Tests

Integration tests validate that different components of the system work together correctly. These tests require the API server to be running:

```bash
# Start the API server (in a separate terminal)
python -m src.api.main

# Run the integration tests
pytest tests/test_integration.py
```

### Running End-to-End Tests

E2E tests validate the entire system workflow, including API server, Prometheus, and Grafana. These tests will automatically start and stop the required Docker containers:

```bash
# Run the E2E tests
pytest tests/test_e2e.py

# Note: These tests take longer to run since they need to start Docker containers
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=src tests/
```

For a more detailed HTML report:

```bash
pytest --cov=src --cov-report=html tests/
# This will generate a report in htmlcov/ directory
```

## Troubleshooting Tests

### Integration Test Issues

If integration tests fail:

1. Make sure the API server is running on port 8000
2. Check that the required modules are installed: `pip install -r requirements.txt`
3. Verify that the test environment has network access to the API server

### End-to-End Test Issues

If E2E tests fail:

1. Ensure Docker is running and Docker Compose is installed
2. Check for port conflicts (8000, 9090, 3000)
3. Increase the timeout values in the tests if services take longer to start up
4. Check Docker logs for any errors: `docker-compose -f docker/docker-compose-grafana.yml logs`

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Unit Tests**: Place them in a directory corresponding to the component being tested
2. **Integration Tests**: Add to `test_integration.py` or create a new file following the same pattern
3. **E2E Tests**: Add to `test_e2e.py` or create a new file with appropriate container setup/teardown

Each test should:
- Have a clear, descriptive name indicating what it tests
- Include docstrings explaining the test's purpose
- Clean up after itself, especially if it creates temporary resources
- Be deterministic (same input should produce same output every time)

## Continuous Integration

These tests are designed to be run in CI environments. In a CI setup, you would typically:

1. Run unit tests for every commit
2. Run integration tests for PRs and merges to main branches
3. Run E2E tests on a schedule or for releases

See the project's CI configuration for specific implementation details.
