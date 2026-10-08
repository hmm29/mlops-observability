"""Pytest configuration.

On GitHub Actions, each failing test is also reported as an annotation, so the
failure is visible on the pull request without opening the full log.
"""
import os
import sys


def _annotate(title: str, details: str) -> None:
    if not os.getenv("GITHUB_ACTIONS"):
        return
    message = details[-3000:].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    sys.__stdout__.write(f"\n::error title={title}::{message}\n")
    sys.__stdout__.flush()


def pytest_runtest_logreport(report):
    if report.failed:
        _annotate(report.nodeid, str(report.longrepr))


def pytest_collectreport(report):
    if report.failed:
        _annotate(f"collection: {report.nodeid}", str(report.longrepr))


def pytest_runtest_teardown(item):
    """Never let one test leave an MLflow run active for the next."""
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:
        mlflow.end_run()
