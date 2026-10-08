import pandas as pd

from src.data_validation.drift import DriftDetector, DriftMonitor, jensen_shannon_divergence
from src.model.training import FEATURES, generate_data


def reference():
    return generate_data(rows=1500, seed=42)[FEATURES]


def test_js_divergence_bounds():
    same = {"a": 0.5, "b": 0.5}
    assert jensen_shannon_divergence(same, same) == 0.0
    assert abs(jensen_shannon_divergence({"a": 1.0}, {"b": 1.0}) - 1.0) < 1e-9
    partial = jensen_shannon_divergence({"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9})
    assert 0.0 < partial < 1.0


def test_js_divergence_is_symmetric():
    first, second = {"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.5, "c": 0.3}
    assert abs(jensen_shannon_divergence(first, second) - jensen_shannon_divergence(second, first)) < 1e-12


def test_drifted_data_is_flagged():
    detector = DriftDetector(reference())
    report = detector.detect_drift(generate_data(rows=300, seed=7, drifted=True)[FEATURES])
    assert report["drift_detected"] is True
    assert "amount" in report["flagged_features"]
    assert "merchant_category" in report["flagged_features"]
    assert report["feature_drifts"]["amount"]["test"] == "ks"
    assert report["feature_drifts"]["amount"]["statistic"] > 0.3
    assert report["feature_drifts"]["merchant_category"]["test"] == "jensen_shannon"


def test_matching_data_has_small_statistics():
    detector = DriftDetector(reference())
    report = detector.detect_drift(generate_data(rows=300, seed=7)[FEATURES])
    # p-values can dip under 0.05 by chance, so assert on effect size instead.
    for name in ["amount", "account_age_days", "transactions_last_24h"]:
        assert report["feature_drifts"][name]["statistic"] < 0.2
    assert report["feature_drifts"]["merchant_category"]["drift"] is False


def test_identical_data_is_not_flagged():
    data = reference()
    report = DriftDetector(data).detect_drift(data)
    assert report["drift_detected"] is False
    assert report["flagged_features"] == []


def test_columns_missing_from_current_data_are_skipped():
    detector = DriftDetector(reference())
    report = detector.detect_drift(pd.DataFrame({"amount": [10.0, 20.0, 30.0]}))
    assert list(report["feature_drifts"]) == ["amount"]


def test_monitor_waits_for_enough_rows_then_checks_periodically():
    monitor = DriftMonitor(DriftDetector(reference()), window_size=100, min_samples=20, check_every=10)
    rows = generate_data(rows=40, seed=3)[FEATURES].to_dict(orient="records")

    reports = [monitor.add(row) for row in rows]

    assert all(report is None for report in reports[:19])
    assert reports[19] is not None            # first check at min_samples
    assert reports[20] is None                # then every 10 rows
    assert reports[29] is not None
    assert reports[29]["window_size"] == 30
    assert monitor.last_report is reports[39]


def test_monitor_window_slides():
    monitor = DriftMonitor(DriftDetector(reference()), window_size=50, min_samples=50, check_every=50)
    normal = generate_data(rows=50, seed=3)[FEATURES].to_dict(orient="records")
    drifted = generate_data(rows=50, seed=4, drifted=True)[FEATURES].to_dict(orient="records")
    for row in normal:
        monitor.add(row)
    assert monitor.last_report["feature_drifts"]["amount"]["statistic"] < 0.25
    for row in drifted:
        monitor.add(row)
    assert monitor.rows_seen == 50
    assert "amount" in monitor.last_report["flagged_features"]
