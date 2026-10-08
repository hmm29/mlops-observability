"""Detect drift between reference (training) data and recent production data."""
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def jensen_shannon_divergence(first: Dict[Any, float], second: Dict[Any, float]) -> float:
    """Jensen-Shannon divergence (base 2) between two categorical distributions.

    Inputs map category to probability. The result is 0 for identical
    distributions and 1 for distributions with no category in common.
    """
    categories = sorted(set(first) | set(second), key=str)
    p = np.array([first.get(category, 0.0) for category in categories], dtype=float)
    q = np.array([second.get(category, 0.0) for category in categories], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    mixture = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return max(0.0, min(1.0, 0.5 * kl(p, mixture) + 0.5 * kl(q, mixture)))


class DriftDetector:
    """Compare a batch of current data with the reference data, per feature.

    Numeric features use the two-sample Kolmogorov-Smirnov test and are
    flagged when the p-value is below `threshold`. Categorical features use
    Jensen-Shannon divergence and are flagged when it exceeds `js_threshold`.
    """

    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.numeric_columns = list(reference_data.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = [
            column for column in reference_data.columns if column not in self.numeric_columns
        ]
        self.reference_distributions = {
            column: reference_data[column].value_counts(normalize=True).to_dict()
            for column in self.categorical_columns
        }

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05,
        js_threshold: float = 0.1,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"drift_detected": False, "feature_drifts": {}, "flagged_features": []}

        for column in self.numeric_columns:
            if column not in current_data.columns:
                continue
            current = pd.to_numeric(current_data[column], errors="coerce").dropna()
            if current.empty:
                continue
            statistic, p_value = stats.ks_2samp(self.reference_data[column].dropna(), current)
            drifted = bool(p_value < threshold)
            results["feature_drifts"][column] = {
                "test": "ks",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift": drifted,
            }
            if drifted:
                results["flagged_features"].append(column)

        for column in self.categorical_columns:
            if column not in current_data.columns:
                continue
            current = current_data[column].dropna()
            if current.empty:
                continue
            divergence = jensen_shannon_divergence(
                self.reference_distributions[column],
                current.value_counts(normalize=True).to_dict(),
            )
            drifted = divergence > js_threshold
            results["feature_drifts"][column] = {
                "test": "jensen_shannon",
                "statistic": divergence,
                "drift": drifted,
            }
            if drifted:
                results["flagged_features"].append(column)

        results["drift_detected"] = bool(results["flagged_features"])
        return results


class DriftMonitor:
    """Run drift detection over a sliding window of recent requests.

    A single request says nothing about a distribution, so rows are collected
    in a window and the detector runs every `check_every` rows once the window
    holds at least `min_samples`.
    """

    def __init__(
        self,
        detector: DriftDetector,
        window_size: int = 200,
        min_samples: int = 50,
        check_every: int = 25,
    ):
        self.detector = detector
        self.min_samples = min_samples
        self.check_every = check_every
        self._window: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self._since_check = 0
        self._lock = threading.Lock()
        self.last_report: Optional[Dict[str, Any]] = None

    def add(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add one row. Returns a new drift report when a check ran, else None."""
        with self._lock:
            self._window.append(row)
            self._since_check += 1
            if len(self._window) < self.min_samples or self._since_check < self.check_every:
                return None
            self._since_check = 0
            window = pd.DataFrame(list(self._window))
        report = self.detector.detect_drift(window)
        report["window_size"] = len(window)
        self.last_report = report
        return report

    @property
    def rows_seen(self) -> int:
        return len(self._window)
