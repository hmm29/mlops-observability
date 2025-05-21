# src/monitoring/metrics.py
from prometheus_client import Counter, Gauge, Histogram, Summary
import time
from functools import wraps
from typing import Dict, List, Any, Callable

class MLMetricsCollector:
    def __init__(self, model_name: str, version: str):
        """Initialize metrics collectors for a specific model version"""
        self.model_name = model_name
        self.version = version
        
        # Define metrics
        self.prediction_count = Counter(
            'model_prediction_count', 
            'Number of predictions made',
            ['model_name', 'version', 'result']
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Time taken for prediction',
            ['model_name', 'version'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
        )
        
        self.feature_values = Gauge(
            'model_feature_value',
            'Feature values seen by the model',
            ['model_name', 'version', 'feature_name']
        )
        
        self.drift_score = Gauge(
            'model_drift_score',
            'Drift score for each feature',
            ['model_name', 'version', 'feature_name', 'drift_method']
        )
        
        self.prediction_errors = Counter(
            'model_prediction_errors',
            'Prediction errors',
            ['model_name', 'version', 'error_type']
        )

    def track_prediction(self, result: str = "success"):
        """Track a prediction count"""
        self.prediction_count.labels(
            model_name=self.model_name, 
            version=self.version,
            result=result
        ).inc()
    
    def track_latency(self):
        """Decorator to track prediction latency"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                
                self.prediction_latency.labels(
                    model_name=self.model_name,
                    version=self.version
                ).observe(latency)
                
                return result
            return wrapper
        return decorator
    
    def track_feature_value(self, feature_name: str, value: float):
        """Track a feature value for monitoring"""
        self.feature_values.labels(
            model_name=self.model_name,
            version=self.version,
            feature_name=feature_name
        ).set(value)
    
    def track_drift_score(self, feature_name: str, score: float, method: str = "ks_test"):
        """Track drift score for a feature"""
        self.drift_score.labels(
            model_name=self.model_name,
            version=self.version,
            feature_name=feature_name,
            drift_method=method
        ).set(score)
    
    def track_error(self, error_type: str):
        """Track a prediction error"""
        self.prediction_errors.labels(
            model_name=self.model_name,
            version=self.version,
            error_type=error_type
        ).inc()
