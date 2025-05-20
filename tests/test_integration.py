import unittest
import requests
import json
import time
import threading
import subprocess
import os
from src.monitoring.metrics import MLMetricsCollector
from src.data_validation.schema import DataSchemaValidator
from src.data_validation.drift import DriftDetector
import pandas as pd

class TestMLOpsObservability(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Start services for integration testing"""
        # This would typically use docker-compose in a real setup
        pass
        
    @classmethod
    def tearDownClass(cls):
        """Stop services after testing"""
        pass
        
    def test_prediction_endpoint(self):
        """Test the prediction API endpoint"""
        url = "http://localhost:8000/predict"
        payload = {
            "features": {
                "feature1": 0.5,
                "feature2": 1.0,
                "feature3": "category_a"
            },
            "request_id": "test-123"
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("model_version", data)
        self.assertEqual(data["request_id"], "test-123")
        
    def test_invalid_input_rejected(self):
        """Test that invalid inputs are properly rejected"""
        url = "http://localhost:8000/predict"
        payload = {
            "features": {
                "invalid_feature": 0.5  # Feature not in schema
            }
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 400)
        
    def test_metrics_collection(self):
        """Test that metrics are being collected properly"""
        # Make a few prediction requests
        url = "http://localhost:8000/predict"
        for i in range(5):
            payload = {
                "features": {
                    "feature1": i * 0.1,
                    "feature2": 1.0,
                    "feature3": "category_a"
                }
            }
            requests.post(url, json=payload)
            
        # Check Prometheus metrics endpoint
        metrics_url = "http://localhost:8000/metrics"
        response = requests.get(metrics_url)
        self.assertEqual(response.status_code, 200)
        
        # Check for our custom metrics in the response
        metric_text = response.text
        self.assertIn("model_prediction_count", metric_text)
        self.assertIn("model_prediction_latency_seconds", metric_text)
        
    def test_drift_detection(self):
        """Test drift detection functionality"""
        # Create a reference dataset
        ref_data = pd.DataFrame({
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature2": [1.0, 1.1, 0.9, 1.0, 1.05]
        })
        
        # Create a current dataset with drift
        current_data = pd.DataFrame({
            "feature1": [0.6, 0.7, 0.8, 0.9, 1.0],  # Shifted distribution
            "feature2": [1.0, 1.1, 0.9, 1.0, 1.05]  # Same distribution
        })
        
        # Initialize drift detector
        detector = DriftDetector(ref_data)
        
        # Detect drift
        result = detector.detect_drift(current_data, threshold=0.05)
        
        # Verify drift was detected in feature1 but not feature2
        self.assertTrue(result["drift_detected"])
        self.assertIn("feature1", result["flagged_features"])
        self.assertNotIn("feature2", result["flagged_features"])
