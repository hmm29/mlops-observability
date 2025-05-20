import unittest
import requests
import json
import time
import subprocess
import os
import pandas as pd
import docker
from contextlib import contextmanager

class TestMLOpsE2E(unittest.TestCase):
    """End-to-end tests for the MLOps Observability platform.
    
    This test suite validates the full system functionality, testing the interactions
    between all components in a real-world usage scenario.
    """
    
    @classmethod
    def setUpClass(cls):
        """Start the entire system using docker-compose"""
        cls.docker_client = docker.from_env()
        
        # Store the current directory
        cls.cwd = os.getcwd()
        
        # Change to the project root directory
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        
        # Start containers using subprocess to call docker-compose
        cls.compose_process = subprocess.Popen(
            ["docker-compose", "-f", "docker/docker-compose-grafana.yml", "up", "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for containers to be ready (simplified approach)
        time.sleep(30)
        
    @classmethod
    def tearDownClass(cls):
        """Stop all containers after tests"""
        try:
            subprocess.run(
                ["docker-compose", "-f", "docker/docker-compose-grafana.yml", "down"],
                check=True
            )
        finally:
            # Return to the original directory
            os.chdir(cls.cwd)
    
    def test_full_prediction_workflow(self):
        """Test the entire prediction workflow from request to metrics"""
        # 1. Make a prediction request
        url = "http://localhost:8000/predict"
        prediction_payload = {
            "features": {
                "feature1": 0.5,
                "feature2": 1.0,
                "feature3": "category_a"
            },
            "request_id": "e2e-test-123"
        }
        
        response = requests.post(url, json=prediction_payload)
        self.assertEqual(response.status_code, 200)
        
        # 2. Check the prediction result
        prediction_data = response.json()
        self.assertIn("prediction", prediction_data)
        self.assertIn("model_version", prediction_data)
        
        # 3. Wait for metrics to be collected
        time.sleep(5)
        
        # 4. Check Prometheus metrics
        metrics_url = "http://localhost:8000/metrics"
        metrics_response = requests.get(metrics_url)
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text
        
        # Verify our custom metrics exist
        self.assertIn("model_prediction_count", metrics_text)
        self.assertIn("model_prediction_latency_seconds", metrics_text)
        
        # 5. Check Prometheus API to verify metrics are being scraped
        prom_url = "http://localhost:9090/api/v1/query"
        query_params = {
            "query": 'model_prediction_count{request_id="e2e-test-123"}'
        }
        
        # Allow time for Prometheus to scrape the metrics (typically 15s)
        max_retries = 10
        for attempt in range(max_retries):
            prom_response = requests.get(prom_url, params=query_params)
            prom_data = prom_response.json()
            
            if prom_data["status"] == "success" and len(prom_data["data"]["result"]) > 0:
                break
                
            time.sleep(5)
        
        # 6. Verify metrics were properly collected in Prometheus
        self.assertEqual(prom_data["status"], "success")
        self.assertTrue(len(prom_data["data"]["result"]) > 0)
        
    def test_data_drift_workflow(self):
        """Test the data drift detection and alerting workflow"""
        # 1. Create initial data distribution (reference data)
        url = "http://localhost:8000/predict"
        
        # Make 20 normal requests
        for i in range(20):
            normal_payload = {
                "features": {
                    "feature1": i * 0.05,  # Values from 0 to 0.95
                    "feature2": 1.0,
                    "feature3": "category_a"
                }
            }
            requests.post(url, json=normal_payload)
        
        # 2. Wait for metrics to be collected
        time.sleep(5)
        
        # 3. Now send data with drift
        for i in range(20):
            drift_payload = {
                "features": {
                    "feature1": i * 0.05 + 2.0,  # Shifted distribution
                    "feature2": 1.0,
                    "feature3": "category_b"  # Category drift
                }
            }
            requests.post(url, json=drift_payload)
        
        # 4. Check drift metrics in Prometheus
        time.sleep(10)  # Allow time for drift calculation
        prom_url = "http://localhost:9090/api/v1/query"
        query_params = {
            "query": 'model_drift_score{drift_method="ks_test"}'
        }
        
        max_retries = 6
        drift_detected = False
        
        for attempt in range(max_retries):
            prom_response = requests.get(prom_url, params=query_params)
            prom_data = prom_response.json()
            
            if (prom_data["status"] == "success" and 
                len(prom_data["data"]["result"]) > 0):
                
                # Check if any drift scores exceed threshold
                for result in prom_data["data"]["result"]:
                    if float(result["value"][1]) > 0.05:
                        drift_detected = True
                        break
                
                if drift_detected:
                    break
                    
            time.sleep(10)
        
        # 5. Verify drift was detected
        self.assertTrue(drift_detected, "Data drift should have been detected")
        
    def test_grafana_dashboard_accessibility(self):
        """Test that Grafana is up and the dashboards are accessible"""
        # 1. Check Grafana is running
        grafana_url = "http://localhost:3000/api/health"
        response = requests.get(grafana_url)
        self.assertEqual(response.status_code, 200)
        
        # 2. Try to log in with default credentials
        login_url = "http://localhost:3000/login"
        session = requests.Session()
        response = session.get(login_url)
        self.assertEqual(response.status_code, 200)
        
        # 3. Check dashboard exists
        # We need to authenticate first
        auth_response = session.post(
            "http://localhost:3000/login", 
            data={"user": "admin", "password": "admin"}
        )
        
        # Now check for our dashboard
        dashboard_url = "http://localhost:3000/api/search?query=Model%20Monitoring"
        dashboard_response = session.get(dashboard_url)
        self.assertEqual(dashboard_response.status_code, 200)
        
        dashboard_data = dashboard_response.json()
        self.assertTrue(len(dashboard_data) > 0, "Dashboard should exist")
