# tests/model_registry/test_model_registry.py
import unittest
from unittest.mock import patch, MagicMock
from src.model_registry.client import ModelRegistry
from src.model_registry.version import compare_model_versions, find_best_model_version

class TestModelRegistry(unittest.TestCase):
    
    @patch('mlflow.tracking.MlflowClient')
    @patch('mlflow.set_tracking_uri')
    def test_register_model(self, mock_set_uri, mock_client):
        # Arrange
        registry = ModelRegistry()
        
        # Act
        with patch('mlflow.start_run'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.register_model') as mock_register:
            
            model_uri = registry.register_model("model/path", "test_model")
            
            # Assert
            mock_register.assert_called_once()
            self.assertEqual(model_uri, "models:/test_model/latest")
    
    @patch('mlflow.tracking.MlflowClient')
    @patch('mlflow.set_tracking_uri')
    def test_get_latest_model(self, mock_set_uri, mock_client):
        # Arrange
        registry = ModelRegistry()
        mock_model = MagicMock()
        mock_client.return_value.get_latest_versions.return_value = [mock_model]
        
        # Act
        result = registry.get_latest_model("test_model")
        
        # Assert
        mock_client.return_value.get_latest_versions.assert_called_with("test_model", stages=["Production"])
        self.assertEqual(result, mock_model)
    
    @patch('mlflow.tracking.MlflowClient')
    @patch('mlflow.set_tracking_uri')
    def test_get_latest_model_not_found(self, mock_set_uri, mock_client):
        # Arrange
        registry = ModelRegistry()
        mock_client.return_value.get_latest_versions.return_value = []
        
        # Act
        result = registry.get_latest_model("test_model")
        
        # Assert
        self.assertIsNone(result)
