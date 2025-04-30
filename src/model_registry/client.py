import os
import mlflow
from datetime import datetime

class ModelRegistry:
    """
    Client for interacting with the model registry
    Handles registration, versioning, and metadata tracking
    """
    def __init__(self, tracking_uri="http://localhost:5000"):
        """
        Initialize the model registry client
        
        Args:
            tracking_uri: URI for the MLflow tracking server
        """
        self.client = mlflow.tracking.MlflowClient(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        
    def register_model(self, model_path, name, tags=None):
        """
        Register a model to the MLflow registry with metadata
        
        Args:
            model_path: Path to the model artifacts
            name: Name for the model
            tags: Dictionary of tags to associate with the model
            
        Returns:
            model_uri: URI of the registered model
        """
        mlflow.set_experiment(name)
        
        with mlflow.start_run():
            # Log model metadata
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Log deployment time
            mlflow.log_param("deployment_time", datetime.now().isoformat())
            
            # Register the model
            model_uri = f"models:/{name}/latest"
            mlflow.register_model(model_path, name)
            
            return model_uri
            
    def get_latest_model(self, name):
        """
        Retrieve the latest model version
        
        Args:
            name: Name of the model
            
        Returns:
            Latest model version or None if not found
        """
        latest_version = self.client.get_latest_versions(name, stages=["Production"])
        if not latest_version:
            return None
        return latest_version[0]
    
    def get_model_versions(self, name):
        """
        Get all versions of a model
        
        Args:
            name: Name of the model
            
        Returns:
            List of all versions of the model
        """
        return self.client.search_model_versions(f"name='{name}'")
    
    def transition_model_stage(self, name, version, stage):
        """
        Transition a model to a different stage
        
        Args:
            name: Name of the model
            version: Version of the model
            stage: Target stage ("Staging", "Production", "Archived")
            
        Returns:
            Updated model version
        """
        return self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
