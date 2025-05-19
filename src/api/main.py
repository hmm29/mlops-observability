# src/api/main.py
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import json
import os
import time

from src.monitoring.metrics import MLMetricsCollector
from src.data_validation.schema import DataSchemaValidator
from src.data_validation.drift import DriftDetector
from src.model_registry.client import ModelRegistry
from src.api.middleware import metrics_middleware

# Load model from registry
MODEL_NAME = os.getenv("MODEL_NAME", "example_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# Initialize the app
app = FastAPI(
    title="MLOps Observability API",
    description="API for model serving with built-in monitoring",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.middleware("http")(metrics_middleware)

# Initialize components
metrics = MLMetricsCollector(MODEL_NAME, MODEL_VERSION)
registry = ModelRegistry()
model = None  # Will be loaded on startup

# Pydantic models for requests/responses
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    request_id: Optional[str] = Field(None, description="Unique request ID")

class PredictionResponse(BaseModel):
    prediction: Any = Field(..., description="Model prediction")
    prediction_probability: Optional[float] = Field(None, description="Prediction probability")
    request_id: Optional[str] = Field(None, description="Original request ID")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, validator, drift_detector
    
    try:
        # Load the model from registry
        model_info = registry.get_latest_model(MODEL_NAME)
        if not model_info:
            raise ValueError(f"Model {MODEL_NAME} not found in registry")
        
        # TODO: Load actual model here
        model = "placeholder"  # Will be replaced with actual model loading
        
        # Load schema and reference data
        schema_path = f"models/{MODEL_NAME}/schema.json"
        reference_data_path = f"models/{MODEL_NAME}/reference_data.csv"
        
        validator = DataSchemaValidator(schema_path=schema_path)
        drift_detector = DriftDetector(pd.read_csv(reference_data_path))
        
    except Exception as e:
        # Log the error but allow the app to start
        print(f"Error loading model: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        return {"status": "warning", "message": "Model not loaded"}
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
@metrics.track_latency()
async def predict(request: PredictionRequest):
    """Make a prediction with the model"""
    start_time = time.time()
    
    if model is None:
        metrics.track_error("model_not_loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame for validation
        features_df = pd.DataFrame([request.features])
        
        # Validate input data
        validation_result = validator.validate(features_df)
        if not validation_result["valid"]:
            metrics.track_error("validation_error")
            raise HTTPException(
                status_code=400, 
                detail=f"Validation error: {validation_result['errors']}"
            )
        
        # Track feature values for monitoring
        for feature, value in request.features.items():
            if isinstance(value, (int, float)):
                metrics.track_feature_value(feature, value)
        
        # Check for drift (in real system, this would be batched)
        drift_result = drift_detector.detect_drift(features_df)
        for feature, drift_info in drift_result["feature_drifts"].items():
            if "statistic" in drift_info:
                metrics.track_drift_score(
                    feature, 
                    drift_info["statistic"],
                    drift_info.get("test", "unknown")
                )
        
        # TODO: Make actual prediction with the model
        # For now, we'll just return a mock response
        prediction = 1  # Placeholder
        probability = 0.85  # Placeholder
        
        # Track successful prediction
        metrics.track_prediction("success")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return PredictionResponse(
            prediction=prediction,
            prediction_probability=probability,
            request_id=request.request_id,
            model_version=MODEL_VERSION,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        # Track error
        metrics.track_error("prediction_error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
