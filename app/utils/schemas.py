"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Schema for prediction requests."""
    features: List[float] = Field(
        ...,
        description="List of feature values for prediction",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.8, 0.2, 0.6]
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Probability of positive class")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Request latency in milliseconds")


class ModelInfo(BaseModel):
    """Schema for model information."""
    model_name: str
    model_version: str
    model_type: str
    trained_at: str
    dataset_hash: str
    feature_count: int
    metrics: Optional[Dict[str, float]] = None


class MetricsResponse(BaseModel):
    """Schema for metrics response."""
    model_version: str
    metrics: Dict[str, float]
    evaluated_at: str
    dataset_size: int


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    app_name: str
    version: str
    model_loaded: bool
