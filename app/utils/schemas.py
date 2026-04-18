"""
Pydantic schemas for NeuralNexus ML Risk Intelligence Platform.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="30-dimensional feature vector", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {"features": [0.5] * 30}
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(..., description="List of feature vectors (max 100)")


class BatchPredictionResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int
    total_latency_ms: float
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    trained_at: str
    dataset_hash: str
    feature_count: int
    metrics: Optional[Dict[str, float]] = None


class MetricsResponse(BaseModel):
    model_version: str
    metrics: Dict[str, float]
    evaluated_at: str
    dataset_size: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool


class SystemStatusResponse(BaseModel):
    status: str
    uptime_seconds: int
    model_loaded: bool
    model_version: str
    active_connections: int
    python_version: str
    platform: str
    server_time: str
