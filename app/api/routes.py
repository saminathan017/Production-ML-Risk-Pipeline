"""
API routes for the ML Risk Pipeline.
"""
from fastapi import APIRouter, HTTPException
from app.utils.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    MetricsResponse,
    HealthResponse
)
from app.config import settings
from app.ml.inference import get_inference_engine
from app.ml.monitoring import get_monitor
from app.utils.logging import app_logger
import json
from pathlib import Path
from app.config import METRICS_DIR

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    engine = get_inference_engine()
    model_loaded = engine.model is not None
    
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "model_loaded": model_loaded
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction based on input features.
    """
    try:
        engine = get_inference_engine()
        
        if engine.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please contact administrator."
            )
        
        # Make prediction
        prediction, probability, risk_level, latency_ms = engine.predict(
            features=request.features,
            preprocess=True
        )
        
        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "model_version": engine.model_version,
            "latency_ms": latency_ms
        }
        
    except ValueError as e:
        app_logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        app_logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/model", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the current model.
    """
    try:
        engine = get_inference_engine()
        
        if engine.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        info = engine.get_model_info()
        
        return info
        
    except Exception as e:
        app_logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get the latest model metrics.
    """
    try:
        engine = get_inference_engine()
        
        if engine.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        # Load metrics from file
        metrics_file = METRICS_DIR / f"{engine.model_version}_metrics.json"
        
        if not metrics_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Metrics file not found"
            )
        
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        return {
            "model_version": metrics_data["model_version"],
            "metrics": metrics_data["metrics"],
            "evaluated_at": metrics_data["evaluated_at"],
            "dataset_size": metrics_data["dataset_size"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/stats")
async def get_monitoring_stats():
    """
    Get monitoring statistics.
    """
    try:
        monitor = get_monitor()
        stats = monitor.get_statistics(limit=1000)
        return stats
    except Exception as e:
        app_logger.error(f"Error getting monitoring stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
