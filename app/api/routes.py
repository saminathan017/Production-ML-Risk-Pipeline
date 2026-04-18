"""
Advanced API routes for NeuralNexus ML Risk Intelligence Platform.
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from app.utils.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    MetricsResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    SystemStatusResponse,
)
from app.config import settings, METRICS_DIR, MODELS_DIR
from app.ml.inference import get_inference_engine
from app.ml.monitoring import get_monitor
from app.utils.logging import app_logger
import json
import asyncio
import platform
import time
from pathlib import Path
from datetime import datetime

router = APIRouter()

# ─── WebSocket connection manager ────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    engine = get_inference_engine()
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "model_loaded": engine.model is not None,
    }


# ─── Prediction ───────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        engine = get_inference_engine()
        if engine.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please contact administrator.")

        prediction, probability, risk_level, latency_ms = engine.predict(
            features=request.features,
            preprocess=True,
        )

        result = {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "model_version": engine.model_version,
            "latency_ms": latency_ms,
        }

        # Broadcast to live WebSocket subscribers
        await manager.broadcast({
            "event": "prediction",
            "timestamp": datetime.now().isoformat(),
            **result,
        })

        return result

    except ValueError as e:
        app_logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch inference endpoint — up to 100 samples per request."""
    if len(request.samples) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 samples per batch request.")

    engine = get_inference_engine()
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    start = time.time()
    for features in request.samples:
        pred, prob, risk, lat = engine.predict(features=features, preprocess=True)
        results.append({
            "prediction": pred,
            "probability": prob,
            "risk_level": risk,
            "latency_ms": lat,
        })

    total_ms = (time.time() - start) * 1000
    return {
        "results": results,
        "count": len(results),
        "total_latency_ms": total_ms,
        "model_version": engine.model_version,
    }


# ─── Model ────────────────────────────────────────────────────────────────────

@router.get("/model", response_model=ModelInfo)
async def get_model_info():
    engine = get_inference_engine()
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine.get_model_info()


@router.get("/models/list")
async def list_models():
    """Return all models in the registry."""
    try:
        registry_path = MODELS_DIR / "registry.json"
        if not registry_path.exists():
            return {"models": [], "count": 0}

        with open(registry_path) as f:
            registry = json.load(f)

        models = list(registry.get("models", {}).values())
        models.sort(key=lambda x: x.get("registered_at", ""), reverse=True)
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Metrics ─────────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    engine = get_inference_engine()
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics_file = METRICS_DIR / f"{engine.model_version}_metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")

    with open(metrics_file) as f:
        data = json.load(f)

    return {
        "model_version": data["model_version"],
        "metrics": data["metrics"],
        "evaluated_at": data["evaluated_at"],
        "dataset_size": data["dataset_size"],
    }


# ─── Feature Importance ───────────────────────────────────────────────────────

@router.get("/feature-importance")
async def get_feature_importance():
    """Return top feature importances for tree-based models."""
    engine = get_inference_engine()
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw_model = engine.model.model
        feature_names = engine.model.feature_names or [f"feature_{i}" for i in range(30)]

        if hasattr(raw_model, "feature_importances_"):
            importances = raw_model.feature_importances_
            paired = sorted(
                zip(feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            importance_list = [{"feature": n, "importance": round(v, 6)} for n, v in paired[:20]]
            return {
                "feature_importance": importance_list,
                "model_type": engine.model.model_type,
                "total_features": len(feature_names),
            }
        elif hasattr(raw_model, "coef_"):
            import numpy as np
            coefs = np.abs(raw_model.coef_[0])
            paired = sorted(
                zip(feature_names, coefs.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            importance_list = [{"feature": n, "importance": round(v, 6)} for n, v in paired[:20]]
            return {
                "feature_importance": importance_list,
                "model_type": engine.model.model_type,
                "total_features": len(feature_names),
            }
        else:
            return {"feature_importance": [], "message": "Model does not expose feature importance", "model_type": engine.model.model_type}

    except Exception as e:
        app_logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Monitoring ───────────────────────────────────────────────────────────────

@router.get("/monitoring/stats")
async def get_monitoring_stats(limit: int = Query(default=1000, le=10000)):
    try:
        monitor = get_monitor()
        return monitor.get_statistics(limit=limit)
    except Exception as e:
        app_logger.error(f"Monitoring stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/recent")
async def get_recent_predictions(limit: int = Query(default=20, le=100)):
    """Return the most recent prediction log entries."""
    try:
        monitor = get_monitor()
        return {
            "predictions": monitor.get_recent_predictions(limit=limit),
            "count": limit,
        }
    except Exception as e:
        app_logger.error(f"Recent predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── System Status ────────────────────────────────────────────────────────────

@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Return lightweight system/runtime status (no psutil dependency)."""
    engine = get_inference_engine()
    uptime_file = Path("/tmp/neuralnexus_start.txt")
    if not uptime_file.exists():
        uptime_file.write_text(str(time.time()))
    start_ts = float(uptime_file.read_text().strip())
    uptime_seconds = int(time.time() - start_ts)

    return {
        "status": "operational",
        "uptime_seconds": uptime_seconds,
        "model_loaded": engine.model is not None,
        "model_version": engine.model_version or "none",
        "active_connections": len(manager.active),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "server_time": datetime.now().isoformat(),
    }


# ─── WebSocket live feed ──────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Real-time prediction event stream via WebSocket."""
    await manager.connect(ws)
    try:
        # Send welcome handshake
        await ws.send_json({"event": "connected", "message": "NeuralNexus live feed active", "timestamp": datetime.now().isoformat()})
        # Keep alive with periodic heartbeat
        while True:
            await asyncio.sleep(25)
            await ws.send_json({"event": "heartbeat", "timestamp": datetime.now().isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)
