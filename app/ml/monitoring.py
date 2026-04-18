"""
Advanced monitoring module — logs predictions, computes statistics, serves live feed.
"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from app.config import MONITORING_DIR
from app.utils.logging import app_logger


class PredictionMonitor:
    """Monitors and logs ML prediction requests."""

    def __init__(self, log_file: Path = None):
        if log_file is None:
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = MONITORING_DIR / f"predictions_{date_str}.jsonl"
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        model_version: str,
        features: list,
        prediction: int,
        probability: float,
        latency_ms: float,
        risk_level: str,
        metadata: dict = None,
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "latency_ms": float(latency_ms),
            "feature_stats": {
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "count": len(features),
            },
            "metadata": metadata or {},
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _read_logs(self, limit: int) -> list:
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text().splitlines()
        parsed = []
        for line in lines[-limit:]:
            try:
                parsed.append(json.loads(line))
            except Exception:
                pass
        return parsed

    def get_recent_predictions(self, limit: int = 20) -> list:
        """Return the most recent prediction entries, newest first."""
        logs = self._read_logs(limit)
        return list(reversed(logs))

    def get_statistics(self, limit: int = 1000) -> dict:
        logs = self._read_logs(limit)
        if not logs:
            return {"total_predictions": 0, "message": "No monitoring data available"}

        predictions = [lg["prediction"] for lg in logs]
        probabilities = [lg["probability"] for lg in logs]
        latencies = [lg["latency_ms"] for lg in logs]
        risk_levels = [lg["risk_level"] for lg in logs]

        return {
            "total_predictions": len(logs),
            "prediction_distribution": {
                "positive": int(sum(1 for p in predictions if p == 1)),
                "negative": int(sum(1 for p in predictions if p == 0)),
                "positive_rate": float(np.mean(predictions)),
            },
            "risk_distribution": {
                "low": int(sum(1 for r in risk_levels if r == "Low")),
                "medium": int(sum(1 for r in risk_levels if r == "Medium")),
                "high": int(sum(1 for r in risk_levels if r == "High")),
            },
            "latency_stats": {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
            },
            "probability_stats": {
                "mean": float(np.mean(probabilities)),
                "median": float(np.median(probabilities)),
                "std": float(np.std(probabilities)),
            },
            "time_range": {
                "first": logs[0]["timestamp"],
                "last": logs[-1]["timestamp"],
            },
        }


_monitor: PredictionMonitor = None


def get_monitor() -> PredictionMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor
