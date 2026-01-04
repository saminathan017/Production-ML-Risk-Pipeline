"""
Monitoring module for ML Risk Pipeline.
Logs prediction requests, latency, and model performance.
"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from app.config import MONITORING_DIR
from app.utils.logging import app_logger


class PredictionMonitor:
    """Monitors and logs ML prediction requests."""
    
    def __init__(self, log_file: Path = None):
        """
        Initialize prediction monitor.
        
        Args:
            log_file: Path to monitoring log file (JSONL format)
        """
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = MONITORING_DIR / f"predictions_{timestamp}.jsonl"
        
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
        metadata: dict = None
    ):
        """
        Log a prediction request.
        
        Args:
            model_version: Model version used
            features: Input features
            prediction: Predicted class
            probability: Prediction probability
            latency_ms: Request latency in milliseconds
            risk_level: Computed risk level
            metadata: Additional metadata
        """
        log_entry = {
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
                "count": len(features)
            },
            "metadata": metadata or {}
        }
        
        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_statistics(self, limit: int = 100) -> dict:
        """
        Get monitoring statistics from recent logs.
        
        Args:
            limit: Number of recent records to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not self.log_file.exists():
            return {
                "total_predictions": 0,
                "message": "No monitoring data available"
            }
        
        # Read recent logs
        logs = []
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    logs.append(json.loads(line))
                except:
                    pass
        
        if not logs:
            return {
                "total_predictions": 0,
                "message": "No valid monitoring data"
            }
        
        # Compute statistics
        predictions = [log["prediction"] for log in logs]
        probabilities = [log["probability"] for log in logs]
        latencies = [log["latency_ms"] for log in logs]
        risk_levels = [log["risk_level"] for log in logs]
        
        stats = {
            "total_predictions": len(logs),
            "prediction_distribution": {
                "positive": sum(1 for p in predictions if p == 1),
                "negative": sum(1 for p in predictions if p == 0),
                "positive_rate": np.mean(predictions)
            },
            "risk_distribution": {
                "low": sum(1 for r in risk_levels if r == "Low"),
                "medium": sum(1 for r in risk_levels if r == "Medium"),
                "high": sum(1 for r in risk_levels if r == "High")
            },
            "latency_stats": {
                "mean_ms": np.mean(latencies),
                "median_ms": np.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "min_ms": np.min(latencies),
                "max_ms": np.max(latencies)
            },
            "probability_stats": {
                "mean": np.mean(probabilities),
                "median": np.median(probabilities),
                "std": np.std(probabilities)
            },
            "time_range": {
                "first": logs[0]["timestamp"],
                "last": logs[-1]["timestamp"]
            }
        }
        
        return stats


# Global monitor instance
_monitor = None


def get_monitor() -> PredictionMonitor:
    """Get or create global monitoring instance."""
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor
