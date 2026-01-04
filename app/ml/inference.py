"""
Inference module for ML Risk Pipeline.
Handles loading models and making predictions.
"""
import time
import numpy as np
import pandas as pd
from typing import Tuple
from app.ml.train import ModelTrainer
from app.ml.preprocessing import DataPreprocessor
from app.ml.registry import ModelRegistry
from app.ml.monitoring import get_monitor
from app.config import MODELS_DIR
from app.utils.logging import app_logger


class InferenceEngine:
    """Handles model inference with preprocessing and monitoring."""
    
    def __init__(self):
        """Initialize inference engine."""
        self.model = None
        self.preprocessor = None
        self.model_version = None
        self.model_info = None
        self.registry = ModelRegistry()
    
    def load_model(self, model_version: str = None):
        """
        Load a model for inference.
        If no version specified, loads the latest approved model.
        
        Args:
            model_version: Optional model version to load
        """
        # Get model from registry
        if model_version:
            model_entry = self.registry.get_model(model_version)
        else:
            # Get best model by F1 score
            model_entry = self.registry.get_best_model(metric="f1", status="approved")
            if not model_entry:
                # Fallback to latest
                model_entry = self.registry.get_latest_model(status="approved")
        
        if not model_entry:
            raise ValueError("No approved models found in registry")
        
        model_version = model_entry["model_version"]
        model_path = MODELS_DIR / f"{model_version}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        self.model = ModelTrainer.load(model_path)
        self.model_version = model_version
        self.model_info = model_entry
        
        # Load preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        if preprocessor_path.exists():
            self.preprocessor = DataPreprocessor.load(preprocessor_path)
        else:
            app_logger.warning("Preprocessor not found. Predictions may fail if data is not preprocessed.")
        
        app_logger.info(f"Loaded model: {model_version} ({self.model.model_type})")
    
    def predict(
        self,
        features: list,
        preprocess: bool = True
    ) -> Tuple[int, float, str, float]:
        """
        Make a prediction.
        
        Args:
            features: List of feature values
            preprocess: Whether to preprocess features
            
        Returns:
            Tuple of (prediction, probability, risk_level, latency_ms)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Convert to DataFrame
        if self.preprocessor and self.preprocessor.feature_names:
            feature_names = self.preprocessor.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(len(features))]
        
        X = pd.DataFrame([features], columns=feature_names)
        
        # Preprocess if needed
        if preprocess and self.preprocessor:
            X = self.preprocessor.transform(X)
        
        # Predict
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        probability = float(proba[1])  # Probability of positive class
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log prediction
        monitor = get_monitor()
        monitor.log_prediction(
            model_version=self.model_version,
            features=features,
            prediction=int(prediction),
            probability=probability,
            latency_ms=latency_ms,
            risk_level=risk_level
        )
        
        return int(prediction), probability, risk_level, latency_ms
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_version": self.model_version,
            "model_type": self.model.model_type,
            "model_name": self.model_info.get("model_type", "unknown"),
            "trained_at": self.model_info.get("registered_at", "unknown"),
            "dataset_hash": self.model_info.get("dataset_hash", "unknown"),
            "feature_count": len(self.model.feature_names) if self.model.feature_names else 0,
            "metrics": self.model_info.get("metrics", {}),
            "status": self.model_info.get("status", "unknown")
        }
        
        return info


# Global inference engine instance
_engine = None


def get_inference_engine() -> InferenceEngine:
    """Get or create global inference engine instance."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        try:
            _engine.load_model()  # Load best model by default
            app_logger.info("Inference engine initialized")
        except Exception as e:
            app_logger.error(f"Failed to initialize inference engine: {e}")
    return _engine
