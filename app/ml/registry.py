"""
Model registry for ML Risk Pipeline.
Manages model versions, metadata, and deployment.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from app.config import MODELS_DIR, METRICS_DIR
from app.utils.logging import app_logger


class ModelRegistry:
    """
    Manages model versions and metadata.
    Maintains a registry.json file with all model information.
    """
    
    def __init__(self, registry_path: Path = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry JSON file
        """
        if registry_path is None:
            registry_path = MODELS_DIR / "registry.json"
        
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self) -> dict:
        """Load registry from disk or create new one."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            app_logger.info(f"Loaded registry with {len(registry.get('models', []))} models")
            return registry
        else:
            app_logger.info("Creating new registry")
            return {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "models": []
            }
    
    def _save_registry(self):
        """Save registry to disk."""
        self.registry["updated_at"] = datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        app_logger.info(f"Registry saved to {self.registry_path}")
    
    def register_model(
        self,
        model_version: str,
        model_type: str,
        model_path: str,
        dataset_hash: str,
        metrics: dict = None,
        status: str = "registered",
        metadata: dict = None
    ) -> dict:
        """
        Register a new model in the registry.
        
        Args:
            model_version: Model version string
            model_type: Type of model (e.g., 'random_forest')
            model_path: Path to model file
            dataset_hash: Hash of training dataset
            metrics: Model metrics dictionary
            status: Model status (registered, approved, production, deprecated)
            metadata: Additional metadata
            
        Returns:
            Registered model entry
        """
        # Check if model already exists
        existing = self.get_model(model_version)
        if existing:
            app_logger.warning(f"Model {model_version} already registered. Updating...")
            return self.update_model(model_version, metrics=metrics, status=status)
        
        model_entry = {
            "model_version": model_version,
            "model_type": model_type,
            "model_path": str(model_path),
            "dataset_hash": dataset_hash,
            "registered_at": datetime.now().isoformat(),
            "status": status,
            "metrics": metrics or {},
            "metadata": metadata or {}
        }
        
        self.registry["models"].append(model_entry)
        self._save_registry()
        
        app_logger.info(f"Registered model: {model_version}")
        
        return model_entry
    
    def update_model(
        self,
        model_version: str,
        metrics: dict = None,
        status: str = None,
        metadata: dict = None
    ) -> Optional[dict]:
        """
        Update an existing model entry.
        
        Args:
            model_version: Model version to update
            metrics: Updated metrics
            status: Updated status
            metadata: Updated metadata
            
        Returns:
            Updated model entry or None if not found
        """
        for model in self.registry["models"]:
            if model["model_version"] == model_version:
                if metrics:
                    model["metrics"] = metrics
                if status:
                    model["status"] = status
                if metadata:
                    model["metadata"].update(metadata)
                
                model["updated_at"] = datetime.now().isoformat()
                self._save_registry()
                
                app_logger.info(f"Updated model: {model_version}")
                return model
        
        app_logger.warning(f"Model {model_version} not found for update")
        return None
    
    def get_model(self, model_version: str) -> Optional[dict]:
        """
        Get model entry by version.
        
        Args:
            model_version: Model version to retrieve
            
        Returns:
            Model entry or None if not found
        """
        for model in self.registry["models"]:
            if model["model_version"] == model_version:
                return model
        return None
    
    def list_models(self, status: str = None) -> List[dict]:
        """
        List all models, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of model entries
        """
        models = self.registry["models"]
        if status:
            models = [m for m in models if m.get("status") == status]
        return models
    
    def get_latest_model(self, status: str = None) -> Optional[dict]:
        """
        Get the most recently registered model.
        
        Args:
            status: Optional status filter
            
        Returns:
            Latest model entry or None
        """
        models = self.list_models(status=status)
        if not models:
            return None
        
        # Sort by registered_at descending
        sorted_models = sorted(
            models,
            key=lambda m: m.get("registered_at", ""),
            reverse=True
        )
        
        return sorted_models[0]
    
    def get_best_model(self, metric: str = "f1", status: str = None) -> Optional[dict]:
        """
        Get the best model by a specific metric.
        
        Args:
            metric: Metric to compare (default: f1)
            status: Optional status filter
            
        Returns:
            Best model entry or None
        """
        models = self.list_models(status=status)
        if not models:
            return None
        
        # Filter models that have the specified metric
        models_with_metric = [
            m for m in models
            if m.get("metrics", {}).get(metric) is not None
        ]
        
        if not models_with_metric:
            return None
        
        # Sort by metric descending
        sorted_models = sorted(
            models_with_metric,
            key=lambda m: m["metrics"].get(metric, 0),
            reverse=True
        )
        
        return sorted_models[0]
    
    def set_production_model(self, model_version: str) -> bool:
        """
        Set a model as the production model.
        Demotes all other models from production.
        
        Args:
            model_version: Model version to promote
            
        Returns:
            True if successful, False otherwise
        """
        # Demote all current production models
        for model in self.registry["models"]:
            if model.get("status") == "production":
                model["status"] = "approved"
        
        # Promote new model
        result = self.update_model(model_version, status="production")
        
        if result:
            app_logger.info(f"Set {model_version} as production model")
            return True
        
        return False


def sync_registry_with_metrics():
    """
    Scan metrics directory and update registry with evaluation results.
    """
    registry = ModelRegistry()
    
    # Load metadata for dataset hash
    from app.config import PROCESSED_DATA_DIR
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"
    dataset_hash = "unknown"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            dataset_hash = metadata.get("dataset_hash", "unknown")
    
    # Scan metrics files
    metrics_files = list(METRICS_DIR.glob("*_metrics.json"))
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            evaluation = json.load(f)
        
        model_version = evaluation["model_version"]
        model_type = evaluation["model_type"]
        metrics = evaluation["metrics"]
        
        # Find model file
        model_path = MODELS_DIR / f"{model_version}.joblib"
        
        if model_path.exists():
            registry.register_model(
                model_version=model_version,
                model_type=model_type,
                model_path=str(model_path),
                dataset_hash=dataset_hash,
                metrics=metrics,
                status="approved"
            )
    
    app_logger.info(f"Registry synced with {len(metrics_files)} metrics files")
    
    return registry
