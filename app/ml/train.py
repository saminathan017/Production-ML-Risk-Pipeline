"""
Model training module for ML Risk Pipeline.
Implements baseline (Logistic Regression) and improved (Random Forest) models.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from datetime import datetime
from pathlib import Path
from app.config import MODELS_DIR, settings
from app.utils.logging import app_logger


class ModelTrainer:
    """Handles model training with multiple algorithms."""
    
    AVAILABLE_MODELS = {
        "logistic_regression": {
            "class": LogisticRegression,
            "params": {
                "max_iter": 1000,
                "random_state": settings.random_seed,
                "class_weight": "balanced"
            },
            "description": "Baseline logistic regression model"
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": settings.random_seed,
                "n_jobs": -1,
                "class_weight": "balanced"
            },
            "description": "Improved random forest model"
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": settings.random_seed
            },
            "description": "Advanced gradient boosting model"
        }
    }
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize trainer with specified model type.
        
        Args:
            model_type: Type of model to train
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model_type = model_type
        model_config = self.AVAILABLE_MODELS[model_type]
        
        self.model = model_config["class"](**model_config["params"])
        self.description = model_config["description"]
        self.feature_names = None
        self.is_trained = False
        
        app_logger.info(f"Initialized {model_type} trainer: {self.description}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'ModelTrainer':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Self for chaining
        """
        app_logger.info(f"Training {self.model_type} on {len(X_train)} samples...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Log training completion
        train_score = self.model.score(X_train, y_train)
        app_logger.info(f"Training complete. Train accuracy: {train_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Probability array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def save(self, filepath: Path):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_metadata = {
            "model": self.model,
            "model_type": self.model_type,
            "description": self.description,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained
        }
        
        joblib.dump(model_metadata, filepath)
        app_logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'ModelTrainer':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded ModelTrainer instance
        """
        model_metadata = joblib.load(filepath)
        
        trainer = ModelTrainer(model_metadata["model_type"])
        trainer.model = model_metadata["model"]
        trainer.description = model_metadata["description"]
        trainer.feature_names = model_metadata["feature_names"]
        trainer.is_trained = model_metadata["is_trained"]
        
        app_logger.info(f"Model loaded from {filepath}")
        
        return trainer


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    save_model: bool = True
) -> tuple[ModelTrainer, str]:
    """
    Train a model and optionally save it.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        save_model: Whether to save the model
        
    Returns:
        Tuple of (trained ModelTrainer, model version string)
    """
    # Initialize and train
    trainer = ModelTrainer(model_type=model_type)
    trainer.train(X_train, y_train)
    
    # Generate version string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_version = f"model_v{timestamp}"
    
    # Save model
    if save_model:
        model_filename = f"{model_version}.joblib"
        model_path = MODELS_DIR / model_filename
        trainer.save(model_path)
    
    return trainer, model_version
