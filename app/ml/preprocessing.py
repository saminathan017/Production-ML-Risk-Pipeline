"""
Data preprocessing pipeline for ML Risk Pipeline.
Handles train/test split, scaling, and missing value handling.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from app.config import PROCESSED_DATA_DIR, MODELS_DIR, settings
from app.utils.logging import app_logger


class DataPreprocessor:
    """Handles data preprocessing including scaling and missing value imputation."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Self for chaining
        """
        self.feature_names = X.columns.tolist()
        
        # Handle missing values by filling with median
        X_clean = X.fillna(X.median())
        
        # Fit scaler
        self.scaler.fit(X_clean)
        self.is_fitted = True
        
        app_logger.info(f"Preprocessor fitted on {len(X)} samples with {len(self.feature_names)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.transform(X_clean)
        
        # Return as DataFrame with original column names
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X).transform(X)
    
    def save(self, filepath: Path):
        """Save preprocessor to disk."""
        joblib.dump(self, filepath)
        app_logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'DataPreprocessor':
        """Load preprocessor from disk."""
        preprocessor = joblib.load(filepath)
        app_logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DataPreprocessor]:
    """
    Complete preprocessing pipeline: split, scale, and prepare data.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    if random_state is None:
        random_state = settings.random_seed
    
    app_logger.info(f"Starting preprocessing with test_size={test_size}, random_state={random_state}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensure balanced classes in both sets
    )
    
    app_logger.info(f"Split data: train={len(X_train)}, test={len(X_test)}")
    
    # Initialize and fit preprocessor on training data only
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    
    app_logger.info("Preprocessing complete")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def get_feature_statistics(X: pd.DataFrame) -> dict:
    """
    Compute summary statistics for features.
    
    Args:
        X: Features DataFrame
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "n_features": X.shape[1],
        "feature_names": X.columns.tolist(),
        "missing_values": X.isnull().sum().to_dict(),
        "mean": X.mean().to_dict(),
        "std": X.std().to_dict(),
        "min": X.min().to_dict(),
        "max": X.max().to_dict(),
    }
    
    return stats
