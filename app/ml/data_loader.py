"""
Data loading utilities for ML Risk Pipeline.
Supports loading from CSV files and sklearn datasets as fallback.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, settings
from app.utils.logging import app_logger
import hashlib


def load_data_from_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load data from a CSV file.
    Assumes the last column is the target variable.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    app_logger.info(f"Loading data from CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    app_logger.info(f"Loaded {len(df)} samples with {X.shape[1]} features")
    
    return X, y


def load_sklearn_fallback_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load breast cancer dataset from sklearn as fallback.
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    app_logger.info("Loading sklearn breast cancer dataset as fallback")
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    app_logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    return X, y


def load_data(csv_filename: str = None) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Load data from CSV if available, otherwise use sklearn fallback.
    
    Args:
        csv_filename: Optional CSV filename in raw data directory
        
    Returns:
        Tuple of (features DataFrame, target Series, dataset name)
    """
    # Try to load from CSV if specified
    if csv_filename:
        csv_path = RAW_DATA_DIR / csv_filename
        if csv_path.exists():
            X, y = load_data_from_csv(csv_path)
            return X, y, csv_filename
        else:
            app_logger.warning(f"CSV file not found: {csv_path}. Using fallback dataset.")
    
    # Check for any CSV in raw data directory
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        app_logger.info(f"Found CSV file: {csv_files[0].name}")
        X, y = load_data_from_csv(csv_files[0])
        return X, y, csv_files[0].name
    
    # Fallback to sklearn dataset
    app_logger.info("No CSV files found. Using sklearn fallback dataset.")
    X, y = load_sklearn_fallback_dataset()
    return X, y, "sklearn_breast_cancer"


def compute_dataset_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """
    Compute a hash of the dataset for tracking.
    
    Args:
        X: Features DataFrame
        y: Target Series
        
    Returns:
        Hash string
    """
    # Compute hash based on shape and sample of data
    data_str = f"{X.shape}_{y.shape}_{X.iloc[0].values.tobytes() if len(X) > 0 else ''}"
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


def save_processed_data(X: pd.DataFrame, y: pd.Series, split_name: str):
    """
    Save processed data to disk.
    
    Args:
        X: Features DataFrame
        y: Target Series
        split_name: Name of the split (e.g., 'train', 'test')
    """
    X_path = PROCESSED_DATA_DIR / f"X_{split_name}.csv"
    y_path = PROCESSED_DATA_DIR / f"y_{split_name}.csv"
    
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)
    
    app_logger.info(f"Saved {split_name} data: X shape {X.shape}, y shape {y.shape}")


def load_processed_data(split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data from disk.
    
    Args:
        split_name: Name of the split (e.g., 'train', 'test')
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    X_path = PROCESSED_DATA_DIR / f"X_{split_name}.csv"
    y_path = PROCESSED_DATA_DIR / f"y_{split_name}.csv"
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    
    app_logger.info(f"Loaded {split_name} data: X shape {X.shape}, y shape {y.shape}")
    
    return X, y
