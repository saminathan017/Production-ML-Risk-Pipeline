#!/usr/bin/env python3
"""
Script to download or prepare dataset for training.
Uses sklearn breast cancer dataset as default fallback.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.data_loader import load_data, save_processed_data, compute_dataset_hash
from app.ml.preprocessing import preprocess_data
from app.utils.logging import app_logger
from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    """Main function to prepare dataset."""
    app_logger.info("="*60)
    app_logger.info("DATA PREPARATION SCRIPT")
    app_logger.info("="*60)
    
    # Load data
    X, y, dataset_name = load_data()
    
    # Compute dataset hash
    dataset_hash = compute_dataset_hash(X, y)
    app_logger.info(f"Dataset: {dataset_name}")
    app_logger.info(f"Dataset hash: {dataset_hash}")
    app_logger.info(f"Shape: {X.shape}")
    app_logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(X, y)
    
    # Save processed data
    save_processed_data(X_train, y_train, "train")
    save_processed_data(X_test, y_test, "test")
    
    # Save metadata
    import json
    metadata = {
        "dataset_name": dataset_name,
        "dataset_hash": dataset_hash,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": X.columns.tolist(),
        "target_distribution": y.value_counts().to_dict()
    }
    
    metadata_path = PROCESSED_DATA_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    app_logger.info(f"Metadata saved to {metadata_path}")
    app_logger.info("="*60)
    app_logger.info("DATA PREPARATION COMPLETE")
    app_logger.info("="*60)
    
    print("\n✓ Data preparation successful!")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Train samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")


if __name__ == "__main__":
    main()
