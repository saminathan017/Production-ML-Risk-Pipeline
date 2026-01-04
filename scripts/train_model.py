#!/usr/bin/env python3
"""
Script to train ML models.
Trains both baseline and improved models, saving artifacts.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.data_loader import load_processed_data
from app.ml.train import train_model
from app.utils.logging import app_logger
import json
from datetime import datetime


def main():
    """Main function to train models."""
    app_logger.info("="*60)
    app_logger.info("MODEL TRAINING SCRIPT")
    app_logger.info("="*60)
    
    # Load processed data
    app_logger.info("Loading processed data...")
    X_train, y_train = load_processed_data("train")
    
    # Train baseline model (Logistic Regression)
    app_logger.info("\n" + "="*60)
    app_logger.info("Training BASELINE model (Logistic Regression)")
    app_logger.info("="*60)
    baseline_trainer, baseline_version = train_model(
        X_train, y_train,
        model_type="logistic_regression",
        save_model=True
    )
    
    # Train improved model (Random Forest)
    app_logger.info("\n" + "="*60)
    app_logger.info("Training IMPROVED model (Random Forest)")
    app_logger.info("="*60)
    improved_trainer, improved_version = train_model(
        X_train, y_train,
        model_type="random_forest",
        save_model=True
    )
    
    # Train advanced model (Gradient Boosting) - optional
    app_logger.info("\n" + "="*60)
    app_logger.info("Training ADVANCED model (Gradient Boosting)")
    app_logger.info("="*60)
    advanced_trainer, advanced_version = train_model(
        X_train, y_train,
        model_type="gradient_boosting",
        save_model=True
    )
    
    app_logger.info("\n" + "="*60)
    app_logger.info("MODEL TRAINING COMPLETE")
    app_logger.info("="*60)
    
    print("\n✓ Model training successful!")
    print(f"\nTrained Models:")
    print(f"  1. Baseline (Logistic Regression): {baseline_version}")
    print(f"  2. Improved (Random Forest): {improved_version}")
    print(f"  3. Advanced (Gradient Boosting): {advanced_version}")
    print(f"\nModels saved in artifacts/models/")
    print(f"\nNext step: Run evaluation script")
    print(f"  ./venv/bin/python scripts/evaluate_model.py")


if __name__ == "__main__":
    main()
