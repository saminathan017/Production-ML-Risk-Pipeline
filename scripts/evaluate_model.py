#!/usr/bin/env python3
"""
Script to evaluate trained models.
Evaluates all models and saves metrics and plots.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.data_loader import load_processed_data
from app.ml.train import ModelTrainer
from app.ml.evaluate import evaluate_model, print_evaluation_summary
from app.utils.logging import app_logger
from app.config import MODELS_DIR


def main():
    """Main function to evaluate models."""
    app_logger.info("="*60)
    app_logger.info("MODEL EVALUATION SCRIPT")
    app_logger.info("="*60)
    
    # Load test data
    app_logger.info("Loading test data...")
    X_test, y_test = load_processed_data("test")
    
    # Find all model files
    model_files = sorted(MODELS_DIR.glob("model_v*.joblib"))
    
    if not model_files:
        print("\n❌ No trained models found!")
        print("Please run training first:")
        print("  ./venv/bin/python scripts/train_model.py")
        return
    
    app_logger.info(f"Found {len(model_files)} model(s) to evaluate")
    
    evaluations = []
    
    # Evaluate each model
    for model_path in model_files:
        model_version = model_path.stem  # filename without extension
        
        app_logger.info(f"\n{'='*60}")
        app_logger.info(f"Evaluating: {model_version}")
        app_logger.info(f"{'='*60}")
        
        try:
            # Load model
            trainer = ModelTrainer.load(model_path)
            
            # Evaluate
            evaluation = evaluate_model(
                trainer, X_test, y_test,
                model_version=model_version,
                save_artifacts=True
            )
            
            evaluations.append(evaluation)
            print_evaluation_summary(evaluation)
            
        except Exception as e:
            app_logger.error(f"Failed to evaluate {model_version}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Evaluated {len(evaluations)} model(s)")
    print(f"\nArtifacts saved:")
    print(f"  - Metrics JSON: artifacts/metrics/")
    print(f"  - Plots: artifacts/plots/")
    
    # Best model
    if evaluations:
        best_model = max(evaluations, key=lambda x: x['metrics']['f1'])
        print(f"\n🏆 Best Model (by F1 score):")
        print(f"  {best_model['model_version']} ({best_model['model_type']})")
        print(f"  F1 Score: {best_model['metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
