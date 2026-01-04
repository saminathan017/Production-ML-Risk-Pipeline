#!/usr/bin/env python3
"""
Smoke test script for ML Risk Pipeline.
Tests the complete pipeline end-to-end.
"""
import sys
import time
import subprocess
from pathlib import Path
import requests
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logging import app_logger


def test_data_preparation():
    """Test data preparation."""
    print("\n" + "="*60)
    print("TEST 1: Data Preparation")
    print("="*60)
    
    from app.ml.data_loader import load_data
    X, y, dataset_name = load_data()
    
    assert X is not None, "Failed to load data"
    assert len(X) > 0, "Dataset is empty"
    
    print(f"✓ Data loaded: {dataset_name}")
    print(f"✓ Samples: {len(X)}, Features: {X.shape[1]}")
    
    return True


def test_model_training():
    """Test model training."""
    print("\n" + "="*60)
    print("TEST 2: Model Training")
    print("="*60)
    
    from app.ml.data_loader import load_processed_data
    from app.ml.train import train_model
    from app.config import MODELS_DIR
    
    X_train, y_train = load_processed_data("train")
    
    # Quick test with a small model
    trainer, version = train_model(X_train[:100], y_train[:100], "logistic_regression", save_model=False)
    
    assert trainer.is_trained, "Model training failed"
    
    print(f"✓ Model trained successfully")
    
    # Check if full models exist
    model_files = list(MODELS_DIR.glob("model_v*.joblib"))
    print(f"✓ Found {len(model_files)} trained model(s)")
    
    return True


def test_model_evaluation():
    """Test model evaluation."""
    print("\n" + "="*60)
    print("TEST 3: Model Evaluation")
    print("="*60)
    
    from app.ml.data_loader import load_processed_data
    from app.ml.train import ModelTrainer
    from app.ml.evaluate import compute_metrics
    from app.config import MODELS_DIR
    
    X_test, y_test = load_processed_data("test")
    
    # Load latest model
    model_files = sorted(MODELS_DIR.glob("model_v*.joblib"))
    assert len(model_files) > 0, "No trained models found"
    
    trainer = ModelTrainer.load(model_files[-1])
    y_pred = trainer.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    print(f"✓ Model evaluated")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
    
    return True


def test_registry():
    """Test model registry."""
    print("\n" + "="*60)
    print("TEST 4: Model Registry")
    print("="*60)
    
    from app.ml.registry import ModelRegistry
    
    registry = ModelRegistry()
    models = registry.list_models()
    
    assert len(models) > 0, "No models in registry"
    
    best_model = registry.get_best_model(metric="f1")
    assert best_model is not None, "Could not find best model"
    
    print(f"✓ Registry has {len(models)} model(s)")
    print(f"✓ Best model: {best_model['model_version']}")
    
    return True


def test_inference():
    """Test inference engine."""
    print("\n" + "="*60)
    print("TEST 5: Inference Engine")
    print("="*60)
    
    from app.ml.inference import InferenceEngine
    
    engine = InferenceEngine()
    engine.load_model()
    
    # Make a test prediction
    test_features = [0.5] * 30  # Example features
    prediction, probability, risk_level, latency = engine.predict(test_features)
    
    print(f"✓ Inference engine loaded")
    print(f"✓ Test prediction made:")
    print(f"  - Prediction: {prediction}")
    print(f"  - Probability: {probability:.4f}")
    print(f"  - Risk Level: {risk_level}")
    print(f"  - Latency: {latency:.2f}ms")
    
    return True


def test_api(port=8000):
    """Test FastAPI endpoints."""
    print("\n" + "="*60)
    print("TEST 6: API Endpoints")
    print("="*60)
    
    base_url = f"http://localhost:{port}/api"
    
    # Start server in background
    print("Starting API server...")
    server_process = subprocess.Popen(
        [
            "./venv/bin/python", "-m", "uvicorn",
            "app.main:app", "--host", "0.0.0.0", f"--port={port}"
        ],
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        print("✓ Health endpoint OK")
        
        # Test model info endpoint
        response = requests.get(f"{base_url}/model")
        assert response.status_code == 200, f"Model info failed: {response.status_code}"
        model_info = response.json()
        print(f"✓ Model info endpoint OK: {model_info['model_version']}")
        
        # Test metrics endpoint
        response = requests.get(f"{base_url}/metrics")
        assert response.status_code == 200, f"Metrics failed: {response.status_code}"
        print("✓ Metrics endpoint OK")
        
        # Test prediction endpoint
        test_features = [0.5] * 30
        response = requests.post(
            f"{base_url}/predict",
            json={"features": test_features}
        )
        assert response.status_code == 200, f"Prediction failed: {response.status_code}"
        pred_result = response.json()
        print(f"✓ Prediction endpoint OK: {pred_result['risk_level']} risk")
        
        return True
        
    finally:
        # Stop server
        server_process.terminate()
        server_process.wait(timeout=5)
        print("✓ Server stopped")


def main():
    """Run all smoke tests."""
    print("\n" + "="*80)
    print(" "*25 + "ML RISK PIPELINE - SMOKE TEST")
    print("="*80)
    
    tests = [
        ("Data Preparation", test_data_preparation),
        ("Model Training", test_model_training),
        ("Model Evaluation", test_model_evaluation),
        ("Model Registry", test_registry),
        ("Inference Engine", test_inference),
        ("API Endpoints", test_api),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("SMOKE TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The ML Risk Pipeline is fully functional.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please investigate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
