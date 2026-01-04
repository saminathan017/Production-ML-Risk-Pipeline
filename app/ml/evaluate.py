"""
Model evaluation module for ML Risk Pipeline.
Computes metrics, generates confusion matrix, and creates plots.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from pathlib import Path
from app.ml.train import ModelTrainer
from app.config import METRICS_DIR, PLOTS_DIR
from app.utils.logging import app_logger


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
    }
    
    # Add ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        except Exception as e:
            app_logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = 0.0
    
    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix"
):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        square=True,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    app_logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve"
):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
        title: Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    app_logger.info(f"ROC curve saved to {save_path}")


def evaluate_model(
    trainer: ModelTrainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_version: str,
    save_artifacts: bool = True
) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        trainer: Trained ModelTrainer instance
        X_test: Test features
        y_test: Test labels
        model_version: Model version string
        save_artifacts: Whether to save metrics and plots
        
    Returns:
        Dictionary of evaluation results
    """
    app_logger.info(f"Evaluating model: {model_version}")
    
    # Make predictions
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    app_logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    # Prepare evaluation results
    evaluation = {
        "model_version": model_version,
        "model_type": trainer.model_type,
        "evaluated_at": datetime.now().isoformat(),
        "dataset_size": len(X_test),
        "metrics": metrics
    }
    
    if save_artifacts:
        # Save metrics
        metrics_filename = f"{model_version}_metrics.json"
        metrics_path = METRICS_DIR / metrics_filename
        with open(metrics_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        app_logger.info(f"Metrics saved to {metrics_path}")
        
        # Save confusion matrix plot
        cm_filename = f"{model_version}_confusion_matrix.png"
        cm_path = PLOTS_DIR / cm_filename
        plot_confusion_matrix(
            y_test, y_pred, cm_path,
            title=f"Confusion Matrix - {trainer.model_type}"
        )
        
        # Save ROC curve plot
        roc_filename = f"{model_version}_roc_curve.png"
        roc_path = PLOTS_DIR / roc_filename
        plot_roc_curve(
            y_test, y_pred_proba, roc_path,
            title=f"ROC Curve - {trainer.model_type}"
        )
    
    return evaluation


def print_evaluation_summary(evaluation: dict):
    """
    Print a formatted evaluation summary.
    
    Args:
        evaluation: Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model Version: {evaluation['model_version']}")
    print(f"Model Type: {evaluation['model_type']}")
    print(f"Evaluated: {evaluation['evaluated_at']}")
    print(f"Test Samples: {evaluation['dataset_size']}")
    print(f"\n{'Metrics':-^60}")
    
    metrics = evaluation['metrics']
    for metric, value in metrics.items():
        print(f"  {metric.upper():15s}: {value:.4f} ({value*100:.2f}%)")
    
    print(f"{'='*60}\n")
