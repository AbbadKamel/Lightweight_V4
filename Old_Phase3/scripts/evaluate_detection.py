"""
Phase 3 - Step 3: Evaluate Detection Performance
=================================================
Evaluate each model's ability to detect attacks.

For each model:
1. Load thresholds and attack data
2. Calculate reconstruction error for attacks
3. Apply threshold to get predictions
4. Calculate metrics (Accuracy, Precision, Recall, F1, AUC)

Usage:
    python evaluate_detection.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)

from config import (
    MODEL_CONFIGS, ATTACK_TYPES, THRESHOLD_PERCENTILES, DEFAULT_THRESHOLD_PERCENTILE,
    NUM_FEATURES, RESULTS_DIR, ERROR_AGGREGATION_PERCENTILE, CRITICAL_FEATURE_INDICES,
    get_model_path, get_test_data_path, get_threshold_path, get_attack_data_path,
    ensure_dirs
)


def load_thresholds(window_size: int, sampling_period: int) -> Dict:
    """Load thresholds for a model."""
    threshold_path = get_threshold_path(window_size, sampling_period)
    with open(threshold_path, 'r') as f:
        return json.load(f)


def load_attack_data(attack_type: str, window_size: int, sampling_period: int) -> np.ndarray:
    """Load attack data for evaluation."""
    attack_path = get_attack_data_path(attack_type, window_size, sampling_period)
    if os.path.exists(attack_path):
        return np.load(attack_path, allow_pickle=True)
    return None


def load_normal_test_data(window_size: int, sampling_period: int) -> np.ndarray:
    """Load normal test data for evaluation."""
    data_path = get_test_data_path(window_size, sampling_period)
    data = np.load(data_path, allow_pickle=True)
    signals = data[:, :, 1:].astype(np.float32)
    return signals.reshape(-1, signals.shape[1], NUM_FEATURES, 1)


def calculate_reconstruction_errors(model, data: np.ndarray) -> np.ndarray:
    """Calculate reconstruction error per sample using high-percentile MSE and critical-signal focus."""
    reconstructed = model.predict(data, verbose=0)
    sq_error = (data - reconstructed) ** 2
    global_score = np.percentile(
        sq_error,
        ERROR_AGGREGATION_PERCENTILE,
        axis=(1, 2, 3)
    )
    if CRITICAL_FEATURE_INDICES:
        critical_sq = sq_error[:, :, CRITICAL_FEATURE_INDICES, :]
        critical_score = np.percentile(
            critical_sq,
            ERROR_AGGREGATION_PERCENTILE,
            axis=(1, 2, 3)
        )
        return np.maximum(global_score, critical_score)
    return global_score


def evaluate_at_threshold(
    normal_errors: np.ndarray,
    attack_errors: np.ndarray,
    threshold: float
) -> Dict:
    """
    Evaluate detection performance at a specific threshold.
    
    Args:
        normal_errors: Reconstruction errors for normal data
        attack_errors: Reconstruction errors for attack data
        threshold: Detection threshold
    
    Returns:
        Dictionary with metrics
    """
    # Predictions: error > threshold = attack (1), else normal (0)
    normal_preds = (normal_errors > threshold).astype(int)
    attack_preds = (attack_errors > threshold).astype(int)
    
    # Create labels
    normal_labels = np.zeros(len(normal_errors), dtype=int)
    attack_labels = np.ones(len(attack_errors), dtype=int)
    
    # Combine
    all_preds = np.concatenate([normal_preds, attack_preds])
    all_labels = np.concatenate([normal_labels, attack_labels])
    all_errors = np.concatenate([normal_errors, attack_errors])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # Derived metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # AUC-ROC (using raw errors as scores)
    try:
        auc_roc = roc_auc_score(all_labels, all_errors)
    except:
        auc_roc = 0.5
    
    # Average Precision (AUC-PR)
    try:
        auc_pr = average_precision_score(all_labels, all_errors)
    except:
        auc_pr = 0.5
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'n_normal': len(normal_errors),
        'n_attack': len(attack_errors)
    }


def evaluate_model(
    window_size: int,
    sampling_period: int
) -> Dict:
    """
    Evaluate a single model on all attack types.
    
    Args:
        window_size: Window size in seconds
        sampling_period: Sampling period in seconds
    
    Returns:
        Evaluation results dictionary
    """
    config_name = f"{window_size}s_{sampling_period}s"
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print("=" * 60)
    
    # 1. Load model
    model_path = get_model_path(window_size, sampling_period)
    print(f"  Loading model: {model_path}")
    model = load_model(model_path, compile=False)
    
    # 2. Load thresholds
    thresholds_data = load_thresholds(window_size, sampling_period)
    thresholds = thresholds_data['thresholds']
    
    # 3. Load and evaluate normal test data
    print("  Evaluating on normal data...")
    normal_data = load_normal_test_data(window_size, sampling_period)
    normal_errors = calculate_reconstruction_errors(model, normal_data)
    print(f"    Normal samples: {len(normal_errors)}")
    print(f"    Error range: [{normal_errors.min():.6f}, {normal_errors.max():.6f}]")
    
    # 4. Evaluate each attack type
    results = {
        'config': config_name,
        'window_size': window_size,
        'sampling_period': sampling_period,
        'thresholds': thresholds,
        'normal_error_stats': {
            'min': float(normal_errors.min()),
            'max': float(normal_errors.max()),
            'mean': float(normal_errors.mean()),
            'std': float(normal_errors.std())
        },
        'attack_results': {},
        'overall_results': {}
    }
    
    all_attack_errors = []
    
    for attack_type in ATTACK_TYPES:
        print(f"\n  Evaluating {attack_type} attacks...")
        
        # Load attack data
        attack_data = load_attack_data(attack_type, window_size, sampling_period)
        if attack_data is None:
            print(f"    No attack data found, skipping...")
            continue
        
        # Calculate errors
        attack_errors = calculate_reconstruction_errors(model, attack_data)
        all_attack_errors.extend(attack_errors)
        
        print(f"    Attack samples: {len(attack_errors)}")
        print(f"    Error range: [{attack_errors.min():.6f}, {attack_errors.max():.6f}]")
        
        # Evaluate at different thresholds
        attack_results = {
            'n_samples': len(attack_errors),
            'error_stats': {
                'min': float(attack_errors.min()),
                'max': float(attack_errors.max()),
                'mean': float(attack_errors.mean()),
                'std': float(attack_errors.std())
            },
            'threshold_results': {}
        }
        
        for percentile in [90, 95, 99]:
            threshold = thresholds.get(str(percentile), thresholds.get(percentile))
            if threshold is None:
                continue
            
            metrics = evaluate_at_threshold(normal_errors, attack_errors, threshold)
            attack_results['threshold_results'][percentile] = metrics
            
            print(f"    @ {percentile}th: Acc={metrics['accuracy']:.3f}, "
                  f"Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}, "
                  f"F1={metrics['f1_score']:.3f}")
        
        results['attack_results'][attack_type] = attack_results
    
    # 5. Overall evaluation (all attacks combined)
    if all_attack_errors:
        print(f"\n  Overall evaluation (all attacks combined)...")
        all_attack_errors = np.array(all_attack_errors)
        
        overall_results = {
            'n_attack_samples': len(all_attack_errors),
            'threshold_results': {}
        }
        
        for percentile in THRESHOLD_PERCENTILES:
            threshold = thresholds.get(str(percentile), thresholds.get(percentile))
            if threshold is None:
                continue
            
            metrics = evaluate_at_threshold(normal_errors, all_attack_errors, threshold)
            overall_results['threshold_results'][percentile] = metrics
        
        # Best threshold (by F1)
        best_percentile = max(
            overall_results['threshold_results'].keys(),
            key=lambda p: overall_results['threshold_results'][p]['f1_score']
        )
        overall_results['best_percentile'] = best_percentile
        overall_results['best_metrics'] = overall_results['threshold_results'][best_percentile]
        
        print(f"    Best threshold: {best_percentile}th percentile")
        best = overall_results['best_metrics']
        print(f"    Accuracy: {best['accuracy']:.4f}")
        print(f"    Precision: {best['precision']:.4f}")
        print(f"    Recall: {best['recall']:.4f}")
        print(f"    F1 Score: {best['f1_score']:.4f}")
        print(f"    AUC-ROC: {best['auc_roc']:.4f}")
        
        results['overall_results'] = overall_results
    
    # 6. Save results
    results_path = os.path.join(RESULTS_DIR, f"{config_name}_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    
    return results


def main():
    """Evaluate all models."""
    print("=" * 80)
    print("PHASE 3 - STEP 3: EVALUATE DETECTION PERFORMANCE")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Ensure output directories exist
    ensure_dirs()
    
    # Evaluate all models
    all_results = {}
    for window_size, sampling_period in MODEL_CONFIGS:
        try:
            results = evaluate_model(window_size, sampling_period)
            config_name = f"{window_size}s_{sampling_period}s"
            
            if 'overall_results' in results and 'best_metrics' in results['overall_results']:
                best = results['overall_results']['best_metrics']
                all_results[config_name] = {
                    'accuracy': best['accuracy'],
                    'precision': best['precision'],
                    'recall': best['recall'],
                    'f1_score': best['f1_score'],
                    'auc_roc': best['auc_roc'],
                    'best_percentile': results['overall_results']['best_percentile']
                }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Config':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Best%':>8}")
    print("-" * 68)
    
    for config, metrics in sorted(all_results.items()):
        print(f"{config:<12} {metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1_score']:>8.4f} "
              f"{metrics['auc_roc']:>8.4f} {metrics['best_percentile']:>8}")
    
    # Find best model
    if all_results:
        best_model = max(all_results.keys(), key=lambda k: all_results[k]['f1_score'])
        print(f"\n🏆 Best model: {best_model} (F1={all_results[best_model]['f1_score']:.4f})")
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_results': all_results,
            'best_model': best_model if all_results else None
        }, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    print(f"Completed: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
