"""
Phase 3 - Step 3: Evaluate Detection Performance (Z-Score)
==========================================================
Evaluate CNN autoencoder detection on both attack scenarios using Z-scores.

Usage:
    python evaluate_dual_scenarios.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import (
    MODEL_CONFIGS, THRESHOLD_PERCENTILES, DEFAULT_THRESHOLD_PERCENTILE,
    get_model_path, get_threshold_path, ATTACKS_DIR, RESULTS_DIR, FIGURES_DIR,
    NUM_FEATURES, ensure_dirs, THRESHOLDS_DIR
)

SCENARIOS = ['A_aggressive', 'B_stealthy']

def load_scenario_data(scenario: str, window_size: int, sampling_period: int) -> Tuple[np.ndarray, np.ndarray]:
    config_name = f"{window_size}s_{sampling_period}s"
    scenario_dir = os.path.join(ATTACKS_DIR, f"scenario_{scenario}")
    combined_path = os.path.join(scenario_dir, f"{config_name}_combined.npy")
    labels_path = os.path.join(scenario_dir, f"{config_name}_labels.npy")
    data = np.load(combined_path)
    labels = np.load(labels_path)
    return data, labels

def load_model_and_threshold(window_size: int, sampling_period: int) -> Tuple[tf.keras.Model, Dict, List[str]]:
    model_path = get_model_path(window_size, sampling_period)
    model = load_model(model_path, compile=False)
    threshold_path = get_threshold_path(window_size, sampling_period)
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
    return model, threshold_data['thresholds'], threshold_data.get('noisy_features', [])

def load_feature_stats(window_size: int, sampling_period: int) -> Tuple[np.ndarray, np.ndarray]:
    stats_file = os.path.join(THRESHOLDS_DIR, "feature_stats.json")
    with open(stats_file, 'r') as f:
        all_stats = json.load(f)
    config_name = f"{window_size}s_{sampling_period}s"
    stats = all_stats[config_name]
    mean = np.array(stats['mean']).reshape(1, 1, NUM_FEATURES, 1)
    std = np.array(stats['std']).reshape(1, 1, NUM_FEATURES, 1)
    return mean, std

def calculate_z_scores(
    model: tf.keras.Model,
    data: np.ndarray,
    mean_error: np.ndarray,
    std_error: np.ndarray,
    noisy_features: List[str] = []
) -> np.ndarray:
    reconstructed = model.predict(data, verbose=0)
    sq_error = (data - reconstructed) ** 2
    
    z_scores = (sq_error - mean_error) / std_error
    
    # Create mask for valid features
    mask = np.ones(NUM_FEATURES, dtype=bool)
    if noisy_features:
        # Need to map feature names to indices
        # Assuming FEATURE_NAMES is available globally or imported
        from config import FEATURE_NAMES
        noisy_indices = [i for i, name in enumerate(FEATURE_NAMES) if name in noisy_features]
        mask[noisy_indices] = False
        
    # Apply mask
    z_scores_filtered = z_scores[:, :, mask, :]
    
    # Aggregate: Take the MAX Z-score across time and VALID features
    global_score = np.max(
        z_scores_filtered,
        axis=(1, 2, 3)
    )
    return global_score

def evaluate_detection(errors: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    predictions = (errors > threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    try:
        auc_roc = roc_auc_score(labels, errors)
    except:
        auc_roc = 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'threshold': float(threshold)
    }

def evaluate_model(window_size: int, sampling_period: int) -> Dict:
    config_name = f"{window_size}s_{sampling_period}s"
    print(f"\nEvaluating {config_name}...")
    
    model, thresholds, noisy_features = load_model_and_threshold(window_size, sampling_period)
    mean_error, std_error = load_feature_stats(window_size, sampling_period)
    
    print(f"  Excluding {len(noisy_features)} noisy features.")
    
    # Use 95th percentile threshold by default
    threshold = thresholds.get(str(DEFAULT_THRESHOLD_PERCENTILE))
    if threshold is None:
        threshold = list(thresholds.values())[0]
    
    results = {}
    for scenario in SCENARIOS:
        print(f"  Scenario: {scenario}")
        data, labels = load_scenario_data(scenario, window_size, sampling_period)
        z_scores = calculate_z_scores(model, data, mean_error, std_error, noisy_features)
        metrics = evaluate_detection(z_scores, labels, threshold)
        results[scenario] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1_score']:.4f}")
        print(f"    Recall:   {metrics['recall']:.4f}")
        print(f"    TP: {metrics['tp']}, FN: {metrics['fn']}")
        
    return results

def main():
    print("PHASE 3 - STEP 3: EVALUATE DETECTION (Z-SCORE)")
    ensure_dirs()
    all_results = {}
    for window_size, sampling_period in MODEL_CONFIGS:
        try:
            results = evaluate_model(window_size, sampling_period)
            config_name = f"{window_size}s_{sampling_period}s"
            all_results[config_name] = results
        except Exception as e:
            print(f"  ERROR evaluating {window_size}s_{sampling_period}s: {e}")
            import traceback
            traceback.print_exc()
            
    results_path = os.path.join(RESULTS_DIR, "detection_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (F1-Score)")
    print("=" * 80)
    print(f"{'Config':<15} {'Aggressive':<12} {'Stealthy':<12}")
    print("-" * 45)
    for config, res in all_results.items():
        f1_a = res['A_aggressive']['f1_score']
        f1_b = res['B_stealthy']['f1_score']
        print(f"{config:<15} {f1_a:<12.4f} {f1_b:<12.4f}")

if __name__ == "__main__":
    main()
