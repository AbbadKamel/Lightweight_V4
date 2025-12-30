"""
Quick Analysis: Find Optimal Threshold
======================================
Test different thresholds to find the best detection performance.
"""

import os
import sys
import json
import numpy as np
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from config import get_model_path, get_threshold_path, ATTACKS_DIR, NUM_FEATURES, ERROR_AGGREGATION_PERCENTILE, CRITICAL_FEATURE_INDICES


def load_data_and_model(config='50s_1s', scenario='A_aggressive'):
    """Load model, threshold data, and attack scenario."""
    ws, sp = int(config.split('s_')[0]), int(config.split('_')[1].replace('s', ''))
    
    # Load model
    model = load_model(get_model_path(ws, sp), compile=False)
    
    # Load threshold data (to get error statistics)
    with open(get_threshold_path(ws, sp), 'r') as f:
        thresh_data = json.load(f)
    
    # Load scenario data
    scenario_dir = os.path.join(ATTACKS_DIR, f"scenario_{scenario}")
    data = np.load(os.path.join(scenario_dir, f"{config}_combined.npy"))
    labels = np.load(os.path.join(scenario_dir, f"{config}_labels.npy"))
    
    return model, thresh_data, data, labels


def find_optimal_threshold(model, data, labels):
    """Find threshold that maximizes F1 score."""
    # Calculate errors
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
        errors = np.maximum(global_score, critical_score)
    else:
        errors = global_score
    
    print(f"\nError Statistics:")
    print(f"  Normal errors: min={errors[labels==0].min():.4f}, max={errors[labels==0].max():.4f}, mean={errors[labels==0].mean():.4f}")
    print(f"  Attack errors: min={errors[labels==1].min():.4f}, max={errors[labels==1].max():.4f}, mean={errors[labels==1].mean():.4f}")
    
    # Test different thresholds
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95, 99]
    print(f"\nThreshold Analysis:")
    print(f"{'Percentile':>12} {'Threshold':>12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    
    best_f1 = 0
    best_percentile = 0
    best_threshold = 0
    
    # Use normal data errors for percentile calculation
    normal_errors = errors[labels == 0]
    
    for p in percentiles:
        threshold = np.percentile(normal_errors, p)
        predictions = (errors > threshold).astype(int)
        
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        print(f"{p:>12}th {threshold:>12.6f} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_percentile = p
            best_threshold = threshold
    
    print(f"\n✓ OPTIMAL: {best_percentile}th percentile (threshold={best_threshold:.6f}) → F1={best_f1:.3f}")
    
    return best_percentile, best_threshold, best_f1


def main():
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 70)
    
    configs = ['50s_1s', '75s_1s', '100s_1s']
    scenarios = ['A_aggressive', 'B_stealthy']
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"MODEL: {config}")
        print("=" * 70)
        
        results[config] = {}
        
        for scenario in scenarios:
            print(f"\n--- Scenario: {scenario} ---")
            try:
                model, thresh_data, data, labels = load_data_and_model(config, scenario)
                percentile, threshold, f1 = find_optimal_threshold(model, data, labels)
                results[config][scenario] = {
                    'optimal_percentile': percentile,
                    'optimal_threshold': threshold,
                    'best_f1': f1
                }
            except Exception as e:
                print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: OPTIMAL THRESHOLDS")
    print("=" * 70)
    print(f"\n{'Config':<12} {'Scenario':<15} {'Percentile':>12} {'Best F1':>10}")
    print("-" * 55)
    
    for config in configs:
        for scenario in scenarios:
            if scenario in results.get(config, {}):
                r = results[config][scenario]
                scenario_short = 'Aggressive' if 'A' in scenario else 'Stealthy'
                print(f"{config:<12} {scenario_short:<15} {r['optimal_percentile']:>12}th {r['best_f1']:>10.3f}")


if __name__ == "__main__":
    main()
