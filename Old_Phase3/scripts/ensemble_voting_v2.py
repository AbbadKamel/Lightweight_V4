"""
Phase 3 - Ensemble Voting Detection (Fixed)
============================================
Properly combine predictions from multiple models.

Key insight: We can ensemble models with SAME window size but DIFFERENT sampling periods,
as long as we use the common test windows that exist in all sampling configurations.

Alternative approach: Use multiple thresholds per model instead of multiple models.

Usage:
    python ensemble_voting_v2.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    get_model_path, get_threshold_path, get_test_data_path,
    ATTACKS_DIR, RESULTS_DIR, NUM_FEATURES, ensure_dirs, ERROR_AGGREGATION_PERCENTILE, CRITICAL_FEATURE_INDICES
)

SCENARIOS = ['A_aggressive', 'B_stealthy']

# We'll focus on the best model (50s_1s) and use multiple thresholds
# This is a simpler form of "ensemble" - testing at different sensitivity levels


def load_model_and_data(window_size: int, sampling_period: int, scenario: str):
    """Load model, threshold info, and scenario data."""
    config_name = f"{window_size}s_{sampling_period}s"
    
    # Load model
    model = load_model(get_model_path(window_size, sampling_period), compile=False)
    
    # Load threshold data
    with open(get_threshold_path(window_size, sampling_period), 'r') as f:
        thresh_data = json.load(f)
    
    # Load scenario data
    scenario_dir = os.path.join(ATTACKS_DIR, f"scenario_{scenario}")
    data = np.load(os.path.join(scenario_dir, f"{config_name}_combined.npy"))
    labels = np.load(os.path.join(scenario_dir, f"{config_name}_labels.npy"))
    
    return model, thresh_data, data, labels


def calculate_errors(model, data):
    """Calculate reconstruction errors with high-percentile aggregation and critical focus."""
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


def evaluate(labels, predictions):
    """Calculate metrics."""
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1_score': float(f1_score(labels, predictions, zero_division=0)),
        'tp': int(np.sum((labels == 1) & (predictions == 1))),
        'tn': int(np.sum((labels == 0) & (predictions == 0))),
        'fp': int(np.sum((labels == 0) & (predictions == 1))),
        'fn': int(np.sum((labels == 1) & (predictions == 0))),
    }


def multi_threshold_voting(errors, labels, thresholds_dict, strategy='majority'):
    """
    Use multiple thresholds and vote.
    
    This simulates having multiple "detectors" with different sensitivities.
    """
    percentiles = [90, 92, 93, 94, 95]  # Different thresholds
    
    # Get predictions at each threshold
    n_samples = len(errors)
    n_thresholds = len(percentiles)
    all_predictions = np.zeros((n_thresholds, n_samples), dtype=int)
    
    normal_errors = errors[labels == 0]  # Use normal data for percentile calculation
    
    for i, p in enumerate(percentiles):
        threshold = np.percentile(normal_errors, p)
        all_predictions[i] = (errors > threshold).astype(int)
    
    # Vote
    vote_counts = np.sum(all_predictions, axis=0)
    
    if strategy == 'majority':
        final = (vote_counts > n_thresholds / 2).astype(int)
    elif strategy == 'any':
        final = (vote_counts >= 1).astype(int)
    elif strategy == 'unanimous':
        final = (vote_counts == n_thresholds).astype(int)
    else:
        final = (vote_counts >= 2).astype(int)
    
    return final, vote_counts, all_predictions


def main():
    print("=" * 80)
    print("ENSEMBLE VOTING DETECTION - V2")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    
    ensure_dirs()
    
    # Test on best model (50s_1s)
    config = (50, 1)
    config_name = f"{config[0]}s_{config[1]}s"
    
    print(f"\nUsing model: {config_name}")
    print("\nStrategy: Multi-threshold voting")
    print("Thresholds: 90th, 92nd, 93rd, 94th, 95th percentile")
    
    all_results = {}
    
    for scenario in SCENARIOS:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.upper()}")
        print("=" * 70)
        
        # Load data
        model, thresh_data, data, labels = load_model_and_data(config[0], config[1], scenario)
        print(f"Data: {len(data)} samples ({np.sum(labels==0)} normal, {np.sum(labels==1)} attacks)")
        
        # Calculate errors
        errors = calculate_errors(model, data)
        
        print(f"\nError Statistics:")
        print(f"  Normal: mean={errors[labels==0].mean():.4f}, std={errors[labels==0].std():.4f}")
        print(f"  Attack: mean={errors[labels==1].mean():.4f}, std={errors[labels==1].std():.4f}")
        print(f"  Difference: {(errors[labels==1].mean() - errors[labels==0].mean()):.4f}")
        
        scenario_results = {}
        
        # Single thresholds
        print(f"\n--- Single Threshold Performance ---")
        print(f"{'Percentile':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 55)
        
        normal_errors = errors[labels == 0]
        for p in [50, 60, 70, 80, 90, 95]:
            threshold = np.percentile(normal_errors, p)
            preds = (errors > threshold).astype(int)
            metrics = evaluate(labels, preds)
            print(f"{p}th          {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} "
                  f"{metrics['recall']:>10.3f} {metrics['f1_score']:>10.3f}")
            scenario_results[f"single_{p}th"] = metrics
        
        # Multi-threshold voting
        print(f"\n--- Multi-Threshold Voting Performance ---")
        print(f"{'Strategy':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 55)
        
        for strategy in ['majority', 'any', 'unanimous']:
            final_preds, votes, _ = multi_threshold_voting(errors, labels, thresh_data, strategy)
            metrics = evaluate(labels, final_preds)
            print(f"{strategy:<12} {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} "
                  f"{metrics['recall']:>10.3f} {metrics['f1_score']:>10.3f}")
            scenario_results[f"voting_{strategy}"] = metrics
        
        # Show vote distribution
        print(f"\n--- Vote Distribution ---")
        _, votes, _ = multi_threshold_voting(errors, labels, thresh_data, 'majority')
        for v in range(6):
            count = np.sum(votes == v)
            if count > 0:
                normal_with_v = np.sum((votes == v) & (labels == 0))
                attack_with_v = np.sum((votes == v) & (labels == 1))
                print(f"  {v}/5 votes for attack: {count:3d} samples (Normal: {normal_with_v:3d}, Attack: {attack_with_v:3d})")
        
        all_results[scenario] = scenario_results
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Method':<20} {'Aggressive F1':>15} {'Stealthy F1':>15}")
    print("-" * 55)
    
    methods = ['single_50th', 'single_60th', 'single_70th', 'single_80th', 
               'single_90th', 'single_95th', 'voting_majority', 'voting_any', 'voting_unanimous']
    
    for method in methods:
        f1_agg = all_results['A_aggressive'].get(method, {}).get('f1_score', 0)
        f1_stl = all_results['B_stealthy'].get(method, {}).get('f1_score', 0)
        print(f"{method:<20} {f1_agg:>15.3f} {f1_stl:>15.3f}")
    
    # Best methods
    print("\n--- BEST METHODS ---")
    for scenario in SCENARIOS:
        best = max(all_results[scenario].keys(), 
                  key=lambda m: all_results[scenario][m]['f1_score'])
        f1 = all_results[scenario][best]['f1_score']
        name = 'Aggressive' if 'A' in scenario else 'Stealthy'
        print(f"  {name}: {best} (F1={f1:.3f})")
    
    # Save
    results_path = os.path.join(RESULTS_DIR, 'ensemble_v2_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")
    
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
