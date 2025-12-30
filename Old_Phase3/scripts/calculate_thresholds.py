
"""
Phase 3 - Step 1: Calculate Detection Thresholds (Z-Score Based)
=================================================================
Calculate Z-score thresholds from normal test data.

For each model:
1. Load normal test data
2. Load feature statistics (mean/std of error per feature)
3. Calculate Z-score of reconstruction error for each window
4. Compute percentile thresholds (90th, 95th, 99th, etc.)
5. Save thresholds for later use in detection

Usage:
    python calculate_thresholds.py
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow.keras.models import load_model

# Local imports
from config import (
    MODEL_CONFIGS, THRESHOLD_PERCENTILES, DEFAULT_THRESHOLD_PERCENTILE,
    get_model_path, get_test_data_path, get_threshold_path,
    ensure_dirs, NUM_FEATURES, ERROR_AGGREGATION_PERCENTILE, CRITICAL_FEATURE_INDICES,
    THRESHOLDS_DIR, FEATURE_NAMES
)


def load_test_data(window_size: int, sampling_period: int) -> np.ndarray:
    """Load test data for a specific configuration."""
    data_path = get_test_data_path(window_size, sampling_period)
    print(f"  Loading: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    signals = data[:, :, 1:].astype(np.float32)
    cnn_input = signals.reshape(-1, signals.shape[1], NUM_FEATURES, 1)
    return cnn_input

def load_feature_stats(window_size: int, sampling_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load mean and std of errors for each feature."""
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
    std_error: np.ndarray
) -> np.ndarray:
    """
    Calculate Z-score of reconstruction error for each window.
    """
    # Get reconstructions
    reconstructed = model.predict(data, verbose=0)
    
    # Calculate per-pixel squared error
    sq_error = (data - reconstructed) ** 2
    
    # Normalize to Z-score
    z_scores = (sq_error - mean_error) / std_error
    
    return z_scores


def calculate_thresholds(errors: np.ndarray, percentiles: List[float]) -> Dict[float, float]:
    """Calculate threshold values at specified percentiles."""
    thresholds = {}
    for p in percentiles:
        thresholds[p] = float(np.percentile(errors, p))
    return thresholds


def process_model(window_size: int, sampling_period: int) -> Dict:
    """Calculate thresholds for a single model."""
    config_name = f"{window_size}s_{sampling_period}s"
    print(f"\n{'='*60}")
    print(f"Processing: {config_name}")
    print("=" * 60)
    
    # 1. Load model
    model_path = get_model_path(window_size, sampling_period)
    print(f"  Loading model: {model_path}")
    model = load_model(model_path, compile=False)
    
    # 2. Load test data
    test_data = load_test_data(window_size, sampling_period)
    
    # 3. Load feature stats
    mean_error, std_error = load_feature_stats(window_size, sampling_period)
    
    # 4. Calculate Z-scores
    print("  Calculating Z-scores...")
    z_scores = calculate_z_scores(model, test_data, mean_error, std_error)
    print(f"  Z-Scores: min={z_scores.min():.2f}, max={z_scores.max():.2f}, mean={z_scores.mean():.2f}")
    
    # 5. Identify and Exclude Noisy Features
    # Calculate max Z-score per feature across all samples
    # Shape: (features,)
    max_z_per_feature = np.max(z_scores, axis=(0, 1, 3))
    
    # Define noisy features as those with Max Z > 10
    noisy_indices = np.where(max_z_per_feature > 10)[0]
    noisy_features = [FEATURE_NAMES[i] for i in noisy_indices]
    print(f"  Noisy features (Max Z > 10): {noisy_features}")
    
    # Create a mask to exclude noisy features
    mask = np.ones(NUM_FEATURES, dtype=bool)
    mask[noisy_indices] = False
    
    # Apply mask for aggregation
    # z_scores shape: (samples, time, features, 1)
    z_scores_filtered = z_scores[:, :, mask, :]
    
    # Aggregate: Take the MAX Z-score across time and VALID features
    global_score = np.max(
        z_scores_filtered, 
        axis=(1, 2, 3)
    )
    
    # 6. Calculate thresholds
    print("  Calculating thresholds...")
    thresholds = calculate_thresholds(global_score, THRESHOLD_PERCENTILES)
    
    # Print threshold summary
    print(f"\n  Threshold Summary (Z-Score, Filtered):")
    for p in [90, 95, 99]:
        if p in thresholds:
            print(f"    {p}th percentile: {thresholds[p]:.2f}")
    
    # 7. Prepare results
    result = {
        'config': config_name,
        'window_size': window_size,
        'sampling_period': sampling_period,
        'n_test_samples': len(global_score),
        'noisy_features': noisy_features,  # Save this!
        'error_statistics': {
            'min': float(global_score.min()),
            'max': float(global_score.max()),
            'mean': float(global_score.mean()),
            'std': float(global_score.std()),
            'median': float(np.median(global_score))
        },
        'thresholds': thresholds,
        'default_threshold': thresholds[DEFAULT_THRESHOLD_PERCENTILE],
        'all_errors': global_score.tolist()
    }
    
    # 7. Save thresholds
    threshold_path = get_threshold_path(window_size, sampling_period)
    with open(threshold_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {threshold_path}")
    
    return result


def main():
    """Calculate thresholds for all models."""
    print("=" * 80)
    print("PHASE 3 - STEP 1: CALCULATE DETECTION THRESHOLDS (Z-SCORE)")
    print("=" * 80)
    
    ensure_dirs()
    
    all_results = {}
    for window_size, sampling_period in MODEL_CONFIGS:
        try:
            result = process_model(window_size, sampling_period)
            config_name = f"{window_size}s_{sampling_period}s"
            all_results[config_name] = {
                'n_samples': result['n_test_samples'],
                'mean_z': result['error_statistics']['mean'],
                'threshold_95': result['thresholds'].get(95, None),
                'threshold_99': result['thresholds'].get(99, None)
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("THRESHOLD CALCULATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Config':<15} {'Samples':>8} {'Mean Z':>12} {'Thresh 95%':>12} {'Thresh 99%':>12}")
    print("-" * 65)
    
    for config, stats in all_results.items():
        print(f"{config:<15} {stats['n_samples']:>8} {stats['mean_z']:>12.2f} "
              f"{stats['threshold_95']:>12.2f} {stats['threshold_99']:>12.2f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
