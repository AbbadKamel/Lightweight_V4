"""
Debug Z-Scores
==============
Inspect Z-scores of specific features during an attack to see if they are being detected.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIGS, get_model_path, ATTACKS_DIR, THRESHOLDS_DIR,
    NUM_FEATURES, FEATURE_NAMES, CRITICAL_FEATURE_INDICES
)

def debug_zscores(window_size=50, sampling_period=1):
    print(f"Debugging Z-scores for {window_size}s_{sampling_period}s...")
    
    # Load model
    model_path = get_model_path(window_size, sampling_period)
    model = load_model(model_path, compile=False)
    
    # Load stats
    stats_file = os.path.join(THRESHOLDS_DIR, "feature_stats.json")
    with open(stats_file, 'r') as f:
        all_stats = json.load(f)
    config_name = f"{window_size}s_{sampling_period}s"
    stats = all_stats[config_name]
    mean_error = np.array(stats['mean']).reshape(1, 1, NUM_FEATURES, 1)
    std_error = np.array(stats['std']).reshape(1, 1, NUM_FEATURES, 1)
    
    # Load Attack Data (Scenario A - Aggressive)
    scenario_dir = os.path.join(ATTACKS_DIR, "scenario_A_aggressive")
    combined_path = os.path.join(scenario_dir, f"{config_name}_combined.npy")
    labels_path = os.path.join(scenario_dir, f"{config_name}_labels.npy")
    
    data = np.load(combined_path)
    labels = np.load(labels_path)
    
    # Find an attack sample
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0:
        print("No attack samples found.")
        return
    
    # Pick the first attack sample
    idx = attack_indices[0]
    sample = data[idx:idx+1]
    
    # Predict
    reconstructed = model.predict(sample, verbose=0)
    sq_error = (sample - reconstructed) ** 2
    z_scores = (sq_error - mean_error) / std_error
    
    print(f"\nAnalysis of Attack Sample #{idx}")
    print("-" * 50)
    
    # Check Z-scores for critical features
    # We take the max Z-score over the time window for each feature
    max_z_per_feature = np.max(z_scores[0, :, :, 0], axis=0)
    
    print(f"{'Feature':<25} {'Max Z-Score':<12} {'Mean Error':<12} {'Std Error':<12}")
    print("-" * 65)
    
    # Sort by Max Z-Score descending
    sorted_indices = np.argsort(max_z_per_feature)[::-1]
    
    for i in sorted_indices[:10]:  # Top 10 features
        name = FEATURE_NAMES[i]
        z = max_z_per_feature[i]
        m = mean_error[0, 0, i, 0]
        s = std_error[0, 0, i, 0]
        print(f"{name:<25} {z:<12.2f} {m:<12.6f} {s:<12.6f}")
        
    print("-" * 65)
    
    # Calculate aggregated score (99th percentile)
    aggregated_score = np.percentile(z_scores, 99)
    print(f"Aggregated Score (99th percentile): {aggregated_score:.2f}")
    
    # Check specific critical features
    print("\nCritical Features Check:")
    # CRITICAL_FEATURE_INDICES is a list of indices
    for i in CRITICAL_FEATURE_INDICES:
        name = FEATURE_NAMES[i]
        z = max_z_per_feature[i]
        print(f"{name:<25} {z:.2f}")

if __name__ == "__main__":
    debug_zscores(50, 1)
