
"""
Analyze Normal Z-Scores
=======================
Analyze the distribution of Max Z-scores in the normal test data.
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
    MODEL_CONFIGS, get_model_path, get_test_data_path, THRESHOLDS_DIR,
    NUM_FEATURES, FEATURE_NAMES
)

def analyze_normal_zscores(window_size=50, sampling_period=1):
    print(f"Analyzing Normal Z-scores for {window_size}s_{sampling_period}s...")
    
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
    
    # Load Normal Test Data
    data_path = get_test_data_path(window_size, sampling_period)
    data = np.load(data_path, allow_pickle=True)
    signals = data[:, :, 1:].astype(np.float32)
    cnn_input = signals.reshape(-1, signals.shape[1], NUM_FEATURES, 1)
    
    # Predict
    reconstructed = model.predict(cnn_input, verbose=0)
    sq_error = (cnn_input - reconstructed) ** 2
    z_scores = (sq_error - mean_error) / std_error
    
    # Calculate Max Z-score per sample
    max_z_per_sample = np.max(z_scores, axis=(1, 2, 3))
    
    print(f"\nNormal Data Max Z-Score Statistics:")
    print(f"  Count: {len(max_z_per_sample)}")
    print(f"  Min: {np.min(max_z_per_sample):.2f}")
    print(f"  Max: {np.max(max_z_per_sample):.2f}")
    print(f"  Mean: {np.mean(max_z_per_sample):.2f}")
    print(f"  Median: {np.median(max_z_per_sample):.2f}")
    print(f"  90th Percentile: {np.percentile(max_z_per_sample, 90):.2f}")
    print(f"  95th Percentile: {np.percentile(max_z_per_sample, 95):.2f}")
    print(f"  99th Percentile: {np.percentile(max_z_per_sample, 99):.2f}")
    
    # Identify the feature causing the max Z-score for the worst outliers
    print("\nTop 5 Worst Outliers in Normal Data:")
    sorted_indices = np.argsort(max_z_per_sample)[::-1]
    for i in sorted_indices[:5]:
        idx = i
        max_z = max_z_per_sample[idx]
        
        # Find which feature and time step caused this max Z
        sample_z = z_scores[idx] # (50, 60, 1)
        flat_idx = np.argmax(sample_z)
        time_idx, feat_idx, _ = np.unravel_index(flat_idx, sample_z.shape)
        feat_name = FEATURE_NAMES[feat_idx]
        
        print(f"  Sample #{idx}: Max Z={max_z:.2f} (Feature: {feat_name}, Time: {time_idx})")

if __name__ == "__main__":
    analyze_normal_zscores(50, 1)
