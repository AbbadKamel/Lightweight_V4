"""
Phase 3 - Step 0: Calculate Feature Error Statistics
====================================================
Calculate Mean and Std Dev of reconstruction error for EACH feature.
This is used for Z-score normalization during detection.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIGS, get_model_path, get_test_data_path, 
    ensure_dirs, THRESHOLDS_DIR, NUM_FEATURES, FEATURE_NAMES
)

def calculate_stats():
    print("Calculating feature error statistics (Mean/Std)...")
    ensure_dirs()
    
    stats_file = os.path.join(THRESHOLDS_DIR, "feature_stats.json")
    all_stats = {}
    
    for window_size, sampling_period in MODEL_CONFIGS:
        config_name = f"{window_size}s_{sampling_period}s"
        print(f"\nProcessing {config_name}...")
        
        model_path = get_model_path(window_size, sampling_period)
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
            
        model = load_model(model_path, compile=False)
        data_path = get_test_data_path(window_size, sampling_period)
        if not os.path.exists(data_path):
            print(f"  Data not found: {data_path}")
            continue

        data = np.load(data_path, allow_pickle=True)
        signals = data[:, :, 1:].astype(np.float32)
        cnn_input = signals.reshape(-1, signals.shape[1], NUM_FEATURES, 1)
        
        reconstructed = model.predict(cnn_input, verbose=0)
        sq_error = (cnn_input - reconstructed) ** 2
        
        # Calculate Mean and Std per Feature
        # Shape: (features,)
        mean_error = np.mean(sq_error, axis=(0, 1, 3))
        std_error = np.std(sq_error, axis=(0, 1, 3))
        
        # Avoid zero division
        std_error = np.where(std_error < 1e-6, 1e-6, std_error)
        
        all_stats[config_name] = {
            'mean': mean_error.tolist(),
            'std': std_error.tolist(),
            'feature_names': FEATURE_NAMES
        }
        
        print(f"  Stats calculated. Mean range: [{mean_error.min():.6f}, {mean_error.max():.6f}]")
        
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    print(f"\nSaved stats to: {stats_file}")

if __name__ == "__main__":
    calculate_stats()
