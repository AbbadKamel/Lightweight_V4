
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    get_model_path, ATTACKS_DIR, FEATURE_NAMES
)

def debug_feature_stats(window_size=50, sampling_period=1):
    print(f"\n=== Analyzing Feature Error Statistics {window_size}s_{sampling_period}s ===")
    
    # Load Model
    model_path = get_model_path(window_size, sampling_period)
    model = load_model(model_path, compile=False)
    
    # Load Normal Data (from Aggressive Combined, label 0)
    scenario_dir = os.path.join(ATTACKS_DIR, "scenario_A_aggressive")
    combined_path = os.path.join(scenario_dir, f"{window_size}s_{sampling_period}s_combined.npy")
    labels_path = os.path.join(scenario_dir, f"{window_size}s_{sampling_period}s_labels.npy")
    
    data = np.load(combined_path)
    labels = np.load(labels_path)
    normal_data = data[labels == 0]
    
    print(f"Normal samples: {len(normal_data)}")
    
    # Predict
    reconstructed = model.predict(normal_data, verbose=0)
    sq_error = (normal_data - reconstructed) ** 2
    
    # Calculate Mean Error per feature (averaged over time and samples)
    # Shape: (samples, time, features, 1) -> (features,)
    mean_errors = np.mean(sq_error, axis=(0, 1, 3))
    std_errors = np.std(sq_error, axis=(0, 1, 3))
    max_errors = np.max(sq_error, axis=(0, 1, 3))
    
    print("\nTop 10 Features with Highest Mean Error (The 'Noisy' ones):")
    top_indices = np.argsort(mean_errors)[::-1][:10]
    for idx in top_indices:
        print(f"  {FEATURE_NAMES[idx]:<25} Mean: {mean_errors[idx]:.6f}  Std: {std_errors[idx]:.6f}  Max: {max_errors[idx]:.6f}")

    print("\nTop 10 Features with Lowest Mean Error (The 'Clean' ones):")
    bottom_indices = np.argsort(mean_errors)[:10]
    for idx in bottom_indices:
        print(f"  {FEATURE_NAMES[idx]:<25} Mean: {mean_errors[idx]:.6f}  Std: {std_errors[idx]:.6f}  Max: {max_errors[idx]:.6f}")

if __name__ == "__main__":
    debug_feature_stats(50, 1)
