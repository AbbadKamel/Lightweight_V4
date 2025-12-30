
import os
import sys
import json
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIGS, THRESHOLD_PERCENTILES, DEFAULT_THRESHOLD_PERCENTILE,
    ensure_dirs
)

def calculate_thresholds_scenario(scenario_name):
    print("="*80)
    print(f"CALCULATING THRESHOLDS FOR SCENARIO: {scenario_name}")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phase1_config_path = os.path.join(project_root, 'Phase1', 'results', f'config_{scenario_name}.json')
    
    with open(phase1_config_path, 'r') as f:
        scenario_config = json.load(f)
    NUM_FEATURES = scenario_config['num_features']
    
    PHASE1_DATA_DIR = os.path.join(project_root, 'Phase1', f'data_{scenario_name}')
    PHASE2_MODELS_DIR = os.path.join(project_root, 'Phase2', f'models_{scenario_name}')
    PHASE3_THRESHOLDS_DIR = os.path.join(project_root, 'Phase3', f'thresholds_{scenario_name}')
    
    if not os.path.exists(PHASE3_THRESHOLDS_DIR):
        os.makedirs(PHASE3_THRESHOLDS_DIR)
        
    # Calculate Feature Stats (Mean/Std) for Z-score
    print("Calculating feature stats...")
    all_stats = {}
    
    for window_size, sampling_period in MODEL_CONFIGS:
        config_name = f"{window_size}s_{sampling_period}s"
        
        model_path = os.path.join(PHASE2_MODELS_DIR, f"{config_name}.h5")
        if not os.path.exists(model_path):
            continue
            
        model = load_model(model_path, compile=False)
        
        data_dir = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s")
        test_path = os.path.join(data_dir, "test.npy") # Use test data (normal) for thresholds
        
        data = np.load(test_path, allow_pickle=True)
        signals = data[:, :, 1:].astype(np.float32)
        cnn_input = signals.reshape(-1, signals.shape[1], NUM_FEATURES, 1)
        
        reconstructed = model.predict(cnn_input, verbose=0)
        sq_error = (cnn_input - reconstructed) ** 2
        
        mean_error = np.mean(sq_error, axis=(0, 1, 3))
        std_error = np.std(sq_error, axis=(0, 1, 3))
        std_error = np.where(std_error < 1e-6, 1e-6, std_error)
        
        all_stats[config_name] = {
            'mean': mean_error.tolist(),
            'std': std_error.tolist()
        }
        
        # Calculate Z-scores
        z_scores = (sq_error - mean_error.reshape(1,1,NUM_FEATURES,1)) / std_error.reshape(1,1,NUM_FEATURES,1)
        
        # Identify Noisy Features (Max Z > 10)
        max_z = np.max(z_scores, axis=(0, 1, 3))
        noisy_indices = np.where(max_z > 10)[0]
        noisy_features = [str(i) for i in noisy_indices] # Store indices as strings
        
        # Mask noisy features
        mask = np.ones(NUM_FEATURES, dtype=bool)
        mask[noisy_indices] = False
        z_scores_filtered = z_scores[:, :, mask, :]
        
        # Aggregate (MAX)
        if z_scores_filtered.shape[2] > 0:
            global_score = np.max(z_scores_filtered, axis=(1, 2, 3))
        else:
            global_score = np.zeros(len(z_scores))
            
        # Calculate Thresholds
        thresholds = {}
        for p in THRESHOLD_PERCENTILES:
            thresholds[p] = float(np.percentile(global_score, p))
            
        # Save
        result = {
            'thresholds': thresholds,
            'noisy_features': noisy_features,
            'mean': mean_error.tolist(),
            'std': std_error.tolist()
        }
        
        out_path = os.path.join(PHASE3_THRESHOLDS_DIR, f"{config_name}_thresholds.json")
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"Saved thresholds for {config_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        calculate_thresholds_scenario(scenario)
    else:
        print("Please provide scenario name")
