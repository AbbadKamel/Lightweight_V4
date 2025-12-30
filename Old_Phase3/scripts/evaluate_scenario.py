
import os
import sys
import json
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIGS, DEFAULT_THRESHOLD_PERCENTILE,
    ensure_dirs
)

SCENARIOS = ['aggressive', 'stealthy']

def evaluate_scenario(scenario_name):
    print("="*80)
    print(f"EVALUATING SCENARIO: {scenario_name}")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phase1_config_path = os.path.join(project_root, 'Phase1', 'results', f'config_{scenario_name}.json')
    
    with open(phase1_config_path, 'r') as f:
        scenario_config = json.load(f)
    NUM_FEATURES = scenario_config['num_features']
    
    PHASE2_MODELS_DIR = os.path.join(project_root, 'Phase2', f'models_{scenario_name}')
    PHASE3_THRESHOLDS_DIR = os.path.join(project_root, 'Phase3', f'thresholds_{scenario_name}')
    PHASE3_ATTACKS_DIR = os.path.join(project_root, 'Phase3', f'attacks_{scenario_name}')
    PHASE3_RESULTS_DIR = os.path.join(project_root, 'Phase3', f'results_{scenario_name}')
    
    if not os.path.exists(PHASE3_RESULTS_DIR):
        os.makedirs(PHASE3_RESULTS_DIR)
        
    all_results = {}
    
    for window_size, sampling_period in MODEL_CONFIGS:
        config_name = f"{window_size}s_{sampling_period}s"
        print(f"\nEvaluating {config_name}...")
        
        # Load Model
        model_path = os.path.join(PHASE2_MODELS_DIR, f"{config_name}.h5")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        model = load_model(model_path, compile=False)
        
        # Load Thresholds & Stats
        thresh_path = os.path.join(PHASE3_THRESHOLDS_DIR, f"{config_name}_thresholds.json")
        if not os.path.exists(thresh_path):
            print(f"Thresholds not found: {thresh_path}")
            continue
            
        with open(thresh_path, 'r') as f:
            thresh_data = json.load(f)
            
        threshold = thresh_data['thresholds'][str(DEFAULT_THRESHOLD_PERCENTILE)]
        mean_error = np.array(thresh_data['mean']).reshape(1, 1, NUM_FEATURES, 1)
        std_error = np.array(thresh_data['std']).reshape(1, 1, NUM_FEATURES, 1)
        noisy_features = thresh_data['noisy_features'] # Indices as strings
        noisy_indices = [int(i) for i in noisy_features]
        
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Excluding {len(noisy_indices)} noisy features")
        
        results = {}
        
        for attack_scenario in SCENARIOS:
            print(f"  Attack Scenario: {attack_scenario}")
            
            # Load Data
            attack_dir = os.path.join(PHASE3_ATTACKS_DIR, f"scenario_{attack_scenario}")
            data_path = os.path.join(attack_dir, f"{config_name}_combined.npy")
            label_path = os.path.join(attack_dir, f"{config_name}_labels.npy")
            
            if not os.path.exists(data_path):
                print(f"Data not found: {data_path}")
                continue
                
            data = np.load(data_path)
            labels = np.load(label_path)
            
            # Predict
            reconstructed = model.predict(data, verbose=0)
            sq_error = (data - reconstructed) ** 2
            
            # Z-Score
            z_scores = (sq_error - mean_error) / std_error
            
            # Mask Noisy Features
            mask = np.ones(NUM_FEATURES, dtype=bool)
            mask[noisy_indices] = False
            z_scores_filtered = z_scores[:, :, mask, :]
            
            # Aggregate (MAX)
            if z_scores_filtered.shape[2] > 0:
                scores = np.max(z_scores_filtered, axis=(1, 2, 3))
            else:
                scores = np.zeros(len(z_scores))
                
            # Detect
            predictions = (scores > threshold).astype(int)
            
            # Metrics
            acc = accuracy_score(labels, predictions)
            prec = precision_score(labels, predictions, zero_division=0)
            rec = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            
            results[attack_scenario] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
            }
            
            print(f"    F1: {f1:.4f} (TP={tp}, FN={fn})")
            
        all_results[config_name] = results
        
    # Save Results
    out_path = os.path.join(PHASE3_RESULTS_DIR, "evaluation_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        evaluate_scenario(scenario)
    else:
        print("Please provide scenario name")
