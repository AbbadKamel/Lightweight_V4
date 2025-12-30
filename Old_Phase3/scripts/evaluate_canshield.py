
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

# New CANShield Attack Types
ATTACK_TYPES = ['plateau', 'continuous', 'playback', 'suppress']

def evaluate_canshield(scenario_name):
    print("="*80)
    print(f"EVALUATING CANSHIELD ATTACKS FOR SCENARIO: {scenario_name}")
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
    
    # Hardcoded from Phase1/scripts/config.py
    TIME_STEPS = [50, 75, 100]
    SAMPLING_PERIODS = [1, 5, 10]
    
    for window_size in TIME_STEPS:
        for sampling_period in SAMPLING_PERIODS:
            config_name = f"{window_size}s_{sampling_period}s"
            print(f"\nEvaluating {config_name}...")
            
            # Load Model
            model_path = os.path.join(PHASE2_MODELS_DIR, f"{config_name}.h5")
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            model = load_model(model_path, compile=False)
            
            # Load Thresholds
            thresh_path = os.path.join(PHASE3_THRESHOLDS_DIR, f"{config_name}_thresholds.json")
            if not os.path.exists(thresh_path):
                print(f"Thresholds not found: {thresh_path}")
                continue
                
            with open(thresh_path, 'r') as f:
                thresh_data = json.load(f)
                
            threshold = thresh_data['threshold']
            
            print(f"  Threshold: {threshold:.6f}")
            
            # 1. Evaluate on Normal Data (Test Set) to get TN and FP
            normal_data_path = os.path.join(project_root, 'Phase1', f'data_{scenario_name}', f"{window_size}s_window", f"sampling_{sampling_period}s", "test.npy")
            if os.path.exists(normal_data_path):
                X_test = np.load(normal_data_path, allow_pickle=True)
                # Ensure float32
                X_test = X_test.astype(np.float32)
                # Reshape if needed (samples, time_steps, features, 1)
                if len(X_test.shape) == 3:
                    X_test = X_test.reshape(X_test.shape[0], window_size, NUM_FEATURES, 1)
                
                rec_normal = model.predict(X_test, verbose=0)
                mse_normal = np.mean(np.square(X_test - rec_normal), axis=(1, 2, 3))
                
                # Normal data should be 0 (Normal)
                # If mse > threshold, it's a False Positive (1)
                fp_count = np.sum((mse_normal > threshold).astype(int))
                tn_count = len(mse_normal) - fp_count
                
                print(f"  Normal Test Set: TN={tn_count}, FP={fp_count} (False Alarm Rate: {fp_count/len(mse_normal):.2%})")
            else:
                print("  Normal test set not found, assuming 0 FP/TN for calculation (Warning!)")
                fp_count = 0
                tn_count = 0

            results = {}
            
            for attack_type in ATTACK_TYPES:
                print(f"  Attack Type: {attack_type}")
                
                # Load Data
                attack_dir = os.path.join(PHASE3_ATTACKS_DIR, f"canshield_{attack_type}")
                data_path = os.path.join(attack_dir, f"{config_name}.npy")
                
                if not os.path.exists(data_path):
                    print(f"Data not found: {data_path}")
                    continue
                    
                # Load attacked data (all are attacks, so label=1)
                attack_data = np.load(data_path)
                # Reshape to match model input: (samples, time_steps, features, 1)
                attack_data = attack_data.reshape(attack_data.shape[0], window_size, NUM_FEATURES, 1)
                
                # Predict
                reconstructed = model.predict(attack_data, verbose=0)
                
                # Calculate MSE
                mse = np.mean(np.square(attack_data - reconstructed), axis=(1, 2, 3))
                
                # Predictions (1 = Attack, 0 = Normal)
                preds = (mse > threshold).astype(int)
                
                # Metrics
                tp = np.sum(preds)
                fn = len(preds) - tp
                
                # Calculate full metrics
                # Accuracy = (TP + TN) / (TP + TN + FP + FN)
                total_samples = tp + tn_count + fp_count + fn
                accuracy = (tp + tn_count) / total_samples if total_samples > 0 else 0
                
                # Precision = TP / (TP + FP)
                precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0
                
                # Recall = TP / (TP + FN)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # F1 Score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"    TP={tp}, FN={fn}, FP={fp_count}, TN={tn_count}")
                print(f"    Accuracy: {accuracy:.2%}, F1: {f1:.2%}, Recall: {recall:.2%}")
                
                results[attack_type] = {
                    "tp": int(tp),
                    "fn": int(fn),
                    "fp": int(fp_count),
                    "tn": int(tn_count),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall), # Same as detection_rate
                    "f1_score": float(f1),
                    "detection_rate": float(recall) # Keep for backward compatibility
                }
                
            all_results[config_name] = results

    # Save Results
    save_path = os.path.join(PHASE3_RESULTS_DIR, "canshield_evaluation_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved results to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        evaluate_canshield(scenario)
    else:
        print("Please provide scenario name")
