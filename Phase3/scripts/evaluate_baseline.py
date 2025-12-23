
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================
# Add Phase 1 scripts to path to import shared config
PHASE1_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Phase1/scripts'))
sys.path.insert(0, PHASE1_SCRIPTS_DIR)
import config

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Phase1/data')
PHASE2_MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Phase2/models')
THRESHOLDS_DIR = os.path.join(BASE_DIR, 'thresholds')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_data(window_size, sampling_period):
    """Load Test data (Normal) for specific config."""
    data_path = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s", "test.npy")
    print(f"  Loading Normal Test Data: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    # Reshape: (samples, time, features) -> (samples, time, features, 1)
    signals = data[:, :, 1:].astype(np.float32)
    return signals.reshape(signals.shape[0], signals.shape[1], signals.shape[2], 1)

def load_threshold(window_size, sampling_period):
    """Load the specific threshold calculated in Phase 2."""
    config_name = f"{window_size}s_{sampling_period}s"
    path = os.path.join(THRESHOLDS_DIR, f"{config_name}_thresholds.json")
    
    if not os.path.exists(path):
        # Try checking strict scenario folders or other locations if needed
        # For now, if missing, we skip
        print(f"  ⚠️ Warning: Threshold file not found: {path}")
        return None
        
    with open(path, 'r') as f:
        data = json.load(f)
        return float(data['threshold'])

def evaluate_baseline():
    print("="*60)
    print("PHASE 3: BASELINE EVALUATION (Normal Test Data Only)")
    print("="*60)
    
    results = {}
    
    for window_size in config.WINDOW_SIZES:
        for sampling_period in config.SAMPLING_PERIODS:
            config_name = f"{window_size}s_{sampling_period}s"
            print(f"\nEvaluating: {config_name}")
            
            # 1. Load Model
            model_path = os.path.join(PHASE2_MODELS_DIR, f"{config_name}.h5")
            if not os.path.exists(model_path):
                print(f"  Skipping (Model not found)")
                continue
                
            model = load_model(model_path, compile=False)
            
            # 2. Load Threshold (Phase 2)
            threshold = load_threshold(window_size, sampling_period)
            if threshold is None:
                continue
            print(f"  Threshold (from Phase 2): {threshold:.6f}")
            
            # 3. Load Normal Test Data
            X_test_normal = load_data(window_size, sampling_period)
            
            # 4. Predict & Calculate Error
            reconstructions = model.predict(X_test_normal, verbose=0)
            mse = np.mean(np.square(X_test_normal - reconstructions), axis=(1, 2, 3))
            
            # 5. False Positive Check
            # Since this is ALL Normal data, ANY error > threshold is a False Positive
            false_positives = np.sum(mse > threshold)
            fp_rate = false_positives / len(mse)
            
            print(f"  Normal Samples: {len(mse)}")
            print(f"  False Positives: {false_positives}")
            print(f"  False Positive Rate (FPR): {fp_rate:.4f} ({fp_rate*100:.2f}%)")
            
            results[config_name] = {
                'threshold': threshold,
                'samples': len(mse),
                'false_positives': int(false_positives),
                'fpr': float(fp_rate)
            }

    # Save Results
    output_path = os.path.join(RESULTS_DIR, 'baseline_fpr.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print(f"Baseline (FPR) Results saved to: {output_path}")

if __name__ == "__main__":
    evaluate_baseline()
