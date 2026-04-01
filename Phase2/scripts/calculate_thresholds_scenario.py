import os
import sys
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add Phase1/scripts to path for config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Phase1/scripts')))
import config

def calculate_thresholds(scenario_name):
    print(f"CALCULATING THRESHOLDS FOR SCENARIO: {scenario_name}")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Phase2

    if scenario_name == 'main':
        DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Phase1', 'data')
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        THRESHOLDS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Phase3', 'thresholds')
    else:
        DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Phase1', f'data_{scenario_name}')
        MODELS_DIR = os.path.join(BASE_DIR, f'models_{scenario_name}')
        THRESHOLDS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Phase3', f'thresholds_{scenario_name}')
    
    if not os.path.exists(THRESHOLDS_DIR):
        os.makedirs(THRESHOLDS_DIR)

    for time_step in config.TIME_STEPS:
        for sampling_period in config.SAMPLING_PERIODS:
            config_name = f"{time_step}s_{sampling_period}s"
            print(f"\nProcessing {config_name}...")
            
            # Load Data
            val_path = os.path.join(DATA_DIR, f"{time_step}s_window", f"sampling_{sampling_period}s", "val.npy")
            if not os.path.exists(val_path):
                print(f"  Data not found: {val_path}")
                continue
                
            X_val_raw = np.load(val_path, allow_pickle=True)
            # Separate timestamp (col 0) and features (col 1:)
            # The raw data includes timestamp + features
            X_val = X_val_raw[:, :, 1:]
            
            # Cast to float32 (fixes object dtype issue)
            X_val = X_val.astype(np.float32)

            # Reshape for model (add channel dimension)
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
            
            # Load Model
            model_path = os.path.join(MODELS_DIR, f"{config_name}.h5")
            if not os.path.exists(model_path):
                print(f"  Model not found: {model_path}")
                continue
                
            try:
                model = load_model(model_path, compile=False)
                
                # Predict
                reconstructions = model.predict(X_val, verbose=0)
                
                # Calculate MSE per sample
                # Shape: (n_samples, time_steps, n_features, 1)
                mse = np.mean(np.square(X_val - reconstructions), axis=(1, 2, 3))
                
                # Calculate Thresholds (Percentiles)
                thresholds = {
                    'threshold': float(np.max(mse)), # Backward compatibility (Max)
                    'max': float(np.max(mse)),
                    '99.9': float(np.percentile(mse, 99.9)),
                    '99.5': float(np.percentile(mse, 99.5)),
                    '99': float(np.percentile(mse, 99)),
                    '95': float(np.percentile(mse, 95)),
                    '90': float(np.percentile(mse, 90)),
                    '85': float(np.percentile(mse, 85)),
                    '80': float(np.percentile(mse, 80)),
                    '75': float(np.percentile(mse, 75))
                }
                print(f"  Thresholds: Max={thresholds['max']:.4f}, 90%={thresholds['90']:.4f}, 75%={thresholds['75']:.4f}")
                
                # Save
                output_path = os.path.join(THRESHOLDS_DIR, f"{config_name}_thresholds.json")
                with open(output_path, 'w') as f:
                    json.dump(thresholds, f, indent=4)
                    
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        calculate_thresholds(sys.argv[1])
    else:
        print("Please provide scenario name")
