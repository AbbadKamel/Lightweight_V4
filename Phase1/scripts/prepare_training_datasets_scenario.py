
import pandas as pd
import numpy as np
import os
import sys
import json
import config

def create_windows(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def prepare_datasets(scenario_name):
    # 1. Load Scenario Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Config is in Phase1/results/
    config_path = os.path.join(os.path.dirname(script_dir), 'results', f'config_{scenario_name}.json')
    
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        scenario_config = json.load(f)

    dataset_path = scenario_config['dataset_path']
    print(f"Loading data from {dataset_path}...")
    
    # 2. Setup Output Directory
    # Phase1/data_{scenario}/
    output_dir = os.path.join(os.path.dirname(script_dir), f"data_{scenario_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Load Data
    df = pd.read_csv(dataset_path)
    
    # Filter only feature columns
    if 'feature_names' in scenario_config:
        df = df[scenario_config['feature_names']]
        
    data_values = df.values
    print(f"Data shape: {data_values.shape}")

    # 4. Generate Windows
    total_files = 0
    for time_step in config.TIME_STEPS:
        for sampling_period in config.SAMPLING_PERIODS:
            print(f"\nProcessing: Window={time_step}s, Sampling={sampling_period}s")
            
            sampled_data = data_values[::sampling_period]
            windows = create_windows(sampled_data, time_step, config.WINDOW_STEP_TRAIN)
            
            n_windows = len(windows)
            train_idx = int(n_windows * config.TRAIN_RATIO)
            val_idx = int(n_windows * (config.TRAIN_RATIO + config.VALID_RATIO))
            
            X_train = windows[:train_idx]
            X_val = windows[train_idx:val_idx]
            X_test = windows[val_idx:]
            
            config_dir = os.path.join(output_dir, f"{time_step}s_window", f"sampling_{sampling_period}s")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            np.save(os.path.join(config_dir, "train.npy"), X_train)
            np.save(os.path.join(config_dir, "val.npy"), X_val)
            np.save(os.path.join(config_dir, "test.npy"), X_test)
            
            total_files += 3
            
    print(f"\nGenerated {total_files} files in {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        prepare_datasets(scenario)
    else:
        print("Please provide scenario name")
