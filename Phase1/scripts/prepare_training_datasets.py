import pandas as pd
import numpy as np
import os
import config

def create_windows(data, window_size, step_size):
    """
    Create sliding windows from data
    data: (n_samples, n_features)
    window_size: int
    step_size: int
    Returns: (n_windows, window_size, n_features)
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def prepare_datasets():
    # 1. Setup directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Load Data
    input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.DATA_DIR)
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    data_values = df.values
    print(f"Data shape: {data_values.shape}")

    # 3. Iterate through all configurations
    total_files = 0
    
    print("\n" + "="*50)
    print("GENERATING DATASETS")
    print("="*50)

    for time_step in config.TIME_STEPS:
        for sampling_period in config.SAMPLING_PERIODS:
            print(f"\nProcessing: Window={time_step}s, Sampling={sampling_period}s")
            
            # A. Downsample data
            # Take every Nth sample based on sampling period
            # If sampling_period=1, takes all data
            sampled_data = data_values[::sampling_period]
            print(f"  - Original rows: {len(data_values)}")
            print(f"  - Sampled rows:  {len(sampled_data)}")
            
            # B. Create Windows
            # Use WINDOW_STEP_TRAIN from config (usually 10)
            windows = create_windows(sampled_data, time_step, config.WINDOW_STEP_TRAIN)
            print(f"  - Generated windows: {windows.shape}")
            
            # C. Split Data (Temporal Split)
            # We split the WINDOWS, not the raw data, to ensure valid sets
            n_windows = len(windows)
            train_idx = int(n_windows * config.TRAIN_RATIO)
            val_idx = int(n_windows * (config.TRAIN_RATIO + config.VALID_RATIO))
            
            X_train = windows[:train_idx]
            X_val = windows[train_idx:val_idx]
            X_test = windows[val_idx:]
            
            print(f"  - Train: {X_train.shape}")
            print(f"  - Val:   {X_val.shape}")
            print(f"  - Test:  {X_test.shape}")
            
            # D. Save .npy files
            # Create specific folder for this configuration
            # e.g., Phase1/data/50s_window/sampling_1s/
            config_dir = os.path.join(output_dir, f"{time_step}s_window", f"sampling_{sampling_period}s")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            np.save(os.path.join(config_dir, "train.npy"), X_train)
            np.save(os.path.join(config_dir, "val.npy"), X_val)
            np.save(os.path.join(config_dir, "test.npy"), X_test)
            
            total_files += 3
            
    print("\n" + "="*50)
    print(f"DONE! Generated {total_files} .npy files in {output_dir}")
    print("="*50)

if __name__ == "__main__":
    prepare_datasets()
