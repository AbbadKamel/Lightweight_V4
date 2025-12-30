#!/usr/bin/env python3
"""
EXP-C: Different Window Sizes - 30, 40, 50 seconds
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Phase2', 'scripts'))

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models import create_autoencoder

EXPERIMENT_NAME = "EXP_C_windows"
MAX_EPOCHS = 200
BATCH_SIZE = 128
WINDOW_SIZES = [30, 40, 50]
SAMPLING_PERIODS = [1, 2, 3]

OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "Phase1", f"data_{EXPERIMENT_NAME}")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Phase2", f"models_{EXPERIMENT_NAME}")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

def create_windows(data, window_size, step=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i+window_size])
    return np.array(windows) if windows else np.array([])

def prepare_data():
    print("Loading master table...")
    master_path = os.path.join(PROJECT_ROOT, "Phase1", "results", "master_table_C_mean_std.csv")
    if not os.path.exists(master_path):
        print(f"ERROR: Master table not found at {master_path}")
        return False
    
    df = pd.read_csv(master_path)
    data = df.iloc[:, 1:].values.astype(np.float32)
    print(f"Data shape: {data.shape}")
    
    for ws in WINDOW_SIZES:
        for sp in SAMPLING_PERIODS:
            print(f"\nCreating {ws}s_{sp}s...")
            sampled = data[::sp]
            windows = create_windows(sampled, ws, step=1)
            
            if len(windows) < 10:
                print(f"  Skipping - too few windows ({len(windows)})")
                continue
                
            n = len(windows)
            train = windows[:int(n*0.7)]
            val = windows[int(n*0.7):int(n*0.85)]
            test = windows[int(n*0.85):]
            
            out_dir = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s")
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "train.npy"), train)
            np.save(os.path.join(out_dir, "val.npy"), val)
            np.save(os.path.join(out_dir, "test.npy"), test)
            print(f"  Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    return True

def train_experiment():
    print("=" * 60)
    print(f"EXPERIMENT C: Different Windows {WINDOW_SIZES}")
    print("=" * 60)
    
    if not prepare_data():
        return
    
    for ws in WINDOW_SIZES:
        for sp in SAMPLING_PERIODS:
            config_name = f"{ws}s_{sp}s"
            print(f"\nTraining {config_name}...")
            
            train_path = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s", "train.npy")
            val_path = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s", "val.npy")
            
            if not os.path.exists(train_path):
                continue
                
            train_data = np.load(train_path).astype(np.float32)
            val_data = np.load(val_path).astype(np.float32)
            
            time_step = train_data.shape[1]
            num_signals = train_data.shape[2] if len(train_data.shape) > 2 else 30
            
            train_data = train_data.reshape(-1, time_step, num_signals, 1)
            val_data = val_data.reshape(-1, time_step, num_signals, 1)
            
            print(f"  Train: {train_data.shape}")
            
            model = create_autoencoder(time_step, num_signals)
            
            callbacks = [
                ModelCheckpoint(os.path.join(MODELS_DIR, f"{config_name}.h5"), save_best_only=True, verbose=1),
                EarlyStopping(patience=20, verbose=1),
                ReduceLROnPlateau(factor=0.2, patience=5, verbose=1)
            ]
            
            model.fit(train_data, train_data, validation_data=(val_data, val_data),
                      epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)
            
    print("\nEXPERIMENT C COMPLETE!")

if __name__ == "__main__":
    train_experiment()
