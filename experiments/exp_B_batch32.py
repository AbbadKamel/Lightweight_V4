#!/usr/bin/env python3
"""
EXP-B: Smaller Batch Size - Batch 32, LR 0.0001
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Phase2', 'scripts'))

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from models import create_autoencoder

EXPERIMENT_NAME = "EXP_B_batch32"
MAX_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

DATA_DIR = os.path.join(PROJECT_ROOT, "Phase1", "data_C_mean_std")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Phase2", f"models_{EXPERIMENT_NAME}")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_experiment():
    print("=" * 60)
    print(f"EXPERIMENT B: Smaller Batch (32) + Lower LR (0.0001)")
    print(f"Data dir: {DATA_DIR}")
    print("=" * 60)
    
    for window_size in [50, 75, 100]:
        for sampling in [1]:
            config_name = f"{window_size}s_{sampling}s"
            print(f"\nTraining {config_name}...")
            
            train_path = os.path.join(DATA_DIR, f"{window_size}s_window", f"sampling_{sampling}s", "train.npy")
            val_path = os.path.join(DATA_DIR, f"{window_size}s_window", f"sampling_{sampling}s", "val.npy")
            
            if not os.path.exists(train_path):
                print(f"  Skipping - data not found")
                continue
                
            train_data = np.load(train_path).astype(np.float32)
            val_data = np.load(val_path).astype(np.float32)
            
            time_step = train_data.shape[1]
            num_signals = train_data.shape[2] if len(train_data.shape) > 2 else 30
            
            train_data = train_data.reshape(-1, time_step, num_signals, 1)
            val_data = val_data.reshape(-1, time_step, num_signals, 1)
            
            print(f"  Train: {train_data.shape}, Val: {val_data.shape}")
            
            model = create_autoencoder(time_step, num_signals)
            # Recompile with lower LR
            model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['accuracy'])
            
            callbacks = [
                ModelCheckpoint(os.path.join(MODELS_DIR, f"{config_name}.h5"), save_best_only=True, verbose=1),
                EarlyStopping(patience=30, verbose=1),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)
            ]
            
            model.fit(train_data, train_data, validation_data=(val_data, val_data),
                      epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)
            
    print("\nEXPERIMENT B COMPLETE!")

if __name__ == "__main__":
    train_experiment()
