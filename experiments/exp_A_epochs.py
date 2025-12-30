#!/usr/bin/env python3
"""
EXP-A: Extended Training - 500 epochs, patience 50
All experiments use Scenario C (30 features = mean + std)
"""
import os
import sys
import numpy as np

# Fix path - get absolute project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Phase2', 'scripts'))

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models import create_autoencoder

# Experiment parameters
EXPERIMENT_NAME = "EXP_A_epochs"
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 50
BATCH_SIZE = 128

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "Phase1", "data_C_mean_std")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Phase2", f"models_{EXPERIMENT_NAME}")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_experiment():
    print("=" * 60)
    print(f"EXPERIMENT A: Extended Training (500 epochs)")
    print(f"Data dir: {DATA_DIR}")
    print(f"Models dir: {MODELS_DIR}")
    print("=" * 60)
    
    for window_size in [50, 75, 100]:
        for sampling in [1]:
            config_name = f"{window_size}s_{sampling}s"
            print(f"\nTraining {config_name}...")
            
            train_path = os.path.join(DATA_DIR, f"{window_size}s_window", f"sampling_{sampling}s", "train.npy")
            val_path = os.path.join(DATA_DIR, f"{window_size}s_window", f"sampling_{sampling}s", "val.npy")
            
            if not os.path.exists(train_path):
                print(f"  Skipping - data not found at {train_path}")
                continue
                
            train_data = np.load(train_path).astype(np.float32)
            val_data = np.load(val_path).astype(np.float32)
            
            # Get dimensions
            time_step = train_data.shape[1]
            num_signals = train_data.shape[2] if len(train_data.shape) > 2 else 30
            
            # Reshape for CNN (samples, time, signals, channels)
            train_data = train_data.reshape(-1, time_step, num_signals, 1)
            val_data = val_data.reshape(-1, time_step, num_signals, 1)
            
            print(f"  Train: {train_data.shape}, Val: {val_data.shape}")
            print(f"  time_step={time_step}, num_signals={num_signals}")
            
            # Build model using correct function
            model = create_autoencoder(time_step, num_signals)
            
            callbacks = [
                ModelCheckpoint(
                    os.path.join(MODELS_DIR, f"{config_name}.h5"),
                    save_best_only=True, monitor='val_loss', verbose=1
                ),
                EarlyStopping(patience=EARLY_STOPPING_PATIENCE, monitor='val_loss', verbose=1),
                ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-6, verbose=1)
            ]
            
            history = model.fit(
                train_data, train_data,
                validation_data=(val_data, val_data),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )
            
            print(f"  Done! Best val_loss: {min(history.history['val_loss']):.6f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT A COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    train_experiment()
