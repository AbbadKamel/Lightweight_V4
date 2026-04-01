"""
Recovery Script: Retrain Corrupted Models (50s_1s and 50s_5s)
Uses the restored original architecture (no bottleneck).
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../Phase1/scripts"))

from models import get_new_autoencoder, compile_autoencoder
from config import (
    BATCH_SIZE, MAX_EPOCHS, get_effective_batch_size, 
    EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, REDUCE_LR_MIN,
    NUM_FEATURES
)

PHASE1_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../Phase1/data")
PHASE2_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

# Only retrain corrupted models
TARGET_MODELS = [
    (50, 1),  # 50s_1s
    (50, 5),  # 50s_5s
]

def load_data(window_size, sampling_period):
    data_dir = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s")
    
    train_raw = np.load(os.path.join(data_dir, "train.npy"), allow_pickle=True)
    val_raw = np.load(os.path.join(data_dir, "val.npy"), allow_pickle=True)
    
    # Remove timestamp (column 0), keep features
    train_data = train_raw[:, :, 1:].astype(np.float32)
    val_data = val_raw[:, :, 1:].astype(np.float32)
    
    # Reshape for CNN: (samples, time, features, 1)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], NUM_FEATURES, 1)
    val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], NUM_FEATURES, 1)
    
    return train_data, val_data

def train_model(window_size, sampling_period):
    config_name = f"{window_size}s_{sampling_period}s"
    print(f"\n{'='*60}")
    print(f"RETRAINING: {config_name} (Restored Architecture)")
    print(f"{'='*60}")
    
    # Load data
    train_data, val_data = load_data(window_size, sampling_period)
    time_steps = train_data.shape[1]
    num_signals = train_data.shape[2]
    
    print(f"  Train shape: {train_data.shape}")
    print(f"  Val shape: {val_data.shape}")
    
    # Build model (original architecture)
    model = get_new_autoencoder(time_steps, num_signals)
    model = compile_autoencoder(model)
    
    # Paths
    model_path = os.path.join(PHASE2_MODELS_DIR, f"{config_name}.h5")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, 
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', 
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, 
                          patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN, verbose=1)
    ]
    
    # Dynamic batch size
    batch_size = get_effective_batch_size(len(train_data))
    
    print(f"  Batch size: {batch_size}")
    print(f"  Training for up to {MAX_EPOCHS} epochs...")
    
    # Train
    history = model.fit(
        train_data, train_data,
        validation_data=(val_data, val_data),
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"  ✅ Model saved: {model_path}")
    return history

def main():
    print("="*60)
    print("RECOVERY: Retraining Corrupted Models")
    print("="*60)
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    for ws, sp in TARGET_MODELS:
        train_model(ws, sp)
    
    print("\n" + "="*60)
    print("✅ Recovery Complete! Models restored.")
    print("="*60)

if __name__ == "__main__":
    main()
