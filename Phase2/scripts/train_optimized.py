
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add Phase 1 scripts for config
sys.path.append(os.path.join(os.path.dirname(__file__), "../../Phase1/scripts"))
from models import get_new_autoencoder, compile_autoencoder
from config import BATCH_SIZE, MAX_EPOCHS, get_effective_batch_size, EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN
# Note: REDUCE_LR_P might be a typo for PATIENCE in import list, checking config...
# It is REDUCE_LR_PATIENCE in Phase1/config.py

# Fix for missing import from config if named differently
try:
    from config import REDUCE_LR_PATIENCE
except ImportError:
    REDUCE_LR_PATIENCE = 5

PHASE1_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../Phase1/data")
PHASE2_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
PHASE2_LOGS_DIR = os.path.join(os.path.dirname(__file__), "../logs")

TARGET_MODELS = [
    (50, 1), # 50s window, 1s sampling
    (50, 5)  # 50s window, 5s sampling
]

def load_data(window_size, sampling_period):
    data_dir = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s")
    train_raw = np.load(os.path.join(data_dir, "train.npy"), allow_pickle=True)
    val_raw = np.load(os.path.join(data_dir, "val.npy"), allow_pickle=True)
    
    # Extract features (drop timestamp)
    train_data = train_raw[:, :, 1:].astype(np.float32)
    val_data = val_raw[:, :, 1:].astype(np.float32)
    
    # Reshape for CNN
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
    val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], 1)
    
    return train_data, val_data

def train_one(window_size, sampling_period):
    print(f"\nTraining Optimized Model: {window_size}s_{sampling_period}s")
    
    # Load Data
    train_data, val_data = load_data(window_size, sampling_period)
    time_steps = train_data.shape[1]
    num_signals = train_data.shape[2]
    
    # Build Model (New Architecture with Bottleneck)
    model = get_new_autoencoder(time_steps, num_signals)
    model = compile_autoencoder(model)
    
    # Model Path
    model_name = f"{window_size}s_{sampling_period}s.h5"
    model_path = os.path.join(PHASE2_MODELS_DIR, model_name)
    
    # Train
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN, verbose=1)
    ]
    
    bs = get_effective_batch_size(len(train_data))
    
    history = model.fit(
        train_data, train_data,
        validation_data=(val_data, val_data),
        epochs=MAX_EPOCHS,
        batch_size=bs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def main():
    print("="*60)
    print("PHASE 4: OPTIMIZED RETRAINING (Committee Only)")
    print("="*60)
    
    for ws, sp in TARGET_MODELS:
        train_one(ws, sp)
        
    print("\noptimization Complete.")

if __name__ == "__main__":
    main()
