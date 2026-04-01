"""
Phase 2: Transfer Learning Cascade Training
============================================
Based on CANShield paper (Section IV.C)

Strategy:
    For each window_size:
        1. Train 1s model from scratch
        2. Transfer weights to 5s model, fine-tune
        3. Transfer weights to 10s model, fine-tune

This reduces training cost as described in the paper:
    "We initialize any tth model AEt with the preceding trained model AEt-1.
     Such a technique reduces the training cost."
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_new_autoencoder, compile_autoencoder

# Import configuration from central config file
from config import (
    WINDOW_SIZES, SAMPLING_PERIODS, NUM_FEATURES,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    ADAM_BETA_1, ADAM_BETA_2,
    get_effective_batch_size
)

# ============================================================================
# PATHS
# ============================================================================

PHASE1_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../Phase1/data")
PHASE2_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
PHASE2_LOGS_DIR = os.path.join(os.path.dirname(__file__), "../logs")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(window_size: int, sampling_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training and validation data for a specific configuration.
    
    Args:
        window_size: Window size in seconds (50, 75, or 100)
        sampling_period: Sampling period in seconds (1, 5, or 10)
    
    Returns:
        Tuple of (train_data, val_data) ready for CNN
    """
    data_dir = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s")
    
    # Load .npy files
    train_path = os.path.join(data_dir, "train.npy")
    val_path = os.path.join(data_dir, "val.npy")
    
    print(f"  Loading: {train_path}")
    train_raw = np.load(train_path, allow_pickle=True)
    val_raw = np.load(val_path, allow_pickle=True)
    
    # Remove timestamp column (column 0), keep only signal features
    # Shape: (n_samples, time_steps, 61) → (n_samples, time_steps, 60)
    train_signals = train_raw[:, :, 1:].astype(np.float32)
    val_signals = val_raw[:, :, 1:].astype(np.float32)
    
    # Add channel dimension for CNN
    # Shape: (n_samples, time_steps, 60) → (n_samples, time_steps, 60, 1)
    train_data = train_signals.reshape(-1, train_signals.shape[1], NUM_FEATURES, 1)
    val_data = val_signals.reshape(-1, val_signals.shape[1], NUM_FEATURES, 1)
    
    print(f"  Train shape: {train_data.shape}")
    print(f"  Val shape:   {val_data.shape}")
    
    return train_data, val_data


# ============================================================================
# MODEL TRAINING
# ============================================================================

def get_callbacks(model_path: str) -> list:
    """
    Create training callbacks for early stopping, model saving, and LR reduction.
    Uses parameters from config.py.
    """
    callbacks = [
        # Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model based on validation loss
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate if validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,  # From config.py
            verbose=1
        )
    ]
    return callbacks


def train_model(
    window_size: int,
    sampling_period: int,
    previous_model_path: Optional[str] = None
) -> dict:
    """
    Train a single autoencoder model.
    
    Args:
        window_size: Window size in seconds
        sampling_period: Sampling period in seconds
        previous_model_path: Path to previous model for transfer learning (None for 1s models)
    
    Returns:
        Training history dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training: {window_size}s window, {sampling_period}s sampling")
    if previous_model_path:
        print(f"Transfer learning from: {os.path.basename(previous_model_path)}")
    else:
        print("Training from SCRATCH (no transfer learning)")
    print("="*60)
    
    # 1. Load data
    print("\n📂 Loading data...")
    train_data, val_data = load_data(window_size, sampling_period)
    
    # Get dimensions from data
    time_steps = train_data.shape[1]
    num_signals = train_data.shape[2]
    
    print(f"  Time steps: {time_steps}")
    print(f"  Num signals: {num_signals}")
    
    # 2. Build model
    print("\n🏗️ Building model...")
    autoencoder = get_new_autoencoder(time_steps, num_signals)
    
    # 3. Transfer learning: load weights from previous model
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"\n🔄 Loading weights from: {os.path.basename(previous_model_path)}")
        try:
            previous_model = load_model(previous_model_path)
            # Transfer weights layer by layer (same architecture, different input size)
            # This works because our architecture uses 'same' padding
            for i, layer in enumerate(autoencoder.layers):
                if layer.get_weights():  # Only layers with weights
                    try:
                        layer.set_weights(previous_model.layers[i].get_weights())
                    except ValueError as e:
                        print(f"  ⚠️ Could not transfer weights for layer {i}: {e}")
            print("  ✅ Weights transferred successfully")
        except Exception as e:
            print(f"  ⚠️ Could not load previous model: {e}")
            print("  Training from scratch instead...")
    
    # 4. Compile model
    print("\n⚙️ Compiling model...")
    autoencoder = compile_autoencoder(autoencoder)
    
    # 5. Setup output paths
    model_name = f"{window_size}s_{sampling_period}s.h5"
    model_path = os.path.join(PHASE2_MODELS_DIR, model_name)
    
    # 6. Calculate effective batch size (handle small datasets)
    num_train_samples = train_data.shape[0]
    effective_batch_size = get_effective_batch_size(num_train_samples)
    
    # 6. Train
    print(f"\n🚀 Training for up to {MAX_EPOCHS} epochs...")
    print(f"  Training samples: {num_train_samples}")
    print(f"  Batch size: {effective_batch_size} (default={BATCH_SIZE})")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    history = autoencoder.fit(
        train_data, train_data,  # Autoencoder: input = target
        validation_data=(val_data, val_data),
        epochs=MAX_EPOCHS,
        batch_size=effective_batch_size,  # Use dynamic batch size
        callbacks=get_callbacks(model_path),
        verbose=1
    )
    
    # 7. Save training history
    history_path = os.path.join(PHASE2_LOGS_DIR, f"{window_size}s_{sampling_period}s_history.json")
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'config': {
            'window_size': window_size,
            'sampling_period': sampling_period,
            'time_steps': time_steps,
            'num_signals': num_signals,
            'batch_size': effective_batch_size,  # Log actual batch size used
            'default_batch_size': BATCH_SIZE,
            'transfer_learning': previous_model_path is not None
        }
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ History saved: {history_path}")
    print(f"📊 Final val_loss: {history_dict['final_val_loss']:.6f}")
    print(f"📊 Best val_loss: {history_dict['best_val_loss']:.6f}")
    print(f"📊 Epochs trained: {history_dict['epochs_trained']}")
    
    return history_dict


# ============================================================================
# CASCADE TRAINING
# ============================================================================

def train_cascade():
    """
    Train all models using the Transfer Learning Cascade strategy.
    
    For each window size:
        1. Train 1s model from scratch
        2. Transfer weights to 5s model, fine-tune
        3. Transfer weights to 10s model, fine-tune
    """
    print("="*60)
    print("PHASE 2: TRANSFER LEARNING CASCADE TRAINING")
    print("="*60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nWindow sizes: {WINDOW_SIZES}")
    print(f"Sampling periods: {SAMPLING_PERIODS}")
    print(f"Total models to train: {len(WINDOW_SIZES) * len(SAMPLING_PERIODS)}")
    
    # Ensure output directories exist
    os.makedirs(PHASE2_MODELS_DIR, exist_ok=True)
    os.makedirs(PHASE2_LOGS_DIR, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Train each window size
    for window_size in WINDOW_SIZES:
        print(f"\n{'#'*60}")
        print(f"# WINDOW SIZE: {window_size}s")
        print(f"{'#'*60}")
        
        previous_model_path = None  # First model trains from scratch
        
        # Train cascade: 1s → 5s → 10s
        for sampling_period in SAMPLING_PERIODS:
            config_name = f"{window_size}s_{sampling_period}s"
            
            # Train model
            history = train_model(
                window_size=window_size,
                sampling_period=sampling_period,
                previous_model_path=previous_model_path
            )
            
            all_results[config_name] = history
            
            # Update previous model path for next iteration (transfer learning)
            previous_model_path = os.path.join(PHASE2_MODELS_DIR, f"{window_size}s_{sampling_period}s.h5")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'Config':<15} {'Epochs':<10} {'Best Val Loss':<15} {'Transfer':<10}")
    print("-"*50)
    
    for config_name, history in all_results.items():
        transfer = "Yes" if history['config']['transfer_learning'] else "No"
        print(f"{config_name:<15} {history['epochs_trained']:<10} {history['best_val_loss']:<15.6f} {transfer:<10}")
    
    # Save overall summary
    summary_path = os.path.join(PHASE2_LOGS_DIR, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✅ Summary saved: {summary_path}")
    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run cascade training
    train_cascade()
