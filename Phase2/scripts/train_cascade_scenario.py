
import os
import sys
import json
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add Phase1/scripts to path for config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Phase1', 'scripts'))

from models import get_new_autoencoder, compile_autoencoder

# Import configuration from central config file
from config import (
    WINDOW_SIZES, SAMPLING_PERIODS,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    ADAM_BETA_1, ADAM_BETA_2,
    get_effective_batch_size
)

def train_scenario(scenario_name):
    print("="*80)
    print(f"TRAINING MODELS FOR SCENARIO: {scenario_name}")
    print("="*80)

    # 1. Load Scenario Config
    # We need to find where Phase1 is relative to this script
    # This script is in Phase2/scripts
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phase1_config_path = os.path.join(project_root, 'Phase1', 'results', f'config_{scenario_name}.json')
    
    if not os.path.exists(phase1_config_path):
        print(f"Config not found: {phase1_config_path}")
        return

    with open(phase1_config_path, 'r') as f:
        scenario_config = json.load(f)
    
    NUM_FEATURES = scenario_config['num_features']
    print(f"Using NUM_FEATURES: {NUM_FEATURES}")

    # 2. Setup Paths
    PHASE1_DATA_DIR = os.path.join(project_root, 'Phase1', f'data_{scenario_name}')
    PHASE2_MODELS_DIR = os.path.join(project_root, 'Phase2', f'models_{scenario_name}')
    
    if not os.path.exists(PHASE2_MODELS_DIR):
        os.makedirs(PHASE2_MODELS_DIR)

    # 3. Training Loop (Cascade)
    for window_size in WINDOW_SIZES:
        print(f"\n{'='*40}")
        print(f"Training Cascade for Window Size: {window_size}s")
        print(f"{'='*40}")
        
        previous_model = None
        
        for sampling_period in SAMPLING_PERIODS:
            config_name = f"{window_size}s_{sampling_period}s"
            print(f"\n--- Configuration: {config_name} ---")
            
            # A. Load Data
            data_dir = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s")
            train_path = os.path.join(data_dir, "train.npy")
            val_path = os.path.join(data_dir, "val.npy")
            
            if not os.path.exists(train_path):
                print(f"Data not found: {train_path}")
                continue
                
            train_raw = np.load(train_path, allow_pickle=True)
            val_raw = np.load(val_path, allow_pickle=True)
            
            # Reshape
            # Data is already clean (no timestamp)
            train_signals = train_raw.astype(np.float32)
            val_signals = val_raw.astype(np.float32)
            
            # Add channel dim
            train_data = train_signals.reshape(-1, train_signals.shape[1], NUM_FEATURES, 1)
            val_data = val_signals.reshape(-1, val_signals.shape[1], NUM_FEATURES, 1)
            
            print(f"Train shape: {train_data.shape}")
            
            # B. Initialize Model
            time_steps = train_data.shape[1]
            
            if previous_model is None:
                print("Initializing new model (from scratch)...")
                model = get_new_autoencoder(time_steps, NUM_FEATURES)
                compile_autoencoder(model)
            else:
                print("Transferring weights from previous model...")
                model = get_new_autoencoder(time_steps, NUM_FEATURES)
                compile_autoencoder(model)
                model.set_weights(previous_model.get_weights())
                
            # C. Train
            batch_size = get_effective_batch_size(train_data.shape[0])
            print(f"Training with batch size: {batch_size}")
            
            model_path = os.path.join(PHASE2_MODELS_DIR, f"{config_name}.h5")
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN, verbose=1)
            ]
            
            history = model.fit(
                train_data, train_data,
                validation_data=(val_data, val_data),
                epochs=MAX_EPOCHS,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # D. Update previous model for next iteration
            previous_model = model
            print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        train_scenario(scenario)
    else:
        print("Please provide scenario name")
