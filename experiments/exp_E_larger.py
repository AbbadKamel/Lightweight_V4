#!/usr/bin/env python3
"""
EXP-E: Larger Model Architecture
- Filters: [64, 32, 32, 32, 64]
- Dropout: 0.2
"""
import os
import sys
import numpy as np

# Fix path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Phase1', 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Phase2', 'scripts'))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

EXPERIMENT_NAME = "EXP_E_larger"
MAX_EPOCHS = 200
BATCH_SIZE = 64
DROPOUT_RATE = 0.2

DATA_DIR = os.path.join(PROJECT_ROOT, "Phase1", "data_C_mean_std")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Phase2", f"models_{EXPERIMENT_NAME}")
os.makedirs(MODELS_DIR, exist_ok=True)

def build_larger_autoencoder(input_shape):
    """Larger CNN autoencoder with more filters and dropout."""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_experiment():
    print("=" * 60)
    print(f"EXPERIMENT E: Larger Model [64,32,32,32,64] + Dropout 0.2")
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
            
            n_features = train_data.shape[2] if len(train_data.shape) > 2 else 30
            train_data = train_data.reshape(-1, window_size, n_features, 1)
            val_data = val_data.reshape(-1, window_size, n_features, 1)
            
            print(f"  Train: {train_data.shape}, Val: {val_data.shape}")
            
            model = build_larger_autoencoder(train_data.shape[1:])
            
            callbacks = [
                ModelCheckpoint(os.path.join(MODELS_DIR, f"{config_name}.h5"), save_best_only=True, verbose=1),
                EarlyStopping(patience=30, verbose=1),
                ReduceLROnPlateau(factor=0.2, patience=10, verbose=1)
            ]
            
            model.fit(train_data, train_data, validation_data=(val_data, val_data),
                      epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)
            
    print("\nEXPERIMENT E COMPLETE!")

if __name__ == "__main__":
    train_experiment()
