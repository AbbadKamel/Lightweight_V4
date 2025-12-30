#!/usr/bin/env python3
"""
Fair Comparison: Scenario A vs Scenario C
==========================================
This script does everything needed for a fair comparison:
1. Regenerate Scenario C data with WINDOW_STEP=1
2. Train models on new data
3. Generate proper attacks
4. Calculate thresholds
5. Run ensemble evaluation
6. Compare AUROC
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Phase2', 'scripts'))

# Configuration
SCENARIO = "C_fair"  # New name for fair comparison data
WINDOW_SIZES = [50, 75, 100]
SAMPLING_PERIODS = [1, 5, 10]
WINDOW_STEP = 1  # Same as Scenario A!
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
MASTER_TABLE = os.path.join(PROJECT_ROOT, "Phase1", "results", "master_table_C_mean_std.csv")
OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "Phase1", f"data_{SCENARIO}")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Phase2", f"models_{SCENARIO}")
THRESHOLDS_DIR = os.path.join(PROJECT_ROOT, "Phase3", f"thresholds_{SCENARIO}")
ATTACKS_DIR = os.path.join(PROJECT_ROOT, "Phase3", f"attacks_{SCENARIO}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Phase3", f"results_{SCENARIO}")

for d in [OUTPUT_DATA_DIR, MODELS_DIR, THRESHOLDS_DIR, ATTACKS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# STEP 1: Regenerate Data with WINDOW_STEP=1
# ============================================================================
def step1_regenerate_data():
    print("=" * 60)
    print("STEP 1: Regenerating Scenario C data with WINDOW_STEP=1")
    print("=" * 60)
    
    # Load master table
    df = pd.read_csv(MASTER_TABLE)
    data = df.iloc[:, 1:].values.astype(np.float32)  # Skip timestamp column
    n_features = data.shape[1]
    
    print(f"Loaded master table: {data.shape} (samples, features)")
    print(f"Features: {n_features}")
    
    total_windows = 0
    
    for ws in WINDOW_SIZES:
        for sp in SAMPLING_PERIODS:
            config_name = f"{ws}s_{sp}s"
            
            # Downsample
            sampled = data[::sp]
            
            # Create windows with STEP=1
            windows = []
            for i in range(0, len(sampled) - ws + 1, WINDOW_STEP):
                windows.append(sampled[i:i+ws])
            
            if len(windows) < 10:
                print(f"  {config_name}: Skipping (only {len(windows)} windows)")
                continue
            
            windows = np.array(windows)
            
            # Split
            n = len(windows)
            n_train = int(n * TRAIN_RATIO)
            n_val = int(n * VAL_RATIO)
            
            train = windows[:n_train]
            val = windows[n_train:n_train+n_val]
            test = windows[n_train+n_val:]
            
            # Save
            out_dir = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s")
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "train.npy"), train)
            np.save(os.path.join(out_dir, "val.npy"), val)
            np.save(os.path.join(out_dir, "test.npy"), test)
            
            total_windows += n
            print(f"  {config_name}: {n} windows → train={len(train)}, val={len(val)}, test={len(test)}")
    
    print(f"\nTotal windows created: {total_windows}")
    return True

# ============================================================================
# STEP 2: Train Models
# ============================================================================
def step2_train_models():
    print("\n" + "=" * 60)
    print("STEP 2: Training models on new data")
    print("=" * 60)
    
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from models import create_autoencoder
    
    for ws in WINDOW_SIZES:
        for sp in SAMPLING_PERIODS:
            config_name = f"{ws}s_{sp}s"
            
            data_dir = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s")
            train_path = os.path.join(data_dir, "train.npy")
            val_path = os.path.join(data_dir, "val.npy")
            
            if not os.path.exists(train_path):
                continue
            
            print(f"\nTraining {config_name}...")
            
            train_data = np.load(train_path).astype(np.float32)
            val_data = np.load(val_path).astype(np.float32)
            
            time_step = train_data.shape[1]
            n_features = train_data.shape[2]
            
            # Add channel dimension
            train_data = train_data.reshape(-1, time_step, n_features, 1)
            val_data = val_data.reshape(-1, time_step, n_features, 1)
            
            print(f"  Train: {train_data.shape}, Val: {val_data.shape}")
            
            # Build model
            model = create_autoencoder(time_step, n_features)
            
            callbacks = [
                ModelCheckpoint(
                    os.path.join(MODELS_DIR, f"{config_name}.h5"),
                    save_best_only=True, monitor='val_loss', verbose=0
                ),
                EarlyStopping(patience=10, monitor='val_loss', verbose=0),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=0)
            ]
            
            history = model.fit(
                train_data, train_data,
                validation_data=(val_data, val_data),
                epochs=100,
                batch_size=128,
                callbacks=callbacks,
                verbose=0
            )
            
            best_val_loss = min(history.history['val_loss'])
            print(f"  Done! Best val_loss: {best_val_loss:.6f}")
    
    return True

# ============================================================================
# STEP 3: Generate Attacks
# ============================================================================
def step3_generate_attacks():
    print("\n" + "=" * 60)
    print("STEP 3: Generating proper attacks")
    print("=" * 60)
    
    ATTACK_TYPES = ['spike', 'constant', 'replay', 'drift', 'noise', 'scaling']
    ATTACKS_PER_TYPE = 50  # 50 * 6 = 300 attacks total
    
    for ws in WINDOW_SIZES:
        for sp in SAMPLING_PERIODS:
            config_name = f"{ws}s_{sp}s"
            
            test_path = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s", "test.npy")
            if not os.path.exists(test_path):
                continue
            
            test_data = np.load(test_path).astype(np.float32)
            n_samples, time_step, n_features = test_data.shape
            
            print(f"\n{config_name}: Generating attacks from {n_samples} test samples...")
            
            attacks = []
            metadata = []
            
            for attack_type in ATTACK_TYPES:
                for _ in range(ATTACKS_PER_TYPE):
                    base_idx = np.random.randint(0, n_samples)
                    attack = test_data[base_idx].copy()
                    feat_idx = np.random.randint(0, n_features)
                    
                    if attack_type == 'spike':
                        t_idx = np.random.randint(0, time_step)
                        magnitude = np.random.uniform(0.8, 1.0)
                        attack[t_idx, feat_idx] = magnitude
                        
                    elif attack_type == 'constant':
                        const_val = np.random.choice([0.0, 1.0])
                        for t in range(time_step):
                            attack[t, feat_idx] = const_val
                            
                    elif attack_type == 'replay':
                        src_idx = np.random.randint(0, n_samples)
                        for t in range(time_step):
                            attack[t, feat_idx] = test_data[src_idx, t, feat_idx]
                            
                    elif attack_type == 'drift':
                        drift_rate = np.random.uniform(0.005, 0.02)
                        for t in range(time_step):
                            attack[t, feat_idx] = min(1.0, attack[t, feat_idx] + drift_rate * t)
                            
                    elif attack_type == 'noise':
                        noise_std = np.random.uniform(0.1, 0.3)
                        for t in range(time_step):
                            attack[t, feat_idx] = np.clip(attack[t, feat_idx] + np.random.normal(0, noise_std), 0, 1)
                            
                    elif attack_type == 'scaling':
                        scale = np.random.uniform(1.5, 3.0)
                        for t in range(time_step):
                            attack[t, feat_idx] = min(1.0, attack[t, feat_idx] * scale)
                    
                    attacks.append(attack)
                    metadata.append({
                        'attack_type': attack_type,
                        'source_idx': int(base_idx),
                        'target_feature': int(feat_idx)
                    })
            
            attacks = np.array(attacks).astype(np.float32)
            attacks = attacks.reshape(-1, time_step, n_features, 1)
            
            # Save
            np.save(os.path.join(ATTACKS_DIR, f"attacks_{config_name}.npy"), attacks)
            with open(os.path.join(ATTACKS_DIR, f"attacks_{config_name}_metadata.json"), 'w') as f:
                json.dump(metadata, f)
            
            print(f"  Saved {len(attacks)} attacks ({ATTACKS_PER_TYPE} per type)")
    
    return True

# ============================================================================
# STEP 4: Calculate Thresholds
# ============================================================================
def step4_calculate_thresholds():
    print("\n" + "=" * 60)
    print("STEP 4: Calculating thresholds")
    print("=" * 60)
    
    from tensorflow.keras.models import load_model
    
    PERCENTILES = [75, 90, 95, 99]
    
    for ws in WINDOW_SIZES:
        for sp in SAMPLING_PERIODS:
            config_name = f"{ws}s_{sp}s"
            
            model_path = os.path.join(MODELS_DIR, f"{config_name}.h5")
            train_path = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s", "train.npy")
            
            if not os.path.exists(model_path) or not os.path.exists(train_path):
                continue
            
            print(f"\n{config_name}: Calculating thresholds...")
            
            model = load_model(model_path, compile=False)
            train_data = np.load(train_path).astype(np.float32)
            
            time_step = train_data.shape[1]
            n_features = train_data.shape[2]
            train_data = train_data.reshape(-1, time_step, n_features, 1)
            
            # Predict
            pred = model.predict(train_data, verbose=0)
            mse = np.mean(np.square(train_data - pred), axis=(1, 2, 3))
            
            # Calculate percentile thresholds
            thresholds = {}
            for p in PERCENTILES:
                thresholds[str(p)] = float(np.percentile(mse, p))
            
            # Save
            with open(os.path.join(THRESHOLDS_DIR, f"{config_name}_thresholds.json"), 'w') as f:
                json.dump(thresholds, f, indent=2)
            
            print(f"  Thresholds: {thresholds}")
    
    return True

# ============================================================================
# STEP 5: Ensemble Evaluation
# ============================================================================
def step5_ensemble_evaluation():
    print("\n" + "=" * 60)
    print("STEP 5: Ensemble AUROC Evaluation")
    print("=" * 60)
    
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from tensorflow.keras.models import load_model
    
    STABLE_MODELS = ['50s_1s', '75s_1s', '50s_5s', '100s_1s']  # Same as Scenario A
    ATTACK_TYPES = ['spike', 'constant', 'replay', 'drift', 'noise', 'scaling']
    
    all_results = {}
    ensemble_normal_scores = []
    ensemble_attack_scores = []
    per_attack_scores = {at: {'normal': [], 'attack': []} for at in ATTACK_TYPES}
    
    for model_name in STABLE_MODELS:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
        
        if not os.path.exists(model_path):
            print(f"  {model_name}: Model not found, skipping")
            continue
        
        parts = model_name.split('_')
        ws = int(parts[0][:-1])
        sp = int(parts[1][:-1])
        
        test_path = os.path.join(OUTPUT_DATA_DIR, f"{ws}s_window", f"sampling_{sp}s", "test.npy")
        attack_path = os.path.join(ATTACKS_DIR, f"attacks_{model_name}.npy")
        meta_path = os.path.join(ATTACKS_DIR, f"attacks_{model_name}_metadata.json")
        
        if not os.path.exists(test_path) or not os.path.exists(attack_path):
            print(f"  {model_name}: Data not found, skipping")
            continue
        
        print(f"\n{model_name}: Evaluating...")
        
        model = load_model(model_path, compile=False)
        
        test_data = np.load(test_path).astype(np.float32)
        time_step, n_features = test_data.shape[1], test_data.shape[2]
        test_data = test_data.reshape(-1, time_step, n_features, 1)
        
        attack_data = np.load(attack_path).astype(np.float32)
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        # Compute MSE
        pred_normal = model.predict(test_data, verbose=0)
        pred_attack = model.predict(attack_data, verbose=0)
        
        mse_normal = np.mean(np.square(test_data - pred_normal), axis=(1, 2, 3))
        mse_attack = np.mean(np.square(attack_data - pred_attack), axis=(1, 2, 3))
        
        print(f"  Normal MSE: {np.mean(mse_normal):.6f}, Attack MSE: {np.mean(mse_attack):.6f}")
        
        # Add to ensemble
        ensemble_normal_scores.append(mse_normal)
        ensemble_attack_scores.append(mse_attack)
        
        # Per-attack type
        for i, meta in enumerate(metadata):
            at = meta['attack_type']
            per_attack_scores[at]['attack'].append(mse_attack[i])
            if i < len(mse_normal):
                per_attack_scores[at]['normal'].append(mse_normal[i % len(mse_normal)])
        
        # Single model AUROC
        y_true = [0] * len(mse_normal) + [1] * len(mse_attack)
        y_scores = list(mse_normal) + list(mse_attack)
        auroc = roc_auc_score(y_true, y_scores)
        all_results[model_name] = {'auroc': auroc}
        print(f"  Single model AUROC: {auroc:.4f}")
    
    # Ensemble AUROC (max across models)
    if ensemble_normal_scores:
        ens_normal = np.max(np.array(ensemble_normal_scores), axis=0)
        ens_attack = np.max(np.array(ensemble_attack_scores), axis=0)
        
        y_true = [0] * len(ens_normal) + [1] * len(ens_attack)
        y_scores = list(ens_normal) + list(ens_attack)
        ensemble_auroc = roc_auc_score(y_true, y_scores)
        
        all_results['ensemble'] = {'auroc': ensemble_auroc, 'n_models': len(ensemble_normal_scores)}
        print(f"\n  ENSEMBLE AUROC ({len(ensemble_normal_scores)} models): {ensemble_auroc:.4f}")
    
    # Per-attack type AUROC
    print("\n  Per-Attack AUROC:")
    for at in ATTACK_TYPES:
        if per_attack_scores[at]['attack'] and per_attack_scores[at]['normal']:
            n_atk = len(per_attack_scores[at]['attack'])
            n_nor = min(n_atk, len(per_attack_scores[at]['normal']))
            
            y_true = [0] * n_nor + [1] * n_atk
            y_scores = per_attack_scores[at]['normal'][:n_nor] + per_attack_scores[at]['attack']
            auroc = roc_auc_score(y_true, y_scores)
            all_results[f'attack_{at}'] = auroc
            print(f"    {at}: {auroc:.4f}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("FAIR COMPARISON: SCENARIO A vs SCENARIO C")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    
    # Step 1
    if not step1_regenerate_data():
        print("Step 1 failed!")
        return
    
    # Step 2
    if not step2_train_models():
        print("Step 2 failed!")
        return
    
    # Step 3
    if not step3_generate_attacks():
        print("Step 3 failed!")
        return
    
    # Step 4
    if not step4_calculate_thresholds():
        print("Step 4 failed!")
        return
    
    # Step 5
    results = step5_ensemble_evaluation()
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: SCENARIO A vs SCENARIO C (Fair)")
    print("=" * 60)
    print(f"Scenario A Ensemble AUROC: 0.923")
    if results and 'ensemble' in results:
        print(f"Scenario C Ensemble AUROC: {results['ensemble']['auroc']:.4f}")
        diff = results['ensemble']['auroc'] - 0.923
        print(f"Difference: {diff:+.4f}")
    
    print(f"\nCompleted at: {datetime.now()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
