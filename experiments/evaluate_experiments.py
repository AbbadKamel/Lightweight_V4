#!/usr/bin/env python3
"""
Evaluate AUROC for Scenario C experiments (EXP_A, EXP_B, EXP_C, EXP_D)
"""
import os
import sys
import numpy as np
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Phase3', 'scripts'))

from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

# Experiment configurations
EXPERIMENTS = {
    'EXP_A_epochs': {'name': '500 epochs', 'models_dir': 'models_EXP_A_epochs'},
    'EXP_B_batch32': {'name': 'Batch 32', 'models_dir': 'models_EXP_B_batch32'},
    'EXP_C_windows': {'name': 'Different windows', 'models_dir': 'models_EXP_C_windows'},
    'EXP_D_highLR': {'name': 'High LR', 'models_dir': 'models_EXP_D_highLR'},
}

DATA_DIR = os.path.join(PROJECT_ROOT, "Phase1", "data_C_mean_std")

def generate_attacks(normal_data, num_attacks=100):
    """Generate synthetic attacks - simple version"""
    attacks = []
    n_samples, time_steps, n_features = normal_data.shape
    
    for i in range(num_attacks):
        attack_type = i % 4
        base_idx = np.random.randint(0, n_samples)
        attack = normal_data[base_idx].copy()  # Shape: (time_steps, n_features)
        
        feat_idx = np.random.randint(0, n_features)
        
        if attack_type == 0:  # Spike
            t_idx = np.random.randint(0, time_steps)
            attack[t_idx, feat_idx] = 1.0  # Max value spike
        elif attack_type == 1:  # Constant zero
            for t in range(time_steps):
                attack[t, feat_idx] = 0.0
        elif attack_type == 2:  # Noise
            for t in range(time_steps):
                attack[t, feat_idx] = attack[t, feat_idx] + np.random.normal(0, 0.3)
        else:  # Scaling
            for t in range(time_steps):
                attack[t, feat_idx] = attack[t, feat_idx] * 2.0
        
        # Clip to valid range
        attack = np.clip(attack, 0, 1)
        attacks.append(attack)
    
    return np.array(attacks)

def evaluate_experiment(exp_key, exp_info):
    """Evaluate AUROC for an experiment"""
    models_path = os.path.join(PROJECT_ROOT, "Phase2", exp_info['models_dir'])
    
    if not os.path.exists(models_path):
        print(f"  Skipping {exp_key} - no models found")
        return None
    
    model_files = [f for f in os.listdir(models_path) if f.endswith('.h5')]
    if not model_files:
        print(f"  Skipping {exp_key} - no .h5 files")
        return None
    
    results = {}
    all_scores_normal = []
    all_scores_attack = []
    
    for mf in model_files:
        config = mf.replace('.h5', '')
        parts = config.split('_')
        ws = int(parts[0].replace('s', ''))
        sp = int(parts[1].replace('s', ''))
        
        model_path = os.path.join(models_path, mf)
        
        # Find matching data
        if 'EXP_C_windows' in exp_key:
            data_exp_dir = os.path.join(PROJECT_ROOT, "Phase1", "data_EXP_C_windows")
        else:
            data_exp_dir = DATA_DIR
            
        data_path = os.path.join(data_exp_dir, f"{ws}s_window", f"sampling_{sp}s")
        test_file = os.path.join(data_path, "test.npy")
        
        if not os.path.exists(test_file):
            print(f"    Data not found: {test_file}")
            continue
        
        try:
            model = load_model(model_path, compile=False)
            test_data = np.load(test_file).astype(np.float32)
            
            print(f"    {config}: data shape = {test_data.shape}")
            
            # Get dimensions from data
            n_samples, time_steps, n_features = test_data.shape
            
            # Generate attacks (same shape as test data)
            attacks = generate_attacks(test_data, num_attacks=100)
            
            # Reshape for model (add channel dim)
            test_data_4d = test_data.reshape(n_samples, time_steps, n_features, 1)
            attacks_4d = attacks.reshape(len(attacks), time_steps, n_features, 1)
            
            # Predict
            pred_normal = model.predict(test_data_4d, verbose=0)
            pred_attack = model.predict(attacks_4d, verbose=0)
            
            # Calculate MSE
            mse_normal = np.mean((test_data_4d - pred_normal) ** 2, axis=(1, 2, 3))
            mse_attack = np.mean((attacks_4d - pred_attack) ** 2, axis=(1, 2, 3))
            
            all_scores_normal.extend(mse_normal.tolist())
            all_scores_attack.extend(mse_attack.tolist())
            
            results[config] = {
                'mse_normal_mean': float(np.mean(mse_normal)),
                'mse_attack_mean': float(np.mean(mse_attack)),
            }
            print(f"    {config}: MSE normal={np.mean(mse_normal):.6f}, attack={np.mean(mse_attack):.6f}")
            
        except Exception as e:
            import traceback
            print(f"    Error with {config}: {e}")
            traceback.print_exc()
            continue
    
    # Calculate overall AUROC
    if all_scores_normal and all_scores_attack:
        y_true = [0] * len(all_scores_normal) + [1] * len(all_scores_attack)
        y_scores = all_scores_normal + all_scores_attack
        auroc = roc_auc_score(y_true, y_scores)
        results['AUROC'] = float(auroc)
        print(f"  Overall AUROC: {auroc:.4f}")
    
    return results

def main():
    print("=" * 60)
    print("SCENARIO C EXPERIMENTS - AUROC EVALUATION")
    print("=" * 60)
    
    all_results = {}
    
    for exp_key, exp_info in EXPERIMENTS.items():
        print(f"\nEvaluating {exp_key} ({exp_info['name']})...")
        results = evaluate_experiment(exp_key, exp_info)
        if results:
            all_results[exp_key] = results
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<25} {'AUROC':<10}")
    print("-" * 35)
    
    for exp_key, results in all_results.items():
        auroc = results.get('AUROC', 'N/A')
        if isinstance(auroc, float):
            print(f"{exp_key:<25} {auroc:.4f}")
        else:
            print(f"{exp_key:<25} {auroc}")
    
    # Save results
    out_path = os.path.join(PROJECT_ROOT, "experiments", "experiment_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
