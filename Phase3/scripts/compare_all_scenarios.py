#!/usr/bin/env python3
"""
Compare All Three Aggregation Scenarios
========================================
Compares:
- Scenario A: 4 aggregations (mean/max/min/std) = 60 features
- Scenario B: Mean only = 15 features  
- Scenario C: Mean + Std = 30 features

With two detection methods:
- Simple MSE threshold
- Three-Step CANShield threshold
"""

import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Phase3
PROJECT_DIR = os.path.dirname(BASE_DIR)
PHASE1_DIR = os.path.join(PROJECT_DIR, "Phase1")
PHASE2_DIR = os.path.join(PROJECT_DIR, "Phase2")

# Scenarios configuration
SCENARIOS = {
    'A_baseline': {
        'name': '4 Aggregations (60 features)',
        'data_dir': os.path.join(PHASE1_DIR, "data"),
        'models_dir': os.path.join(PHASE2_DIR, "models"),
        'num_features': 60,
        'skip_col': 1  # Skip timestamp column (data has 61 cols)
    },
    'B_mean_only': {
        'name': 'Mean Only (15 features)',
        'data_dir': os.path.join(PHASE1_DIR, "data_B_mean_only"),
        'models_dir': os.path.join(PHASE2_DIR, "models_B_mean_only"),
        'num_features': 15,
        'skip_col': 0  # No timestamp column in this data
    },
    'C_mean_std': {
        'name': 'Mean + Std (30 features)',
        'data_dir': os.path.join(PHASE1_DIR, "data_C_mean_std"),
        'models_dir': os.path.join(PHASE2_DIR, "models_C_mean_std"),
        'num_features': 30,
        'skip_col': 0  # No timestamp column in this data
    }
}

# Model configs to test
MODEL_CONFIGS = [(50, 1), (75, 1), (100, 1)]  # Focus on 1s sampling


def load_scenario_data(scenario_key, window_size, sampling_period):
    """Load model and data for a specific scenario."""
    scenario = SCENARIOS[scenario_key]
    config_name = f"{window_size}s_{sampling_period}s"
    
    # Load model
    model_path = os.path.join(scenario['models_dir'], f"{config_name}.h5")
    if not os.path.exists(model_path):
        return None, None, None
    model = load_model(model_path, compile=False)
    
    # Load train and test data
    train_path = os.path.join(
        scenario['data_dir'], 
        f"{window_size}s_window", f"sampling_{sampling_period}s", "train.npy"
    )
    test_path = os.path.join(
        scenario['data_dir'],
        f"{window_size}s_window", f"sampling_{sampling_period}s", "test.npy"
    )
    
    if not os.path.exists(test_path):
        return None, None, None
    
    train_raw = np.load(train_path, allow_pickle=True)
    test_raw = np.load(test_path, allow_pickle=True)
    
    # Skip timestamp column if needed, convert to float32, reshape
    skip = scenario['skip_col']
    if skip > 0:
        train_data = train_raw[:, :, skip:].astype(np.float32).reshape(train_raw.shape[0], train_raw.shape[1], -1, 1)
        test_data = test_raw[:, :, skip:].astype(np.float32).reshape(test_raw.shape[0], test_raw.shape[1], -1, 1)
    else:
        train_data = train_raw.astype(np.float32).reshape(train_raw.shape[0], train_raw.shape[1], -1, 1)
        test_data = test_raw.astype(np.float32).reshape(test_raw.shape[0], test_raw.shape[1], -1, 1)
    
    return model, train_data, test_data


def generate_attacks_for_scenario(normal_data, num_attacks=50):
    """Generate simple synthetic attacks within a scenario."""
    attacks = []
    n_samples = len(normal_data)
    n_features = normal_data.shape[2]  # Get actual feature count
    
    for _ in range(num_attacks):
        # Pick random normal sample
        idx = np.random.randint(0, n_samples)
        sample = normal_data[idx].copy()
        
        # Apply random attack
        attack_type = np.random.choice(['spike', 'constant', 'noise', 'drift'])
        feature_idx = np.random.randint(0, n_features)  # Use actual feature count
        
        if attack_type == 'spike':
            # Sudden value jump
            start = np.random.randint(0, max(1, sample.shape[0] - 5))
            end = min(start + 5, sample.shape[0])
            sample[start:end, feature_idx, 0] += np.random.choice([-1, 1]) * 0.5
        elif attack_type == 'constant':
            # Freeze at extreme value
            start = np.random.randint(0, max(1, sample.shape[0] // 2))
            sample[start:, feature_idx, 0] = np.random.choice([0.0, 1.0])
        elif attack_type == 'noise':
            # Add noise  
            sample[:, feature_idx, 0] += np.random.normal(0, 0.3, sample.shape[0])
        elif attack_type == 'drift':
            # Gradual drift
            drift = np.linspace(0, 0.5, sample.shape[0])
            sample[:, feature_idx, 0] += drift
        
        attacks.append(sample)
    
    return np.array(attacks, dtype=np.float32)


def evaluate_simple_mse(model, normal_data, attack_data, percentile=75):
    """Evaluate using simple MSE threshold."""
    # Get MSE for normal data
    recon_normal = model.predict(normal_data, verbose=0)
    mse_normal = np.mean(np.square(normal_data - recon_normal), axis=(1, 2, 3))
    threshold = np.percentile(mse_normal, percentile)
    
    # Get MSE for attack data
    recon_attack = model.predict(attack_data, verbose=0)
    mse_attack = np.mean(np.square(attack_data - recon_attack), axis=(1, 2, 3))
    
    # Predictions
    pred_normal = (mse_normal > threshold).astype(int)
    pred_attack = (mse_attack > threshold).astype(int)
    
    # Metrics
    y_true = np.concatenate([np.zeros(len(normal_data)), np.ones(len(attack_data))])
    y_pred = np.concatenate([pred_normal, pred_attack])
    y_scores = np.concatenate([mse_normal, mse_attack])
    
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': np.mean(pred_normal),
        'auroc': roc_auc_score(y_true, y_scores) if len(np.unique(y_scores)) > 1 else 0.5
    }


def evaluate_three_step(model, normal_data, attack_data, p_loss=95, p_time=99, r_signal=0.15):
    """Evaluate using CANShield three-step threshold."""
    # Step 1: Get reconstruction loss for normal data
    recon_normal = model.predict(normal_data, verbose=0)
    L_normal = np.abs(normal_data - recon_normal)[:, :, :, 0]
    
    # Calculate R_Loss per feature
    R_Loss = np.percentile(L_normal, p_loss, axis=(0, 1))
    
    # Calculate R_Time
    B_normal = (L_normal > R_Loss).astype(int)
    V_normal = np.sum(B_normal, axis=1)
    R_Time = np.percentile(V_normal, p_time, axis=0)
    
    # Evaluate normal data
    S_normal = (V_normal > R_Time).astype(int)
    P_normal = np.mean(S_normal, axis=1)
    pred_normal = (P_normal > r_signal).astype(int)
    
    # Evaluate attack data
    recon_attack = model.predict(attack_data, verbose=0)
    L_attack = np.abs(attack_data - recon_attack)[:, :, :, 0]
    B_attack = (L_attack > R_Loss).astype(int)
    V_attack = np.sum(B_attack, axis=1)
    S_attack = (V_attack > R_Time).astype(int)
    P_attack = np.mean(S_attack, axis=1)
    pred_attack = (P_attack > r_signal).astype(int)
    
    # Metrics
    y_true = np.concatenate([np.zeros(len(normal_data)), np.ones(len(attack_data))])
    y_pred = np.concatenate([pred_normal, pred_attack])
    y_scores = np.concatenate([P_normal, P_attack])
    
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': np.mean(pred_normal),
        'auroc': roc_auc_score(y_true, y_scores) if len(np.unique(y_scores)) > 1 else 0.5
    }


def main():
    print("=" * 70)
    print("COMPARISON: All 3 Aggregation Scenarios × 2 Detection Methods")
    print("=" * 70)
    
    results = []
    
    for scenario_key, scenario_info in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_info['name']}")
        print(f"{'='*60}")
        
        scenario_results = {'scenario': scenario_key, 'name': scenario_info['name']}
        mse_metrics = []
        three_step_metrics = []
        
        for ws, sp in MODEL_CONFIGS:
            config_name = f"{ws}s_{sp}s"
            
            model, train_data, test_data = load_scenario_data(scenario_key, ws, sp)
            
            if model is None:
                print(f"  ⚠️ {config_name}: Model or data not found, skipping...")
                continue
            
            print(f"\n  📂 {config_name} - {len(test_data)} test samples")
            
            # Generate attacks for this scenario's feature space
            attack_data = generate_attacks_for_scenario(test_data, num_attacks=100)
            print(f"     Generated {len(attack_data)} attacks")
            
            # Evaluate Simple MSE
            mse_result = evaluate_simple_mse(model, test_data, attack_data)
            mse_metrics.append(mse_result)
            print(f"     Simple MSE:  Recall={mse_result['recall']:.2f}, FPR={mse_result['fpr']:.2f}, AUROC={mse_result['auroc']:.3f}")
            
            # Evaluate Three-Step
            ts_result = evaluate_three_step(model, test_data, attack_data)
            three_step_metrics.append(ts_result)
            print(f"     Three-Step:  Recall={ts_result['recall']:.2f}, FPR={ts_result['fpr']:.2f}, AUROC={ts_result['auroc']:.3f}")
        
        if mse_metrics:
            scenario_results['mse_avg'] = {
                'recall': np.mean([m['recall'] for m in mse_metrics]),
                'fpr': np.mean([m['fpr'] for m in mse_metrics]),
                'auroc': np.mean([m['auroc'] for m in mse_metrics]),
                'f1': np.mean([m['f1'] for m in mse_metrics])
            }
            scenario_results['three_step_avg'] = {
                'recall': np.mean([m['recall'] for m in three_step_metrics]),
                'fpr': np.mean([m['fpr'] for m in three_step_metrics]),
                'auroc': np.mean([m['auroc'] for m in three_step_metrics]),
                'f1': np.mean([m['f1'] for m in three_step_metrics])
            }
            results.append(scenario_results)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    print("\n### Simple MSE Method:")
    print(f"{'Scenario':<35} {'Recall':>10} {'FPR':>10} {'AUROC':>10}")
    print("-" * 65)
    for r in results:
        m = r['mse_avg']
        print(f"{r['name']:<35} {m['recall']:>10.2f} {m['fpr']:>10.2f} {m['auroc']:>10.3f}")
    
    print("\n### Three-Step Method:")
    print(f"{'Scenario':<35} {'Recall':>10} {'FPR':>10} {'AUROC':>10}")
    print("-" * 65)
    for r in results:
        m = r['three_step_avg']
        print(f"{r['name']:<35} {m['recall']:>10.2f} {m['fpr']:>10.2f} {m['auroc']:>10.3f}")
    
    print("\n" + "=" * 70)
    
    # Determine best
    if results:
        best_mse = max(results, key=lambda x: x['mse_avg']['auroc'])
        best_ts = max(results, key=lambda x: x['three_step_avg']['auroc'])
        print(f"\n🏆 Best with Simple MSE: {best_mse['name']} (AUROC={best_mse['mse_avg']['auroc']:.3f})")
        print(f"🏆 Best with Three-Step: {best_ts['name']} (AUROC={best_ts['three_step_avg']['auroc']:.3f})")


if __name__ == "__main__":
    main()
