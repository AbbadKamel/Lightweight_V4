"""
N2KShield: Hybrid Maritime Intrusion Detection Evaluation
=========================================================
Combines CANShield multi-level thresholding with maritime-specific adaptations:
1. Signal Groups by Criticality (GPS, Steering, Support)
2. Weighted Expert Voting (reduces unstable model influence)
3. Per-Signal Threshold Calculation
4. Comparison with baseline OR voting
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# SIGNAL GROUPS BY CRITICALITY
# ============================================================================
# Signal order in our data (15 signals × 4 aggregations = 60 features)
SIGNAL_NAMES = [
    'wind_speed', 'wind_angle', 'yaw', 'cog', 'heading',
    'roll', 'rudder_angle_order', 'rudder_position', 'rate_of_turn',
    'depth', 'variation', 'latitude', 'longitude', 'pitch', 'sog'
]

# Critical groups with weights
SIGNAL_GROUPS = {
    'CRITICAL': {
        'signals': ['latitude', 'longitude', 'sog', 'cog', 'heading', 'rudder_position'],
        'weight': 3.0,
        'indices': []  # Will be computed
    },
    'IMPORTANT': {
        'signals': ['depth', 'wind_speed', 'wind_angle'],
        'weight': 2.0,
        'indices': []
    },
    'SUPPORT': {
        'signals': ['yaw', 'pitch', 'roll', 'rate_of_turn', 'variation', 'rudder_angle_order'],
        'weight': 1.0,
        'indices': []
    }
}

# Compute feature indices for each group (each signal has 4 features: mean, max, min, std)
for group_name, group_data in SIGNAL_GROUPS.items():
    indices = []
    for signal in group_data['signals']:
        if signal in SIGNAL_NAMES:
            base_idx = SIGNAL_NAMES.index(signal) * 4
            indices.extend([base_idx, base_idx+1, base_idx+2, base_idx+3])
    group_data['indices'] = indices

# ============================================================================
# EXPERT MODEL WEIGHTS (reduce unstable models)
# ============================================================================
EXPERT_WEIGHTS = {
    '50s_1s': 1.0,   # Good FPR (3.9%)
    '50s_5s': 1.0,   # Perfect FPR (0%)
    '100s_10s': 0.3  # High FPR (85.7%) - reduce influence
}

SELECTED_MODELS = list(EXPERT_WEIGHTS.keys())

# ============================================================================
# THRESHOLDS
# ============================================================================
THRESHOLD_STRATEGY = '75'  # Use 75th percentile (more sensitive)

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_resources():
    """Load models, thresholds, and data."""
    models = {}
    thresholds = {}
    normal_data = {}
    attack_data = {}
    
    for name in SELECTED_MODELS:
        # Load model
        model_path = os.path.join(config.PHASE2_DIR, "models", f"{name}.h5")
        models[name] = load_model(model_path, compile=False)
        
        # Load threshold
        thresh_path = os.path.join(config.THRESHOLDS_DIR, f"{name}_thresholds.json")
        with open(thresh_path) as f:
            thresholds[name] = json.load(f)
        
        # Parse config name
        parts = name.split('_')
        ts, sp = int(parts[0][:-1]), int(parts[1][:-1])
        
        # Load normal data
        normal_path = os.path.join(config.TEST_DATA_DIR, f"{ts}s_window", f"sampling_{sp}s", "test.npy")
        d_n = np.load(normal_path, allow_pickle=True)
        normal_data[name] = d_n[:, :, 1:].astype(np.float32).reshape(d_n.shape[0], d_n.shape[1], -1, 1)
        
        # Load attack data
        attack_path = os.path.join(config.ATTACKS_DIR, f"attacks_{name}.npy")
        attack_data[name] = np.load(attack_path)
    
    return models, thresholds, normal_data, attack_data


def compute_per_signal_mse(original, reconstructed):
    """
    Compute MSE per signal group.
    Returns dict with MSE for each group.
    """
    # MSE per feature: (samples, time, features, 1) -> (samples, features)
    mse_per_feature = np.mean(np.square(original - reconstructed), axis=(1, 3))  # (samples, 60)
    
    group_mse = {}
    for group_name, group_data in SIGNAL_GROUPS.items():
        indices = group_data['indices']
        if len(indices) > 0:
            group_mse[group_name] = np.mean(mse_per_feature[:, indices], axis=1)  # (samples,)
        else:
            group_mse[group_name] = np.zeros(mse_per_feature.shape[0])
    
    return group_mse


def evaluate_baseline_or(models, thresholds, normal_data, attack_data):
    """
    Baseline: Simple OR voting with threshold.
    If ANY model detects anomaly -> ANOMALY
    """
    n_limit = min([len(attack_data[m]) for m in models])
    n_normal = min([len(normal_data[m]) for m in models])
    
    votes_normal = []
    votes_attack = []
    
    for name, model in models.items():
        thresh = thresholds[name][THRESHOLD_STRATEGY]
        
        # Normal
        d_n = normal_data[name][:n_normal]
        rec_n = model.predict(d_n, verbose=0)
        mse_n = np.mean(np.square(d_n - rec_n), axis=(1, 2, 3))
        votes_normal.append((mse_n > thresh).astype(int))
        
        # Attack
        d_a = attack_data[name][:n_limit]
        rec_a = model.predict(d_a, verbose=0)
        mse_a = np.mean(np.square(d_a - rec_a), axis=(1, 2, 3))
        votes_attack.append((mse_a > thresh).astype(int))
    
    # OR voting
    final_n = np.any(np.array(votes_normal).T, axis=1).astype(int)
    final_a = np.any(np.array(votes_attack).T, axis=1).astype(int)
    
    y_true = np.concatenate([np.zeros_like(final_n), np.ones_like(final_a)])
    y_pred = np.concatenate([final_n, final_a])
    
    return compute_metrics(y_true, y_pred, "Baseline OR")


def evaluate_n2kshield_weighted(models, thresholds, normal_data, attack_data):
    """
    N2KShield: Weighted voting with reduced influence for unstable models.
    """
    n_limit = min([len(attack_data[m]) for m in models])
    n_normal = min([len(normal_data[m]) for m in models])
    
    scores_normal = []
    scores_attack = []
    
    for name, model in models.items():
        weight = EXPERT_WEIGHTS[name]
        thresh = thresholds[name][THRESHOLD_STRATEGY]
        
        # Normal
        d_n = normal_data[name][:n_normal]
        rec_n = model.predict(d_n, verbose=0)
        mse_n = np.mean(np.square(d_n - rec_n), axis=(1, 2, 3))
        scores_normal.append((mse_n > thresh).astype(float) * weight)
        
        # Attack
        d_a = attack_data[name][:n_limit]
        rec_a = model.predict(d_a, verbose=0)
        mse_a = np.mean(np.square(d_a - rec_a), axis=(1, 2, 3))
        scores_attack.append((mse_a > thresh).astype(float) * weight)
    
    # Weighted score: sum of weighted votes / sum of weights
    total_weight = sum(EXPERT_WEIGHTS.values())
    score_n = np.sum(np.array(scores_normal).T, axis=1) / total_weight
    score_a = np.sum(np.array(scores_attack).T, axis=1) / total_weight
    
    # Threshold at 0.5 (majority weighted voting)
    final_n = (score_n > 0.5).astype(int)
    final_a = (score_a > 0.5).astype(int)
    
    y_true = np.concatenate([np.zeros_like(final_n), np.ones_like(final_a)])
    y_pred = np.concatenate([final_n, final_a])
    
    return compute_metrics(y_true, y_pred, "N2KShield Weighted")


def evaluate_n2kshield_groups(models, thresholds, normal_data, attack_data):
    """
    N2KShield with Signal Groups: Weight MSE by signal criticality.
    """
    n_limit = min([len(attack_data[m]) for m in models])
    n_normal = min([len(normal_data[m]) for m in models])
    
    # Use only the best model for group analysis (50s_1s has good resolution)
    best_model = '50s_1s'
    model = models[best_model]
    thresh_global = thresholds[best_model][THRESHOLD_STRATEGY]
    
    # Normal
    d_n = normal_data[best_model][:n_normal]
    rec_n = model.predict(d_n, verbose=0)
    group_mse_n = compute_per_signal_mse(d_n, rec_n)
    
    # Attack
    d_a = attack_data[best_model][:n_limit]
    rec_a = model.predict(d_a, verbose=0)
    group_mse_a = compute_per_signal_mse(d_a, rec_a)
    
    # Weighted score by group
    total_weight = sum(g['weight'] for g in SIGNAL_GROUPS.values())
    
    score_n = np.zeros(n_normal)
    score_a = np.zeros(n_limit)
    
    for group_name, group_data in SIGNAL_GROUPS.items():
        weight = group_data['weight']
        # Use a fraction of global threshold per group (scaled by expected contribution)
        group_thresh = thresh_global / 3  # Split threshold across groups
        
        score_n += (group_mse_n[group_name] > group_thresh).astype(float) * weight
        score_a += (group_mse_a[group_name] > group_thresh).astype(float) * weight
    
    score_n /= total_weight
    score_a /= total_weight
    
    # Threshold at 0.3 (if critical group is anomalous, that's enough)
    final_n = (score_n > 0.3).astype(int)
    final_a = (score_a > 0.3).astype(int)
    
    y_true = np.concatenate([np.zeros_like(final_n), np.ones_like(final_a)])
    y_pred = np.concatenate([final_n, final_a])
    
    return compute_metrics(y_true, y_pred, "N2KShield Groups")


def evaluate_n2kshield_hybrid(models, thresholds, normal_data, attack_data):
    """
    N2KShield Hybrid: Combines weighted voting + group analysis.
    """
    n_limit = min([len(attack_data[m]) for m in models])
    n_normal = min([len(normal_data[m]) for m in models])
    
    # Part 1: Expert weighted votes
    expert_scores_n = []
    expert_scores_a = []
    
    for name, model in models.items():
        weight = EXPERT_WEIGHTS[name]
        thresh = thresholds[name][THRESHOLD_STRATEGY]
        
        d_n = normal_data[name][:n_normal]
        rec_n = model.predict(d_n, verbose=0)
        mse_n = np.mean(np.square(d_n - rec_n), axis=(1, 2, 3))
        expert_scores_n.append((mse_n > thresh).astype(float) * weight)
        
        d_a = attack_data[name][:n_limit]
        rec_a = model.predict(d_a, verbose=0)
        mse_a = np.mean(np.square(d_a - rec_a), axis=(1, 2, 3))
        expert_scores_a.append((mse_a > thresh).astype(float) * weight)
    
    total_expert_weight = sum(EXPERT_WEIGHTS.values())
    expert_score_n = np.sum(np.array(expert_scores_n).T, axis=1) / total_expert_weight
    expert_score_a = np.sum(np.array(expert_scores_a).T, axis=1) / total_expert_weight
    
    # Part 2: Group analysis (using best model)
    best_model = '50s_1s'
    model = models[best_model]
    thresh_global = thresholds[best_model][THRESHOLD_STRATEGY]
    
    d_n = normal_data[best_model][:n_normal]
    rec_n = model.predict(d_n, verbose=0)
    group_mse_n = compute_per_signal_mse(d_n, rec_n)
    
    d_a = attack_data[best_model][:n_limit]
    rec_a = model.predict(d_a, verbose=0)
    group_mse_a = compute_per_signal_mse(d_a, rec_a)
    
    # Check if CRITICAL group is anomalous
    critical_thresh = thresh_global / 2
    critical_anomaly_n = (group_mse_n['CRITICAL'] > critical_thresh).astype(float)
    critical_anomaly_a = (group_mse_a['CRITICAL'] > critical_thresh).astype(float)
    
    # Hybrid decision: 
    # - If expert score > 0.5 AND critical group anomalous -> HIGH CONFIDENCE
    # - If expert score > 0.7 (strong consensus) -> MEDIUM CONFIDENCE
    # - If critical group strongly anomalous (>2x threshold) -> MEDIUM CONFIDENCE
    
    final_n = ((expert_score_n > 0.5) & (critical_anomaly_n > 0.5)).astype(int)
    final_n |= (expert_score_n > 0.7).astype(int)
    final_n |= (group_mse_n['CRITICAL'] > critical_thresh * 2).astype(int)
    
    final_a = ((expert_score_a > 0.5) & (critical_anomaly_a > 0.5)).astype(int)
    final_a |= (expert_score_a > 0.7).astype(int)
    final_a |= (group_mse_a['CRITICAL'] > critical_thresh * 2).astype(int)
    
    y_true = np.concatenate([np.zeros_like(final_n), np.ones_like(final_a)])
    y_pred = np.concatenate([final_n, final_a])
    
    return compute_metrics(y_true, y_pred, "N2KShield Hybrid")


def compute_metrics(y_true, y_pred, method_name):
    """Compute and return metrics dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'method': method_name,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def main():
    print("=" * 70)
    print("N2KSHIELD EVALUATION: Comparing Detection Methods")
    print("=" * 70)
    
    print("\n📂 Loading models and data...")
    models, thresholds, normal_data, attack_data = load_resources()
    
    n_normal = min([len(normal_data[m]) for m in models])
    n_attack = min([len(attack_data[m]) for m in models])
    print(f"   Normal samples: {n_normal}")
    print(f"   Attack samples: {n_attack}")
    print(f"   Threshold strategy: {THRESHOLD_STRATEGY}th percentile")
    
    results = []
    
    # Method 1: Baseline OR voting
    print("\n🔍 Evaluating Baseline OR Voting...")
    results.append(evaluate_baseline_or(models, thresholds, normal_data, attack_data))
    
    # Method 2: N2KShield Weighted voting
    print("🔍 Evaluating N2KShield Weighted Voting...")
    results.append(evaluate_n2kshield_weighted(models, thresholds, normal_data, attack_data))
    
    # Method 3: N2KShield with Signal Groups
    print("🔍 Evaluating N2KShield Signal Groups...")
    results.append(evaluate_n2kshield_groups(models, thresholds, normal_data, attack_data))
    
    # Method 4: N2KShield Hybrid
    print("🔍 Evaluating N2KShield Hybrid...")
    results.append(evaluate_n2kshield_hybrid(models, thresholds, normal_data, attack_data))
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<25} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'FPR':<8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['method']:<25} | {r['recall']:.4f}   | {r['precision']:.4f}     | {r['f1']:.4f}   | {r['fpr']:.4f}")
    
    print("-" * 70)
    
    # Highlight best
    best_f1 = max(results, key=lambda x: x['f1'])
    best_fpr = min(results, key=lambda x: x['fpr'])
    
    print(f"\n🏆 Best F1 Score: {best_f1['method']} ({best_f1['f1']:.4f})")
    print(f"🛡️ Lowest FPR: {best_fpr['method']} ({best_fpr['fpr']:.4f})")
    
    # Confusion matrix details
    print("\n" + "=" * 70)
    print("📋 CONFUSION MATRIX DETAILS")
    print("=" * 70)
    for r in results:
        print(f"\n{r['method']}:")
        print(f"   TP={r['tp']}, FP={r['fp']}, TN={r['tn']}, FN={r['fn']}")
    
    # Save results
    output_path = os.path.join(config.RESULTS_DIR, "n2kshield_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
