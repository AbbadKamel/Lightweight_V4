"""
N2KShield FINAL: Optimized Maritime Intrusion Detection
========================================================
Uses only stable models (excluding *_10s which have >85% FPR)
with 75th percentile threshold for optimal Recall/FPR balance.

Configuration:
- Models: 50s_1s, 50s_5s, 75s_1s, 100s_1s
- Threshold: 75th percentile
- Voting: OR (if ANY model detects anomaly → ATTACK)

Results:
- Recall: 66.3% (199/300 attacks detected)
- FPR: 0% (0 false alarms)
- Precision: 100%
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# OPTIMAL CONFIGURATION
# ============================================================================
# Only stable models (exclude *_10s which have 85-100% FPR)
STABLE_MODELS = ['50s_1s', '50s_5s', '75s_1s', '100s_1s']
THRESHOLD_PERCENTILE = '75'

# ============================================================================
# COMPARISON CONFIGURATIONS
# ============================================================================
CONFIGS = {
    'Baseline (All 9)': {
        'models': ['50s_1s', '50s_5s', '50s_10s', '75s_1s', '75s_5s', '75s_10s', 
                   '100s_1s', '100s_5s', '100s_10s'],
        'threshold': 'max'
    },
    'Baseline (3 Experts)': {
        'models': ['50s_1s', '50s_5s', '100s_10s'],
        'threshold': '75'
    },
    'N2KShield (4 Stable)': {
        'models': STABLE_MODELS,
        'threshold': '75'
    },
    'N2KShield (6 Models)': {
        'models': ['50s_1s', '50s_5s', '75s_1s', '75s_5s', '100s_1s', '100s_5s'],
        'threshold': '75'
    }
}


def load_all_resources():
    """Load all models, thresholds, and data."""
    all_model_names = list(set(
        m for cfg in CONFIGS.values() for m in cfg['models']
    ))
    
    models = {}
    thresholds = {}
    normal_data = {}
    attack_data = {}
    
    for name in all_model_names:
        try:
            # Model
            model_path = os.path.join(config.PHASE2_DIR, "models", f"{name}.h5")
            models[name] = load_model(model_path, compile=False)
            
            # Threshold
            thresh_path = os.path.join(config.THRESHOLDS_DIR, f"{name}_thresholds.json")
            with open(thresh_path) as f:
                thresholds[name] = json.load(f)
            
            # Data
            parts = name.split('_')
            ts, sp = int(parts[0][:-1]), int(parts[1][:-1])
            
            normal_path = os.path.join(config.TEST_DATA_DIR, f"{ts}s_window", f"sampling_{sp}s", "test.npy")
            d_n = np.load(normal_path, allow_pickle=True)
            normal_data[name] = d_n[:, :, 1:].astype(np.float32).reshape(d_n.shape[0], d_n.shape[1], -1, 1)
            
            attack_path = os.path.join(config.ATTACKS_DIR, f"attacks_{name}.npy")
            attack_data[name] = np.load(attack_path)
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")
    
    return models, thresholds, normal_data, attack_data


def evaluate_config(config_name, model_names, threshold_key, models, thresholds, normal_data, attack_data):
    """Evaluate a specific configuration."""
    # Get common sample sizes
    available_models = [m for m in model_names if m in models]
    n_normal = min([len(normal_data[m]) for m in available_models])
    n_attack = min([len(attack_data[m]) for m in available_models])
    
    votes_n = []
    votes_a = []
    
    for name in available_models:
        model = models[name]
        thresh = thresholds[name].get(threshold_key, thresholds[name]['threshold'])
        
        # Normal
        d_n = normal_data[name][:n_normal]
        rec_n = model.predict(d_n, verbose=0)
        mse_n = np.mean(np.square(d_n - rec_n), axis=(1, 2, 3))
        votes_n.append((mse_n > thresh).astype(int))
        
        # Attack
        d_a = attack_data[name][:n_attack]
        rec_a = model.predict(d_a, verbose=0)
        mse_a = np.mean(np.square(d_a - rec_a), axis=(1, 2, 3))
        votes_a.append((mse_a > thresh).astype(int))
    
    # OR voting
    final_n = np.any(np.array(votes_n).T, axis=1).astype(int)
    final_a = np.any(np.array(votes_a).T, axis=1).astype(int)
    
    y_true = np.concatenate([np.zeros_like(final_n), np.ones_like(final_a)])
    y_pred = np.concatenate([final_n, final_a])
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'name': config_name,
        'models': len(available_models),
        'threshold': threshold_key,
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


def plot_comparison(results):
    """Generate comparison plots."""
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Confusion Matrices (2x2 grid in top half)
    for idx, result in enumerate(results):
        ax = fig.add_subplot(2, 2, idx + 1)
        
        cm = np.array([[result['tn'], result['fp']],
                       [result['fn'], result['tp']]])
        
        # Color based on FPR
        cmap = 'Greens' if result['fpr'] < 0.1 else ('YlOrBr' if result['fpr'] < 0.5 else 'Reds')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    annot_kws={'size': 16, 'weight': 'bold'},
                    cbar=False)
        
        # Add border for best result
        if result['fpr'] == 0 and result['recall'] > 0.5:
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(4)
        
        ax.set_title(f"{result['name']}\nRecall: {result['recall']:.1%} | FPR: {result['fpr']:.1%}", 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_final_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # 2. Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.2
    
    metrics = ['recall', 'precision', 'f1', 'fpr']
    colors = ['#27ae60', '#3498db', '#9b59b6', '#e74c3c']
    labels = ['Recall (Attack Detection)', 'Precision', 'F1 Score', 'FPR (False Alarms)']
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i*width, values, width, label=label, color=color, edgecolor='black', alpha=0.85)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1%}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('N2KShield: Configuration Comparison\nGoal: High Recall + Zero FPR', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight best
    best_idx = next((i for i, r in enumerate(results) if r['fpr'] == 0 and r['recall'] > 0.5), None)
    if best_idx is not None:
        ax.axvspan(best_idx - 0.3, best_idx + 0.9, alpha=0.15, color='green')
        ax.text(best_idx + 0.3, 1.05, 'BEST', ha='center', fontsize=12, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_metrics_final.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # 3. Recall vs FPR Trade-off
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red', 'orange', 'green', 'blue']
    markers = ['o', 's', '*', 'D']
    
    for result, color, marker in zip(results, colors, markers):
        size = 500 if result['fpr'] == 0 else 300
        ax.scatter(result['fpr'], result['recall'], 
                   s=size, c=color, marker=marker, 
                   label=f"{result['name']} ({result['models']}m)", 
                   edgecolors='black', linewidth=2, alpha=0.8)
    
    # Ideal point
    ax.scatter(0, 1, s=600, c='gold', marker='*', 
               label='Ideal (0% FPR, 100% Recall)', 
               edgecolors='black', linewidth=2, zorder=10)
    
    # Good zone
    ax.axvspan(0, 0.1, alpha=0.1, color='green')
    ax.axhspan(0.5, 1.0, alpha=0.1, color='blue')
    
    # Annotations
    for result in results:
        offset = (15, -15) if result['fpr'] > 0.3 else (15, 10)
        ax.annotate(f"{result['name']}\n({result['recall']:.0%}, {result['fpr']:.0%})", 
                    xy=(result['fpr'], result['recall']),
                    xytext=offset, textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('False Positive Rate (FPR) →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (Attack Detection) →', fontsize=12, fontweight='bold')
    ax.set_title('N2KShield: Recall vs FPR Trade-off\nGoal: Top-Left Corner', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_tradeoff_final.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("N2KSHIELD FINAL EVALUATION")
    print("=" * 70)
    
    # Ensure output directory exists
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Load resources
    print("\n📂 Loading models and data...")
    models, thresholds, normal_data, attack_data = load_all_resources()
    print(f"   Loaded {len(models)} models")
    
    # Evaluate all configurations
    print("\n🔍 Evaluating configurations...")
    results = []
    
    for config_name, cfg in CONFIGS.items():
        print(f"   Testing: {config_name}...")
        result = evaluate_config(
            config_name, 
            cfg['models'], 
            cfg['threshold'],
            models, thresholds, normal_data, attack_data
        )
        results.append(result)
    
    # Print results table
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"\n{'Configuration':<25} | {'Recall':<8} | {'FPR':<8} | {'F1':<8} | {'TP':<6} | {'FP':<6}")
    print("-" * 70)
    
    for r in results:
        marker = '✅ BEST' if r['fpr'] == 0 and r['recall'] > 0.5 else ''
        print(f"{r['name']:<25} | {r['recall']:.1%}    | {r['fpr']:.1%}    | {r['f1']:.1%}    | {r['tp']:<6} | {r['fp']:<6} {marker}")
    
    print("-" * 70)
    
    # Generate plots
    print("\n📊 Generating plots...")
    plot_comparison(results)
    
    # Save results
    output_path = os.path.join(config.RESULTS_DIR, "n2kshield_final_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved: {output_path}")
    
    # Summary
    best = next((r for r in results if r['fpr'] == 0 and r['recall'] > 0.5), None)
    if best:
        print("\n" + "=" * 70)
        print("🏆 RECOMMENDED CONFIGURATION")
        print("=" * 70)
        print(f"   Name: {best['name']}")
        print(f"   Models: {best['models']} models")
        print(f"   Threshold: {best['threshold']}th percentile")
        print(f"   Recall: {best['recall']:.1%} ({best['tp']}/300 attacks detected)")
        print(f"   FPR: {best['fpr']:.1%} ({best['fp']} false alarms)")
        print(f"   F1 Score: {best['f1']:.1%}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
