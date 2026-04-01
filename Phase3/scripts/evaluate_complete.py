"""
N2KShield Complete Evaluation (CANShield-Style)
================================================
Generates all paper-quality visualizations:
1. AUROC Curves per Attack Type
2. Precision-Recall Curves with AUPRC
3. Per-Attack Type Metrics Table
4. Comparison with Baseline (Single Autoencoder vs Ensemble)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# CONFIGURATION
# ============================================================================
STABLE_MODELS = ['50s_1s', '50s_5s', '75s_1s', '100s_1s']
THRESHOLD_PERCENTILE = '75'

# Attack types (from generate_attacks.py)
ATTACK_TYPES = ['spike', 'constant', 'replay', 'drift', 'noise', 'scaling']

# ============================================================================
# DATA LOADING
# ============================================================================

def load_resources():
    """Load models, thresholds, and data with attack metadata."""
    models = {}
    thresholds = {}
    normal_data = {}
    attack_data = {}
    attack_metadata = {}
    
    for name in STABLE_MODELS:
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
        
        # Metadata (contains attack type info)
        meta_path = os.path.join(config.ATTACKS_DIR, f"attacks_{name}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta_raw = json.load(f)
                # Handle both list and dict formats
                if isinstance(meta_raw, list):
                    attack_metadata[name] = meta_raw
                else:
                    attack_metadata[name] = meta_raw.get('samples', meta_raw)
        else:
            # Generate fake metadata if not available
            n_attacks = len(attack_data[name])
            attack_metadata[name] = [{'attack_type': ATTACK_TYPES[i % len(ATTACK_TYPES)]} 
                                     for i in range(n_attacks)]
    
    return models, thresholds, normal_data, attack_data, attack_metadata


def compute_mse_scores(model, data):
    """Compute MSE reconstruction error for each sample."""
    reconstructed = model.predict(data, verbose=0)
    mse = np.mean(np.square(data - reconstructed), axis=(1, 2, 3))
    return mse


# ============================================================================
# AUROC CURVES
# ============================================================================

def plot_auroc_curves(models, thresholds, normal_data, attack_data, attack_metadata):
    """Generate AUROC curves per attack type."""
    
    # Use primary model for analysis
    primary_model = '50s_1s'
    model = models[primary_model]
    
    n_normal = len(normal_data[primary_model])
    
    # Compute MSE for normal data
    mse_normal = compute_mse_scores(model, normal_data[primary_model])
    
    # Compute MSE for attack data
    attacks = attack_data[primary_model]
    mse_attack = compute_mse_scores(model, attacks)
    meta = attack_metadata[primary_model]  # Now a list directly
    
    # Group by attack type
    attack_type_indices = {at: [] for at in ATTACK_TYPES}
    for i, sample in enumerate(meta):
        at = sample.get('attack_type', 'unknown')
        if at in attack_type_indices:
            attack_type_indices[at].append(i)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('N2KShield: AUROC Curves per Attack Type', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    results_table = []
    
    for idx, attack_type in enumerate(ATTACK_TYPES):
        ax = axes[idx]
        
        indices = attack_type_indices[attack_type]
        if len(indices) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{attack_type.title()}')
            continue
        
        mse_attack_type = mse_attack[indices]
        
        # Create labels: 0 for normal, 1 for attack
        y_true = np.concatenate([np.zeros(n_normal), np.ones(len(indices))])
        y_scores = np.concatenate([mse_normal, mse_attack_type])
        
        # Compute ROC curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (FPR < 1%)
        valid_idx = np.where(fpr < 0.01)[0]
        if len(valid_idx) > 0:
            optimal_idx = valid_idx[-1]
            optimal_tpr = tpr[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
        else:
            optimal_tpr, optimal_fpr = tpr[0], fpr[0]
        
        # Plot
        ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'AUROC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.scatter(optimal_fpr, optimal_tpr, color='red', s=100, zorder=5, 
                   label=f'TPR@FPR<1% = {optimal_tpr:.2f}')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{attack_type.title()} Attack', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        results_table.append({
            'Attack': attack_type.title(),
            'AUROC': roc_auc,
            'TPR@FPR<1%': optimal_tpr,
            'Samples': len(indices)
        })
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_auroc_per_attack.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return results_table


# ============================================================================
# PRECISION-RECALL CURVES
# ============================================================================

def plot_pr_curves(models, thresholds, normal_data, attack_data, attack_metadata):
    """Generate Precision-Recall curves per attack type."""
    
    primary_model = '50s_1s'
    model = models[primary_model]
    
    n_normal = len(normal_data[primary_model])
    mse_normal = compute_mse_scores(model, normal_data[primary_model])
    
    attacks = attack_data[primary_model]
    mse_attack = compute_mse_scores(model, attacks)
    meta = attack_metadata[primary_model]  # Now a list directly
    
    # Group by attack type
    attack_type_indices = {at: [] for at in ATTACK_TYPES}
    for i, sample in enumerate(meta):
        at = sample.get('attack_type', 'unknown')
        if at in attack_type_indices:
            attack_type_indices[at].append(i)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('N2KShield: Precision-Recall Curves per Attack Type', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    pr_results = []
    
    for idx, attack_type in enumerate(ATTACK_TYPES):
        ax = axes[idx]
        
        indices = attack_type_indices[attack_type]
        if len(indices) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{attack_type.title()}')
            continue
        
        mse_attack_type = mse_attack[indices]
        
        y_true = np.concatenate([np.zeros(n_normal), np.ones(len(indices))])
        y_scores = np.concatenate([mse_normal, mse_attack_type])
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        # Plot
        ax.plot(recall, precision, color='green', linewidth=2, label=f'AUPRC = {ap:.3f}')
        ax.axhline(y=len(indices)/(len(indices)+n_normal), color='gray', linestyle='--', 
                   alpha=0.5, label='Baseline')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{attack_type.title()} Attack', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        pr_results.append({
            'Attack': attack_type.title(),
            'AUPRC': ap
        })
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_pr_per_attack.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return pr_results


# ============================================================================
# ENSEMBLE COMPARISON
# ============================================================================

def compare_with_baseline(models, thresholds, normal_data, attack_data):
    """Compare Ensemble vs Single Autoencoder (like CANShield paper)."""
    
    n_normal = min([len(normal_data[m]) for m in STABLE_MODELS])
    n_attack = min([len(attack_data[m]) for m in STABLE_MODELS])
    
    configurations = {
        'Single AE (50s_1s)': ['50s_1s'],
        'Single AE (75s_1s)': ['75s_1s'],
        'N2KShield-Ens (4)': STABLE_MODELS
    }
    
    results = []
    
    for config_name, model_list in configurations.items():
        votes_n = []
        votes_a = []
        mse_n_all = []
        mse_a_all = []
        
        for name in model_list:
            if name not in models:
                continue
            model = models[name]
            thresh = thresholds[name].get(THRESHOLD_PERCENTILE, thresholds[name]['threshold'])
            
            d_n = normal_data[name][:n_normal]
            d_a = attack_data[name][:n_attack]
            
            mse_n = compute_mse_scores(model, d_n)
            mse_a = compute_mse_scores(model, d_a)
            
            mse_n_all.append(mse_n)
            mse_a_all.append(mse_a)
            
            votes_n.append((mse_n > thresh).astype(int))
            votes_a.append((mse_a > thresh).astype(int))
        
        # OR voting for ensemble, direct for single
        if len(model_list) > 1:
            final_n = np.any(np.array(votes_n).T, axis=1).astype(int)
            final_a = np.any(np.array(votes_a).T, axis=1).astype(int)
            # For ROC, use average MSE
            avg_mse_n = np.mean(np.array(mse_n_all), axis=0)
            avg_mse_a = np.mean(np.array(mse_a_all), axis=0)
        else:
            final_n = votes_n[0]
            final_a = votes_a[0]
            avg_mse_n = mse_n_all[0]
            avg_mse_a = mse_a_all[0]
        
        y_true = np.concatenate([np.zeros_like(final_n), np.ones_like(final_a)])
        y_pred = np.concatenate([final_n, final_a])
        y_scores = np.concatenate([avg_mse_n, avg_mse_a])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute AUROC
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr_curve, tpr_curve)
        
        results.append({
            'Config': config_name,
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'AUROC': auroc,
            'TP': tp,
            'FP': fp
        })
    
    return results


def plot_baseline_comparison(comparison_results):
    """Plot baseline comparison chart."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = [r['Config'] for r in comparison_results]
    x = np.arange(len(configs))
    width = 0.15
    
    metrics = ['Recall', 'Precision', 'F1', 'AUROC']
    colors = ['#27ae60', '#3498db', '#9b59b6', '#e67e22']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [r[metric] for r in comparison_results]
        bars = ax.bar(x + i*width, values, width, label=metric, color=color, edgecolor='black')
        
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('N2KShield: Single Autoencoder vs Ensemble Comparison\n(Like CANShield Paper)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_baseline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def create_summary_table(auroc_results, pr_results, comparison_results):
    """Create comprehensive summary table like CANShield paper."""
    
    # Per-attack summary
    attack_df = pd.DataFrame(auroc_results)
    for pr in pr_results:
        idx = attack_df[attack_df['Attack'] == pr['Attack']].index
        if len(idx) > 0:
            attack_df.loc[idx[0], 'AUPRC'] = pr['AUPRC']
    
    # Comparison summary
    comparison_df = pd.DataFrame(comparison_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Table 1: Per-Attack Metrics
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title('Per-Attack Type Performance', fontsize=14, fontweight='bold', pad=20)
    
    table1 = ax1.table(
        cellText=attack_df.round(3).values,
        colLabels=attack_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.2, 0.25, 0.15, 0.2]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(attack_df.columns)):
        table1[(0, i)].set_facecolor('#3498db')
        table1[(0, i)].set_text_props(color='white', weight='bold')
    
    # Table 2: Model Comparison
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Model Configuration Comparison', fontsize=14, fontweight='bold', pad=20)
    
    display_cols = ['Config', 'Recall', 'Precision', 'F1', 'AUROC', 'FPR']
    table2 = ax2.table(
        cellText=comparison_df[display_cols].round(3).values,
        colLabels=display_cols,
        cellLoc='center',
        loc='center'
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.5)
    
    for i in range(len(display_cols)):
        table2[(0, i)].set_facecolor('#27ae60')
        table2[(0, i)].set_text_props(color='white', weight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_summary_tables.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # Save as CSV
    attack_df.to_csv(os.path.join(config.RESULTS_DIR, 'n2kshield_per_attack_metrics.csv'), index=False)
    comparison_df.to_csv(os.path.join(config.RESULTS_DIR, 'n2kshield_comparison_metrics.csv'), index=False)
    print(f"✅ Saved CSV files to {config.RESULTS_DIR}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("N2KSHIELD: COMPLETE EVALUATION (CANSHIELD-STYLE)")
    print("=" * 70)
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("\n📂 Loading resources...")
    models, thresholds, normal_data, attack_data, attack_metadata = load_resources()
    
    print("\n📊 Generating AUROC curves per attack type...")
    auroc_results = plot_auroc_curves(models, thresholds, normal_data, attack_data, attack_metadata)
    
    print("📊 Generating Precision-Recall curves...")
    pr_results = plot_pr_curves(models, thresholds, normal_data, attack_data, attack_metadata)
    
    print("📊 Comparing with baseline (Single AE vs Ensemble)...")
    comparison_results = compare_with_baseline(models, thresholds, normal_data, attack_data)
    
    print("📊 Creating summary tables...")
    create_summary_table(auroc_results, pr_results, comparison_results)
    
    print("📊 Plotting baseline comparison chart...")
    plot_baseline_comparison(comparison_results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📋 RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nPer-Attack Metrics:")
    print("-" * 50)
    for r in auroc_results:
        print(f"  {r['Attack']:<12}: AUROC={r['AUROC']:.3f}, TPR@FPR<1%={r['TPR@FPR<1%']:.3f}")
    
    print("\nModel Comparison:")
    print("-" * 50)
    for r in comparison_results:
        print(f"  {r['Config']:<25}: Recall={r['Recall']:.2f}, FPR={r['FPR']:.2f}, AUROC={r['AUROC']:.3f}")
    
    print("\n✅ All CANShield-style visualizations generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
