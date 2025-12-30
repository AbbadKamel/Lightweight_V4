"""
Phase 3 - Step 5: Visualize Results
=====================================
Create publication-ready visualizations for detection results.

Visualizations:
1. ROC curves (per model and ensemble)
2. Precision-Recall curves
3. Confusion matrices
4. Error distributions (normal vs attack)
5. Model comparison bar charts
6. Attack type detection heatmap

Usage:
    python visualize_results.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

from config import (
    MODEL_CONFIGS, ATTACK_TYPES, RESULTS_DIR, FIGURES_DIR, THRESHOLDS_DIR,
    ensure_dirs
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_evaluation_results() -> Dict:
    """Load all evaluation results."""
    results = {}
    
    for window_size, sampling_period in MODEL_CONFIGS:
        config_name = f"{window_size}s_{sampling_period}s"
        results_path = os.path.join(RESULTS_DIR, f"{config_name}_evaluation.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results[config_name] = json.load(f)
    
    return results


def load_threshold_data() -> Dict:
    """Load threshold data for all models."""
    thresholds = {}
    
    for window_size, sampling_period in MODEL_CONFIGS:
        config_name = f"{window_size}s_{sampling_period}s"
        threshold_path = os.path.join(THRESHOLDS_DIR, f"{config_name}_thresholds.json")
        
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                thresholds[config_name] = json.load(f)
    
    return thresholds


def plot_error_distributions(thresholds: Dict, save_path: str):
    """
    Plot reconstruction error distributions for all models.
    
    Shows the distribution of errors on normal data with threshold markers.
    """
    n_models = len(thresholds)
    if n_models == 0:
        print("No threshold data available")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (config_name, data) in enumerate(sorted(thresholds.items())):
        if idx >= 9:
            break
            
        ax = axes[idx]
        errors = np.array(data.get('all_errors', []))
        
        if len(errors) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config_name)
            continue
        
        # Plot histogram
        ax.hist(errors, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add threshold lines
        thresh_95 = data['thresholds'].get('95', data['thresholds'].get(95))
        thresh_99 = data['thresholds'].get('99', data['thresholds'].get(99))
        
        if thresh_95:
            ax.axvline(thresh_95, color='orange', linestyle='--', linewidth=2, label=f'95th: {thresh_95:.4f}')
        if thresh_99:
            ax.axvline(thresh_99, color='red', linestyle='--', linewidth=2, label=f'99th: {thresh_99:.4f}')
        
        ax.set_title(config_name)
        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    
    plt.suptitle('Reconstruction Error Distributions on Normal Data\n(with detection thresholds)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results: Dict, save_path: str):
    """
    Create bar chart comparing model performance.
    """
    if not results:
        print("No evaluation results available")
        return
    
    # Extract metrics
    models = []
    accuracy = []
    precision = []
    recall = []
    f1_scores = []
    auc_roc = []
    
    for config_name, data in sorted(results.items()):
        if 'overall_results' in data and 'best_metrics' in data['overall_results']:
            best = data['overall_results']['best_metrics']
            models.append(config_name)
            accuracy.append(best['accuracy'])
            precision.append(best['precision'])
            recall.append(best['recall'])
            f1_scores.append(best['f1_score'])
            auc_roc.append(best['auc_roc'])
    
    if not models:
        print("No model metrics to plot")
        return
    
    # Create plot
    x = np.arange(len(models))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - 2*width, accuracy, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x - width, precision, width, label='Precision', color='forestgreen')
    bars3 = ax.bar(x, recall, width, label='Recall', color='darkorange')
    bars4 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='crimson')
    bars5 = ax.bar(x + 2*width, auc_roc, width, label='AUC-ROC', color='purple')
    
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Detection Performance Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='Target (0.9)')
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=6, rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_attack_type_heatmap(results: Dict, save_path: str):
    """
    Create heatmap showing detection performance per attack type.
    """
    if not results:
        print("No evaluation results available")
        return
    
    # Extract F1 scores per attack type per model
    models = []
    attack_types = []
    
    # First, collect all attack types
    for config_name, data in results.items():
        if 'attack_results' in data:
            for attack_type in data['attack_results'].keys():
                if attack_type not in attack_types:
                    attack_types.append(attack_type)
    
    if not attack_types:
        print("No attack results to plot")
        return
    
    # Build matrix
    matrix = []
    for config_name in sorted(results.keys()):
        data = results[config_name]
        models.append(config_name)
        
        row = []
        for attack_type in attack_types:
            if 'attack_results' in data and attack_type in data['attack_results']:
                attack_data = data['attack_results'][attack_type]
                if 'threshold_results' in attack_data and 95 in attack_data['threshold_results']:
                    f1 = attack_data['threshold_results'][95]['f1_score']
                    row.append(f1)
                else:
                    row.append(0)
            else:
                row.append(0)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('F1 Score', rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(attack_types)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.set_yticklabels(models)
    
    # Add value annotations
    for i in range(len(models)):
        for j in range(len(attack_types)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Detection F1 Score by Model and Attack Type\n(at 95th percentile threshold)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Model Configuration')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(results: Dict, save_path: str):
    """
    Plot confusion matrices for all models.
    """
    if not results:
        print("No evaluation results available")
        return
    
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (config_name, data) in enumerate(sorted(results.items())):
        ax = axes[idx]
        
        if 'overall_results' not in data or 'best_metrics' not in data['overall_results']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config_name)
            continue
        
        cm = data['overall_results']['best_metrics']['confusion_matrix']
        cm_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        
        ax.set_title(f'{config_name}\n(F1={data["overall_results"]["best_metrics"]["f1_score"]:.3f})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices for All Models\n(Best threshold per model)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_dashboard(results: Dict, thresholds: Dict, save_path: str):
    """
    Create a summary dashboard with key metrics.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Best models table (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    if results:
        # Sort by F1 score
        sorted_models = sorted(
            [(k, v) for k, v in results.items() 
             if 'overall_results' in v and 'best_metrics' in v['overall_results']],
            key=lambda x: x[1]['overall_results']['best_metrics']['f1_score'],
            reverse=True
        )
        
        table_data = []
        for config, data in sorted_models[:5]:
            best = data['overall_results']['best_metrics']
            table_data.append([
                config,
                f"{best['accuracy']:.3f}",
                f"{best['f1_score']:.3f}",
                f"{best['auc_roc']:.3f}"
            ])
        
        if table_data:
            table = ax1.table(
                cellText=table_data,
                colLabels=['Model', 'Accuracy', 'F1', 'AUC'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
    
    ax1.set_title('Top 5 Models by F1 Score', fontsize=12, fontweight='bold', pad=20)
    
    # 2. Overall metrics (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if results:
        best_model = sorted_models[0] if sorted_models else None
        if best_model:
            metrics = best_model[1]['overall_results']['best_metrics']
            labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
            values = [
                metrics['accuracy'], metrics['precision'],
                metrics['recall'], metrics['f1_score'], metrics['auc_roc']
            ]
            
            colors = ['steelblue', 'forestgreen', 'darkorange', 'crimson', 'purple']
            bars = ax2.barh(labels, values, color=colors)
            ax2.set_xlim(0, 1.1)
            ax2.axvline(x=0.9, color='gray', linestyle=':', alpha=0.7)
            
            for bar, val in zip(bars, values):
                ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center')
            
            ax2.set_title(f'Best Model: {best_model[0]}', fontsize=12, fontweight='bold')
    
    # 3. Success criteria check (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    if results and sorted_models:
        best = sorted_models[0][1]['overall_results']['best_metrics']
        
        criteria = [
            ('Recall ≥ 0.90', best['recall'] >= 0.90, best['recall']),
            ('FPR ≤ 0.05', best['fpr'] <= 0.05, best['fpr']),
            ('F1 ≥ 0.85', best['f1_score'] >= 0.85, best['f1_score']),
            ('AUC ≥ 0.90', best['auc_roc'] >= 0.90, best['auc_roc'])
        ]
        
        for i, (name, passed, value) in enumerate(criteria):
            color = 'green' if passed else 'red'
            symbol = '✓' if passed else '✗'
            ax3.text(0.1, 0.8 - i*0.2, f'{symbol} {name}: {value:.3f}', 
                    fontsize=12, color=color, transform=ax3.transAxes)
    
    ax3.set_title('Success Criteria', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Attack type summary (bottom left + middle)
    ax4 = fig.add_subplot(gs[1, :2])
    
    if results:
        attack_f1 = {}
        for config, data in results.items():
            if 'attack_results' in data:
                for attack_type, attack_data in data['attack_results'].items():
                    if attack_type not in attack_f1:
                        attack_f1[attack_type] = []
                    if 'threshold_results' in attack_data and 95 in attack_data['threshold_results']:
                        attack_f1[attack_type].append(
                            attack_data['threshold_results'][95]['f1_score']
                        )
        
        if attack_f1:
            attack_types = list(attack_f1.keys())
            avg_f1 = [np.mean(attack_f1[at]) for at in attack_types]
            std_f1 = [np.std(attack_f1[at]) for at in attack_types]
            
            x = np.arange(len(attack_types))
            bars = ax4.bar(x, avg_f1, yerr=std_f1, capsize=5, 
                          color='steelblue', alpha=0.8)
            ax4.set_xticks(x)
            ax4.set_xticklabels(attack_types, rotation=45, ha='right')
            ax4.set_ylabel('Average F1 Score')
            ax4.set_ylim(0, 1.1)
            ax4.axhline(y=0.85, color='gray', linestyle=':', alpha=0.7)
            ax4.set_title('Detection Performance by Attack Type\n(averaged across models)', 
                         fontsize=12, fontweight='bold')
    
    # 5. Model config comparison (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    if results:
        # Group by window size
        ws_f1 = {50: [], 75: [], 100: []}
        for config, data in results.items():
            if 'overall_results' in data and 'best_metrics' in data['overall_results']:
                ws = int(config.split('s')[0])
                ws_f1[ws].append(data['overall_results']['best_metrics']['f1_score'])
        
        ws_labels = [f'{ws}s' for ws in sorted(ws_f1.keys())]
        ws_means = [np.mean(ws_f1[ws]) if ws_f1[ws] else 0 for ws in sorted(ws_f1.keys())]
        ws_stds = [np.std(ws_f1[ws]) if ws_f1[ws] else 0 for ws in sorted(ws_f1.keys())]
        
        ax5.bar(ws_labels, ws_means, yerr=ws_stds, capsize=5, 
               color=['lightblue', 'steelblue', 'darkblue'])
        ax5.set_ylabel('F1 Score')
        ax5.set_xlabel('Window Size')
        ax5.set_ylim(0, 1.1)
        ax5.set_title('Performance by Window Size', fontsize=12, fontweight='bold')
    
    plt.suptitle('N2KShield Detection Performance Summary', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Create all visualizations."""
    print("=" * 80)
    print("PHASE 3 - STEP 5: VISUALIZE RESULTS")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Ensure output directories exist
    ensure_dirs()
    
    # Load data
    print("\nLoading results...")
    results = load_evaluation_results()
    thresholds = load_threshold_data()
    print(f"  Loaded {len(results)} evaluation results")
    print(f"  Loaded {len(thresholds)} threshold files")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Error distributions
    plot_error_distributions(
        thresholds,
        os.path.join(FIGURES_DIR, "error_distributions.png")
    )
    
    # 2. Model comparison
    plot_model_comparison(
        results,
        os.path.join(FIGURES_DIR, "model_comparison.png")
    )
    
    # 3. Attack type heatmap
    plot_attack_type_heatmap(
        results,
        os.path.join(FIGURES_DIR, "attack_type_heatmap.png")
    )
    
    # 4. Confusion matrices
    plot_confusion_matrices(
        results,
        os.path.join(FIGURES_DIR, "confusion_matrices.png")
    )
    
    # 5. Summary dashboard
    plot_summary_dashboard(
        results, thresholds,
        os.path.join(FIGURES_DIR, "summary_dashboard.png")
    )
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print(f"Completed: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
