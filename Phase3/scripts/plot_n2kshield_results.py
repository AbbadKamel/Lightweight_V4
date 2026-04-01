"""
N2KShield Results Visualization
================================
Generates plots for comparing detection methods:
1. Confusion Matrix Grid (2x2 for 4 methods)
2. Bar Chart: Metrics Comparison
3. Trade-off Plot: Recall vs FPR
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Results from evaluation
RESULTS = [
    {
        'method': 'Baseline OR',
        'recall': 0.9533,
        'precision': 0.9795,
        'f1': 0.9662,
        'fpr': 0.8571,
        'tp': 286, 'fp': 6, 'tn': 1, 'fn': 14
    },
    {
        'method': 'N2KShield Weighted',
        'recall': 0.4167,
        'precision': 1.0000,
        'f1': 0.5882,
        'fpr': 0.0000,
        'tp': 125, 'fp': 0, 'tn': 7, 'fn': 175
    },
    {
        'method': 'N2KShield Groups',
        'recall': 1.0000,
        'precision': 0.9772,
        'f1': 0.9885,
        'fpr': 1.0000,
        'tp': 300, 'fp': 7, 'tn': 0, 'fn': 0
    },
    {
        'method': 'N2KShield Hybrid',
        'recall': 0.5100,
        'precision': 0.9745,
        'f1': 0.6696,
        'fpr': 0.5714,
        'tp': 153, 'fp': 4, 'tn': 3, 'fn': 147
    }
]


def plot_confusion_matrices():
    """Plot 2x2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Confusion Matrices: Method Comparison', fontsize=16, fontweight='bold')
    
    for idx, (ax, result) in enumerate(zip(axes.flatten(), RESULTS)):
        # Build confusion matrix
        cm = np.array([[result['tn'], result['fp']],
                       [result['fn'], result['tp']]])
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    annot_kws={'size': 14, 'weight': 'bold'},
                    cbar=False)
        
        # Title with key metrics
        fpr_color = 'green' if result['fpr'] < 0.1 else ('orange' if result['fpr'] < 0.5 else 'red')
        recall_color = 'green' if result['recall'] > 0.7 else ('orange' if result['recall'] > 0.4 else 'red')
        
        ax.set_title(f"{result['method']}\nRecall: {result['recall']:.1%} | FPR: {result['fpr']:.1%}", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        
        # Add colored border based on FPR
        if result['fpr'] == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        elif result['fpr'] > 0.8:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    plt.tight_layout()
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_metrics_comparison():
    """Bar chart comparing all metrics across methods."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    methods = [r['method'] for r in RESULTS]
    x = np.arange(len(methods))
    width = 0.2
    
    metrics = ['recall', 'precision', 'f1', 'fpr']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    labels = ['Recall (↑)', 'Precision (↑)', 'F1 Score (↑)', 'FPR (↓)']
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        values = [r[metric] for r in RESULTS]
        bars = ax.bar(x + i*width, values, width, label=label, color=color, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1%}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('N2KShield: Metrics Comparison\n(↑ = Higher is Better, ↓ = Lower is Better)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add recommendation box
    best_method = "N2KShield Weighted"
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.02, 0.98, f'🏆 Recommended: {best_method}\n   • 0% False Positive Rate\n   • 100% Precision',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_recall_vs_fpr():
    """Scatter plot showing Recall vs FPR trade-off."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red', 'green', 'blue', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for result, color, marker in zip(RESULTS, colors, markers):
        ax.scatter(result['fpr'], result['recall'], 
                   s=300, c=color, marker=marker, 
                   label=result['method'], edgecolors='black', linewidth=2, alpha=0.8)
        
        # Add annotation
        offset = (10, 10) if result['method'] != 'Baseline OR' else (-80, -20)
        ax.annotate(result['method'], 
                    xy=(result['fpr'], result['recall']),
                    xytext=offset, textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    # Ideal point (0,1)
    ax.scatter(0, 1, s=400, c='gold', marker='*', label='Ideal (FPR=0, Recall=1)', 
               edgecolors='black', linewidth=2, zorder=10)
    
    # Zones
    ax.axvspan(0, 0.1, alpha=0.1, color='green', label='Good FPR Zone (<10%)')
    ax.axhspan(0.7, 1.0, alpha=0.1, color='blue', label='Good Recall Zone (>70%)')
    
    ax.set_xlabel('False Positive Rate (FPR) →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (Sensitivity) →', fontsize=12, fontweight='bold')
    ax.set_title('N2KShield: Recall vs FPR Trade-off\n(Closer to top-left corner is better)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    
    plt.tight_layout()
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_recall_vs_fpr.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_detection_breakdown():
    """Stacked bar showing TP/FP/TN/FN breakdown."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = [r['method'] for r in RESULTS]
    x = np.arange(len(methods))
    
    tp = [r['tp'] for r in RESULTS]
    fn = [r['fn'] for r in RESULTS]
    fp = [r['fp'] for r in RESULTS]
    tn = [r['tn'] for r in RESULTS]
    
    # Stacked bars
    width = 0.6
    
    ax.bar(x, tp, width, label=f'True Positive (Attack detected)', color='#27ae60', edgecolor='black')
    ax.bar(x, fn, width, bottom=tp, label=f'False Negative (Attack missed)', color='#e74c3c', edgecolor='black')
    ax.bar(x, fp, width, bottom=np.array(tp)+np.array(fn), label=f'False Positive (False alarm)', color='#f39c12', edgecolor='black')
    ax.bar(x, tn, width, bottom=np.array(tp)+np.array(fn)+np.array(fp), label=f'True Negative (Normal OK)', color='#3498db', edgecolor='black')
    
    ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('N2KShield: Detection Breakdown\n(300 Attack samples + 7 Normal samples)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add total counts
    totals = [sum([tp[i], fn[i], fp[i], tn[i]]) for i in range(len(methods))]
    for i, total in enumerate(totals):
        ax.text(i, total + 5, f'n={total}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(config.FIGURES_DIR, 'n2kshield_detection_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("N2KSHIELD: Generating Visualization Plots")
    print("=" * 60)
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    print("\n📊 Creating plots...")
    
    # Plot 1: Confusion Matrices
    print("  1. Confusion Matrices Grid...")
    plot_confusion_matrices()
    
    # Plot 2: Metrics Comparison
    print("  2. Metrics Comparison Bar Chart...")
    plot_metrics_comparison()
    
    # Plot 3: Recall vs FPR
    print("  3. Recall vs FPR Trade-off...")
    plot_recall_vs_fpr()
    
    # Plot 4: Detection Breakdown
    print("  4. Detection Breakdown...")
    plot_detection_breakdown()
    
    print("\n" + "=" * 60)
    print(f"✅ All plots saved to: {config.FIGURES_DIR}")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • n2kshield_confusion_matrices.png")
    print("  • n2kshield_metrics_comparison.png")
    print("  • n2kshield_recall_vs_fpr.png")
    print("  • n2kshield_detection_breakdown.png")


if __name__ == "__main__":
    main()
