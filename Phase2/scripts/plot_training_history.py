
import os
import sys
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

def plot_all_histories():
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    PLOTS_DIR = os.path.join(LOGS_DIR, 'plots')
    
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    # Find all history files
    history_files = glob.glob(os.path.join(LOGS_DIR, "*_history.json"))
    history_files = [f for f in history_files if "summary" not in f]
    history_files.sort()
    
    print(f"Found {len(history_files)} history files.")
    
    # Create 3x3 Grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Phase 2: Training History (Loss Curves)', fontsize=16)
    
    axes = axes.flatten()
    
    for i, file_path in enumerate(history_files):
        if i >= 9: break # Safety limit
        
        filename = os.path.basename(file_path)
        config_name = filename.replace('_history.json', '')
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        loss = data['loss']
        val_loss = data['val_loss']
        epochs = range(1, len(loss) + 1)
        
        ax = axes[i]
        
        # Plot curves
        ax.plot(epochs, loss, label='Train Loss', linewidth=2, color='blue', alpha=0.7)
        ax.plot(epochs, val_loss, label='Val Loss', linewidth=2, color='red', linestyle='--')
        
        # Highlight best validation point
        best_val = min(val_loss)
        best_epoch = val_loss.index(best_val) + 1
        ax.scatter(best_epoch, best_val, color='green', s=100, zorder=5, label=f'Best Val ({best_val:.4f})')
        
        ax.set_title(f"Model: {config_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log scale if range is huge
        if max(loss) / (min(loss) + 1e-9) > 100:
             ax.set_yscale('log')
             
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = os.path.join(PLOTS_DIR, 'training_history_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid plot to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    plot_all_histories()
