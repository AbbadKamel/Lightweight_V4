#!/usr/bin/env python3
"""
Generate visualization plots for all windowing configurations
Creates organized folder structure:
  visualizations/
    ├── 50s_window/
    │   ├── sampling_1s/
    │   ├── sampling_5s/
    │   └── sampling_10s/
    ├── 75s_window/
    │   ├── sampling_1s/
    │   ├── sampling_5s/
    │   └── sampling_10s/
    └── 100s_window/
        ├── sampling_1s/
        ├── sampling_5s/
        └── sampling_10s/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import sys

# Import config
import sys
sys.path.insert(0, 'scripts')
from config import (TIME_STEPS, SAMPLING_PERIODS, WINDOW_STEP_TRAIN,
                   DATA_DIR, SIGNAL_ORDER_FILE, NUM_SIGNALS, NUM_FEATURES)

print("="*80)
print("GENERATING WINDOWING VISUALIZATIONS")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_DIR, index_col=0, parse_dates=True)
with open(SIGNAL_ORDER_FILE, 'r') as f:
    signal_order = json.load(f)['clustered_order']

print(f"Dataset: {df.shape[0]} samples × {df.shape[1]} features")
print(f"Signals: {signal_order}")

# Create base visualization directory
vis_dir = Path("visualizations")
vis_dir.mkdir(exist_ok=True)

# ============================================================================
# FUNCTION: Create windows from data
# ============================================================================
def create_windows(data, window_size, sampling_period, step_size):
    """
    Create sliding windows from time series data
    
    Args:
        data: pandas DataFrame (n_samples, n_features)
        window_size: int, number of timesteps per window
        sampling_period: int, downsampling factor
        step_size: int, sliding step between windows
    
    Returns:
        windows: numpy array (n_windows, window_size, n_features)
        window_indices: list of (start_idx, end_idx) for each window
    """
    # Downsample if needed
    if sampling_period > 1:
        data_sampled = data.iloc[::sampling_period].copy()
    else:
        data_sampled = data.copy()
    
    n_samples = len(data_sampled)
    
    # Calculate number of windows
    if n_samples < window_size:
        print(f"  ⚠️  Warning: Not enough samples ({n_samples}) for window size {window_size}")
        return np.array([]), []
    
    # Create windows
    windows = []
    window_indices = []
    
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window = data_sampled.iloc[start_idx:end_idx].values
        windows.append(window)
        
        # Store original indices (before downsampling)
        orig_start = start_idx * sampling_period
        orig_end = end_idx * sampling_period
        window_indices.append((orig_start, orig_end))
    
    return np.array(windows), window_indices


# ============================================================================
# FUNCTION: Plot window examples
# ============================================================================
def plot_window_examples(windows, window_indices, output_dir, window_size, sampling_period, 
                         signal_order, num_examples=6):
    """Plot example windows as heatmaps"""
    
    if len(windows) == 0:
        print(f"  ⚠️  No windows to plot")
        return
    
    # Select evenly spaced examples
    indices = np.linspace(0, len(windows)-1, min(num_examples, len(windows)), dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Example Windows: {window_size}s window × {sampling_period}s sampling\n'
                 f'Total windows: {len(windows)}', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        window = windows[idx]
        start_time, end_time = window_indices[idx]
        
        # Plot heatmap
        im = ax.imshow(window.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        
        # Add signal boundaries
        for j in range(1, len(signal_order)):
            ax.axhline(j * 4 - 0.5, color='red', linewidth=1, alpha=0.7)
        
        # Labels
        ax.set_xlabel('Time (steps)', fontsize=10)
        ax.set_ylabel('Feature Index', fontsize=10)
        ax.set_title(f'Window {idx}\nTime: {start_time}-{end_time}s', fontsize=11)
        
        # Y-axis: signal names
        y_ticks = [i * 4 + 1.5 for i in range(len(signal_order))]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(signal_order, fontsize=8)
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_file = output_dir / "example_windows.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")


# ============================================================================
# FUNCTION: Plot window statistics
# ============================================================================
def plot_window_statistics(windows, output_dir, window_size, sampling_period):
    """Plot statistical analysis of windows"""
    
    if len(windows) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Window Statistics: {window_size}s × {sampling_period}s sampling\n'
                 f'{len(windows)} windows', fontsize=16, fontweight='bold')
    
    # Flatten windows for global statistics
    all_values = windows.flatten()
    
    # Plot 1: Distribution of all values
    ax = axes[0, 0]
    ax.hist(all_values, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Normalized Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Value Distribution (All {len(all_values):,} values)', fontsize=12)
    ax.axvline(all_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_values.mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Per-window mean values
    ax = axes[0, 1]
    window_means = windows.mean(axis=(1, 2))  # Mean of each window
    ax.plot(window_means, linewidth=1, alpha=0.7)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Window Mean Value', fontsize=12)
    ax.set_title('Mean Value per Window (Temporal Trend)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Per-feature variance across all windows
    ax = axes[1, 0]
    feature_variance = windows.var(axis=0).mean(axis=0)  # Variance per feature
    ax.bar(range(len(feature_variance)), feature_variance, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Feature Variance (How much each feature varies)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add signal boundaries
    for i in range(1, len(signal_order)):
        ax.axvline(i * 4 - 0.5, color='red', linewidth=1, alpha=0.5)
    
    # Plot 4: Window diversity (pairwise correlation)
    ax = axes[1, 1]
    # Sample subset of windows for correlation (too expensive for all)
    sample_size = min(100, len(windows))
    sample_indices = np.linspace(0, len(windows)-1, sample_size, dtype=int)
    
    # Flatten each window to 1D for correlation
    windows_flat = windows.reshape(len(windows), -1)
    windows_sample = windows_flat[sample_indices]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(windows_sample)
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Window Index (sampled)', fontsize=12)
    ax.set_ylabel('Window Index (sampled)', fontsize=12)
    ax.set_title(f'Window Similarity\n(Correlation between {sample_size} windows)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Correlation')
    
    plt.tight_layout()
    output_file = output_dir / "window_statistics.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")


# ============================================================================
# FUNCTION: Plot temporal coverage
# ============================================================================
def plot_temporal_coverage(window_indices, output_dir, window_size, sampling_period, total_samples):
    """Visualize how windows cover the timeline"""
    
    if len(window_indices) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Plot each window as a horizontal bar
    for i, (start, end) in enumerate(window_indices[:100]):  # Show first 100 windows
        ax.barh(i, end - start, left=start, height=0.8, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Window Index', fontsize=14, fontweight='bold')
    ax.set_title(f'Temporal Coverage: {window_size}s window × {sampling_period}s sampling\n'
                 f'Showing first 100 of {len(window_indices)} windows', fontsize=16, fontweight='bold')
    ax.set_xlim(0, total_samples)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add vertical lines at train/valid/test splits
    train_end = int(total_samples * 0.7)
    valid_end = int(total_samples * 0.85)
    ax.axvline(train_end, color='red', linestyle='--', linewidth=2, label='Train/Valid split')
    ax.axvline(valid_end, color='orange', linestyle='--', linewidth=2, label='Valid/Test split')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    output_file = output_dir / "temporal_coverage.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")


# ============================================================================
# FUNCTION: Plot window shape summary
# ============================================================================
def plot_window_info(windows, window_indices, output_dir, window_size, sampling_period):
    """Create summary infographic"""
    
    if len(windows) == 0:
        return
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create text summary
    info_text = f"""
    WINDOW CONFIGURATION SUMMARY
    {'='*60}
    
    Window Size: {window_size} seconds
    Sampling Period: {sampling_period} second(s)
    Step Size: {WINDOW_STEP_TRAIN} seconds
    Overlap: {((window_size - WINDOW_STEP_TRAIN) / window_size * 100):.1f}%
    
    {'='*60}
    DATA STATISTICS
    {'='*60}
    
    Total Windows Generated: {len(windows)}
    Window Shape: ({window_size} timesteps, {NUM_FEATURES} features)
    
    Memory Size: {windows.nbytes / 1024 / 1024:.2f} MB
    Values per Window: {window_size * NUM_FEATURES:,}
    Total Values: {windows.size:,}
    
    {'='*60}
    TEMPORAL COVERAGE
    {'='*60}
    
    First Window: {window_indices[0][0]}-{window_indices[0][1]}s
    Last Window: {window_indices[-1][0]}-{window_indices[-1][1]}s
    Time Span Covered: {window_indices[-1][1]}s ({window_indices[-1][1]/60:.1f} min)
    
    {'='*60}
    VALUE STATISTICS
    {'='*60}
    
    Min Value: {windows.min():.6f}
    Max Value: {windows.max():.6f}
    Mean Value: {windows.mean():.6f}
    Std Deviation: {windows.std():.6f}
    
    {'='*60}
    CNN INPUT SPECIFICATIONS
    {'='*60}
    
    Input Tensor Shape: (batch_size, {window_size}, {NUM_FEATURES})
    
    This will be fed to CNN autoencoder:
    - Each window is a "{window_size}×{NUM_FEATURES}" grayscale image
    - CNN learns to reconstruct normal patterns
    - Reconstruction error → anomaly score
    
    {'='*60}
    """
    
    plt.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=fig.transFigure)
    plt.axis('off')
    
    plt.tight_layout()
    output_file = output_dir / "window_info.txt.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")


# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS FOR ALL CONFIGURATIONS")
print("="*80)

for time_step in TIME_STEPS:
    print(f"\n{'='*80}")
    print(f"WINDOW SIZE: {time_step} seconds")
    print(f"{'='*80}")
    
    # Create window size directory
    window_dir = vis_dir / f"{time_step}s_window"
    window_dir.mkdir(exist_ok=True)
    
    for sampling_period in SAMPLING_PERIODS:
        print(f"\n  Sampling Period: {sampling_period} second(s)")
        print(f"  {'-'*60}")
        
        # Create sampling directory
        sampling_dir = window_dir / f"sampling_{sampling_period}s"
        sampling_dir.mkdir(exist_ok=True)
        
        # Generate windows
        print(f"  Generating windows...")
        windows, window_indices = create_windows(
            df, time_step, sampling_period, WINDOW_STEP_TRAIN
        )
        
        if len(windows) == 0:
            print(f"  ⚠️  Skipping (no windows generated)")
            continue
        
        print(f"  Generated {len(windows)} windows of shape {windows.shape}")
        
        # Generate all plots
        print(f"  Creating visualizations...")
        
        plot_window_examples(windows, window_indices, sampling_dir, 
                           time_step, sampling_period, signal_order)
        
        plot_window_statistics(windows, sampling_dir, 
                             time_step, sampling_period)
        
        # plot_temporal_coverage(window_indices, sampling_dir, 
        #                      time_step, sampling_period, len(df))
        
        plot_window_info(windows, window_indices, sampling_dir,
                       time_step, sampling_period)
        
        print(f"  ✓ Complete: {sampling_dir}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*80)

print(f"\n📁 Output directory structure:")
print(f"   {vis_dir}/")

for time_step in TIME_STEPS:
    print(f"   ├── {time_step}s_window/")
    for i, sampling_period in enumerate(SAMPLING_PERIODS):
        prefix = "└──" if i == len(SAMPLING_PERIODS)-1 else "├──"
        print(f"   │   {prefix} sampling_{sampling_period}s/")
        print(f"   │       ├── example_windows.png")
        print(f"   │       ├── window_statistics.png")
        print(f"   │       └── window_info.txt.png")

print("\n✅ All visualizations generated!")
print("   Open the folders to inspect different window configurations.")
print("="*80)
