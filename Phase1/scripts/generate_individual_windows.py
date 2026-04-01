#!/usr/bin/env python3
"""
Generate INDIVIDUAL PNG files for EVERY window
Creates separate image for each window in each configuration
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Import config
sys.path.insert(0, 'scripts')
from config import (TIME_STEPS, SAMPLING_PERIODS, WINDOW_STEP_TRAIN,
                   DATA_DIR, SIGNAL_ORDER_FILE, NUM_SIGNALS, NUM_FEATURES)

print("="*80)
print("GENERATING INDIVIDUAL WINDOW IMAGES")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_DIR, index_col=0, parse_dates=True)
with open(SIGNAL_ORDER_FILE, 'r') as f:
    signal_order = json.load(f)['clustered_order']

print(f"Dataset: {df.shape[0]} samples × {df.shape[1]} features")

# Create base directory
vis_dir = Path("visualizations")

# ============================================================================
# FUNCTION: Create windows
# ============================================================================
def create_windows(data, window_size, sampling_period, step_size):
    """Create sliding windows from time series data"""
    # Downsample if needed
    if sampling_period > 1:
        data_sampled = data.iloc[::sampling_period].copy()
    else:
        data_sampled = data.copy()
    
    n_samples = len(data_sampled)
    
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
        
        # Store original indices
        orig_start = start_idx * sampling_period
        orig_end = end_idx * sampling_period
        window_indices.append((orig_start, orig_end))
    
    return np.array(windows), window_indices


# ============================================================================
# FUNCTION: Save individual window images
# ============================================================================
def save_individual_windows(windows, window_indices, output_dir, window_size, 
                           sampling_period, signal_order):
    """Save each window as a separate PNG file"""
    
    if len(windows) == 0:
        print(f"  ⚠️  No windows to save")
        return
    
    # Create individual_windows subdirectory
    individual_dir = output_dir / "individual_windows"
    individual_dir.mkdir(exist_ok=True)
    
    print(f"  Saving {len(windows)} individual window images...")
    
    for i, (window, (start_time, end_time)) in enumerate(zip(windows, window_indices)):
        # Create figure for this window
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(window.T, aspect='auto', cmap='viridis', vmin=0, vmax=1, 
                      interpolation='nearest')
        
        # Add signal boundaries
        for j in range(1, len(signal_order)):
            ax.axhline(j * 4 - 0.5, color='red', linewidth=1.5, alpha=0.8)
        
        # Labels
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        # Title with window info
        title = f'Window {i:04d}: Time {start_time}-{end_time}s\n'
        title += f'{window_size}s window × {sampling_period}s sampling'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Y-axis: signal names
        y_ticks = [j * 4 + 1.5 for j in range(len(signal_order))]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(signal_order, fontsize=10)
        
        # X-axis: timesteps
        ax.set_xticks(np.arange(0, window_size, max(1, window_size//10)))
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Normalized Value [0, 1]', fraction=0.046)
        cbar.ax.tick_params(labelsize=10)
        
        # Add info box
        info_text = f'Window Index: {i}/{len(windows)-1}\n'
        info_text += f'Shape: {window.shape[0]} × {window.shape[1]}\n'
        info_text += f'Mean: {window.mean():.4f}\n'
        info_text += f'Std: {window.std():.4f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
        
        plt.tight_layout()
        
        # Save with zero-padded filename
        output_file = individual_dir / f"window_{i:04d}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Progress indicator every 50 windows
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(windows)} windows saved...")
    
    print(f"  ✓ Saved {len(windows)} individual window images to: {individual_dir}")
    print(f"    Files: window_0000.png to window_{len(windows)-1:04d}.png")


# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================
print("\n" + "="*80)
print("GENERATING INDIVIDUAL WINDOW IMAGES FOR ALL CONFIGURATIONS")
print("="*80)

total_images = 0

for time_step in TIME_STEPS:
    print(f"\n{'='*80}")
    print(f"WINDOW SIZE: {time_step} seconds")
    print(f"{'='*80}")
    
    window_dir = vis_dir / f"{time_step}s_window"
    window_dir.mkdir(exist_ok=True)
    
    for sampling_period in SAMPLING_PERIODS:
        print(f"\n  Sampling Period: {sampling_period} second(s)")
        print(f"  {'-'*60}")
        
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
        
        # Save individual window images
        save_individual_windows(windows, window_indices, sampling_dir, 
                               time_step, sampling_period, signal_order)
        
        total_images += len(windows)
        print(f"  ✓ Complete: {sampling_dir}/individual_windows/")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("INDIVIDUAL WINDOW IMAGE GENERATION COMPLETE!")
print("="*80)

print(f"\n📁 Total individual window images created: {total_images}")

print(f"\n📁 Directory structure:")
print(f"   visualizations/")

for time_step in TIME_STEPS:
    print(f"   ├── {time_step}s_window/")
    for i, sampling_period in enumerate(SAMPLING_PERIODS):
        prefix = "└──" if i == len(SAMPLING_PERIODS)-1 else "├──"
        
        # Count windows for this config
        sampling_dir = vis_dir / f"{time_step}s_window" / f"sampling_{sampling_period}s" / "individual_windows"
        if sampling_dir.exists():
            num_files = len(list(sampling_dir.glob("*.png")))
            print(f"   │   {prefix} sampling_{sampling_period}s/")
            print(f"   │       └── individual_windows/ ({num_files} PNG files)")

print("\n✅ All individual window images generated!")
print("   Each window is saved as a separate PNG file.")
print("="*80)
