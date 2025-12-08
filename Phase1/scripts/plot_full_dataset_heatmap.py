#!/usr/bin/env python3
"""
Create ONE BIG heatmap showing the ENTIRE master_table_final.csv dataset
All 5,095 rows (seconds) × 60 features as one large visualization
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

print("Loading master table...")
df = pd.read_csv('results/master_table_final.csv', index_col=0, parse_dates=True)

with open('results/optimal_signal_order.json', 'r') as f:
    signal_order = json.load(f)['clustered_order']

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Duration: {len(df)} seconds = {len(df)/60:.1f} minutes")
print(f"Creating full dataset heatmap...")

# ============================================================================
# FULL DATASET HEATMAP
# ============================================================================

fig = plt.figure(figsize=(24, 12))
ax = plt.gca()

# Plot ENTIRE dataset (all 5,095 rows)
im = ax.imshow(df.T, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Normalized Value [0, 1]', fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=12)

# X-axis: Time
ax.set_xlabel('Time (seconds)', fontsize=16, fontweight='bold')
ax.set_ylabel('Feature Index', fontsize=16, fontweight='bold')

# Add time markers every 10 minutes
time_markers = np.arange(0, len(df), 600)  # Every 600 seconds = 10 minutes
ax.set_xticks(time_markers)
ax.set_xticklabels([f'{int(t/60)}m' for t in time_markers], fontsize=10)

# Y-axis: Features (group by signal)
# Add horizontal red lines to separate signals
signal_boundaries = []
for i in range(1, len(signal_order)):
    boundary = i * 4 - 0.5
    ax.axhline(boundary, color='red', linewidth=1.5, alpha=0.8)
    signal_boundaries.append(boundary)

# Add signal labels on Y-axis
y_ticks = [i * 4 + 1.5 for i in range(len(signal_order))]  # Center of each signal block
ax.set_yticks(y_ticks)
ax.set_yticklabels(signal_order, fontsize=11)

# Add grid for better readability
ax.set_xticks(np.arange(0, len(df), 300), minor=True)  # Minor ticks every 5 minutes
ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.3, alpha=0.3, axis='x')

# Title
title = f'Complete Dataset Heatmap: {len(df):,} seconds × {df.shape[1]} features\n'
title += f'({len(df)/60:.1f} minutes of NMEA 2000 data)'
ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

# Add annotations
textstr = f'• Each row (horizontal) = 1 second of data\n'
textstr += f'• Each signal has 4 features: mean, max, min, std\n'
textstr += f'• Red lines separate the 15 signals\n'
textstr += f'• Color: Dark purple = 0.0, Yellow = 1.0'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('results/full_dataset_heatmap.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: results/full_dataset_heatmap.png")
print(f"   Size: {len(df):,} timepoints × {df.shape[1]} features")
print(f"   Duration: {len(df)} seconds = {len(df)/60:.1f} minutes")

# ============================================================================
# Also create a VERY HIGH RESOLUTION version for detailed inspection
# ============================================================================
print("\nCreating ultra-high resolution version...")

fig = plt.figure(figsize=(40, 16))
ax = plt.gca()

im = ax.imshow(df.T, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)

cbar = plt.colorbar(im, ax=ax, label='Normalized Value [0, 1]', fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14)

ax.set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
ax.set_ylabel('Feature Index', fontsize=20, fontweight='bold')

# Time markers every 5 minutes for high-res version
time_markers = np.arange(0, len(df), 300)
ax.set_xticks(time_markers)
ax.set_xticklabels([f'{int(t/60)}m' for t in time_markers], fontsize=12, rotation=45)

# Signal boundaries
for i in range(1, len(signal_order)):
    ax.axhline(i * 4 - 0.5, color='red', linewidth=2, alpha=0.9)

# Signal labels
y_ticks = [i * 4 + 1.5 for i in range(len(signal_order))]
ax.set_yticks(y_ticks)
ax.set_yticklabels(signal_order, fontsize=14, fontweight='bold')

# Grid
ax.set_xticks(np.arange(0, len(df), 60), minor=True)  # Minor ticks every 1 minute
ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.4, axis='x')

title = f'Ultra High-Resolution Dataset Heatmap\n'
title += f'{len(df):,} seconds × {df.shape[1]} features ({len(df)/60:.1f} minutes)'
ax.set_title(title, fontsize=22, fontweight='bold', pad=20)

textstr = f'Full temporal resolution: Every pixel column = 1 second\n'
textstr += f'Feature resolution: 15 signals × 4 aggregations = 60 features\n'
textstr += f'Red lines separate signals (clustered by correlation)\n'
textstr += f'Zoom in to see detailed temporal patterns'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('results/full_dataset_heatmap_highres.png', dpi=200, bbox_inches='tight')
print(f"✓ Saved: results/full_dataset_heatmap_highres.png")
print(f"   (High resolution for detailed inspection)")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print("\nTwo heatmaps created:")
print("  1. full_dataset_heatmap.png (24×12 inches, 150 DPI)")
print("  2. full_dataset_heatmap_highres.png (40×16 inches, 200 DPI)")
print("\nBoth show ALL 5,095 seconds of data in one visualization!")
print("="*70)
