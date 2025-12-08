#!/usr/bin/env python3
"""
Simple validation visualizations for master_table_final.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load data
print("Loading master table...")
df = pd.read_csv('results/master_table_final.csv', index_col=0, parse_dates=True)
print(f"Shape: {df.shape}")

with open('results/scaler_params.json', 'r') as f:
    scaler = json.load(f)

with open('results/optimal_signal_order.json', 'r') as f:
    signal_order = json.load(f)['clustered_order']

print(f"\nSignals: {len(signal_order)}")
print(f"Features: {df.shape[1]}\n")

# ============================================================================
# 1. NORMALIZATION CHECK
# ============================================================================
print("=== Validation 1: Normalization ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Validation 1: Normalization Check', fontsize=14, fontweight='bold')

# Histogram
ax = axes[0]
all_values = df.values.flatten()
ax.hist(all_values, bins=100, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.axvline(1, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Normalized Value')
ax.set_ylabel('Frequency')
ax.set_title(f'All {len(all_values):,} Values')
ax.text(0.5, 0.95, f'Range: [{all_values.min():.6f}, {all_values.max():.6f}]',
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Out of bounds check
ax = axes[1]
out_low = (df < 0).sum().sum()
out_high = (df > 1).sum().sum()
in_bounds = df.size - out_low - out_high
categories = ['Valid [0,1]', 'Too Low', 'Too High']
counts = [in_bounds, out_low, out_high]
colors = ['green', 'red', 'red']
bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Count')
ax.set_title('Bounds Check')
for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontweight='bold')

# NaN check
ax = axes[2]
nan_total = df.isna().sum().sum()
categories = ['Valid', 'NaN']
counts = [df.size - nan_total, nan_total]
colors = ['green', 'red' if nan_total > 0 else 'green']
bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Count')
ax.set_title('Missing Values')
ax.set_yscale('symlog')  # Symmetric log to handle zeros
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., max(1, bar.get_height()),
            f'{count:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/validation_1_normalization.png', dpi=120)
print("✓ Saved: validation_1_normalization.png")
plt.close()

# ============================================================================
# 2. FEATURE RELATIONSHIPS
# ============================================================================
print("\n=== Validation 2: Feature Relationships ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Validation 2: Are Min ≤ Mean ≤ Max?', fontsize=14, fontweight='bold')

# Check violations
violations = {'min>mean': [], 'mean>max': [], 'invalid_std': []}
for signal in signal_order:
    if (df[f'{signal}_min'] > df[f'{signal}_mean']).any():
        violations['min>mean'].append(signal)
    if (df[f'{signal}_mean'] > df[f'{signal}_max']).any():
        violations['mean>max'].append(signal)
    if (df[f'{signal}_std'] < 0).any():
        violations['invalid_std'].append(signal)

# Violation bar chart
ax = axes[0, 0]
viols = [len(violations['min>mean']), len(violations['mean>max']), len(violations['invalid_std'])]
ax.bar(['Min>Mean', 'Mean>Max', 'Std<0'], viols, 
       color=['red' if v > 0 else 'green' for v in viols], alpha=0.7, edgecolor='black')
ax.set_ylabel('Signal Count')
ax.set_title('Violations (should be 0)')
for i, v in enumerate(viols):
    ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Example signal
ax = axes[0, 1]
sig = signal_order[0]
time_slice = df.iloc[:200]
ax.plot(time_slice[f'{sig}_max'], label='Max', linewidth=2)
ax.plot(time_slice[f'{sig}_mean'], label='Mean', linewidth=2)
ax.plot(time_slice[f'{sig}_min'], label='Min', linewidth=2)
ax.fill_between(range(len(time_slice)), time_slice[f'{sig}_min'], time_slice[f'{sig}_max'], alpha=0.2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized Value')
ax.set_title(f'Example: {sig}')
ax.legend()
ax.grid(True, alpha=0.3)

# Scatter: Mean vs Max
ax = axes[1, 0]
sample = df.sample(min(1000, len(df)))
for sig in signal_order[:5]:
    ax.scatter(sample[f'{sig}_mean'], sample[f'{sig}_max'], alpha=0.5, s=10, label=sig)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Mean=Max')
ax.set_xlabel('Mean')
ax.set_ylabel('Max')
ax.set_title('Mean vs Max (points below diagonal)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Scatter: Min vs Mean
ax = axes[1, 1]
for sig in signal_order[:5]:
    ax.scatter(sample[f'{sig}_min'], sample[f'{sig}_mean'], alpha=0.5, s=10, label=sig)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Min=Mean')
ax.set_xlabel('Min')
ax.set_ylabel('Mean')
ax.set_title('Min vs Mean (points below diagonal)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/validation_2_relationships.png', dpi=120)
print("✓ Saved: validation_2_relationships.png")
plt.close()

# ============================================================================
# 3. SIGNAL PATTERNS (Denormalized)
# ============================================================================
print("\n=== Validation 3: Signal Patterns (Real Units) ===")
fig, axes = plt.subplots(5, 3, figsize=(18, 14))
fig.suptitle('Validation 3: Signals in Real Units', fontsize=14, fontweight='bold')
axes = axes.flatten()

for idx, signal in enumerate(signal_order):
    ax = axes[idx]
    
    # Denormalize
    mean_col = f'{signal}_mean'
    mean_norm = df[mean_col]
    min_val = scaler[mean_col]['min']
    max_val = scaler[mean_col]['max']
    mean_real = mean_norm * (max_val - min_val) + min_val
    
    # Plot first 500 seconds
    time_slice = slice(0, min(500, len(df)))
    ax.plot(mean_real.iloc[time_slice], linewidth=1, alpha=0.8)
    ax.axhline(min_val, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(max_val, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Real Value', fontsize=8)
    ax.set_title(f'{signal}\n[{min_val:.2f}, {max_val:.2f}]', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/validation_3_signal_patterns.png', dpi=120)
print("✓ Saved: validation_3_signal_patterns.png")
plt.close()

# ============================================================================
# 4. CORRELATION MATRIX
# ============================================================================
print("\n=== Validation 4: Correlation Structure ===")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Validation 4: Signal Correlations', fontsize=14, fontweight='bold')

mean_cols = [f'{signal}_mean' for signal in signal_order]
corr_matrix = df[mean_cols].corr()

# Heatmap
ax = axes[0]
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(signal_order)))
ax.set_yticks(range(len(signal_order)))
ax.set_xticklabels(signal_order, rotation=90, fontsize=8)
ax.set_yticklabels(signal_order, fontsize=8)
ax.set_title('Correlation Matrix (Clustered Order)')
plt.colorbar(im, ax=ax)

# High correlations
ax = axes[1]
high_corr = []
for i in range(len(signal_order)):
    for j in range(i+1, len(signal_order)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.5:
            high_corr.append((signal_order[i], signal_order[j], r))
high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

if high_corr:
    y_pos = np.arange(min(10, len(high_corr)))
    labels = [f'{p[0]} ↔ {p[1]}' for p in high_corr[:10]]
    values = [p[2] for p in high_corr[:10]]
    colors = ['red' if v < 0 else 'blue' for v in values]
    ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Correlation')
    ax.set_title(f'Top Correlations (|r| > 0.5)')
    ax.axvline(0, color='black', linewidth=1)
    for i, v in enumerate(values):
        ax.text(v, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/validation_4_correlations.png', dpi=120)
print("✓ Saved: validation_4_correlations.png")
plt.close()

# ============================================================================
# 5. TEMPORAL CONTINUITY
# ============================================================================
print("\n=== Validation 5: Temporal Continuity ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Validation 5: Time Series Continuity', fontsize=14, fontweight='bold')

# Rate of change
ax = axes[0, 0]
derivatives = df.diff().abs().mean()
ax.bar(range(len(signal_order)), derivatives[mean_cols], alpha=0.7, edgecolor='black')
ax.set_xlabel('Signal Index')
ax.set_ylabel('Mean Absolute Change')
ax.set_title('Average Rate of Change')
ax.set_xticks(range(len(signal_order)))
ax.set_xticklabels(signal_order, rotation=90, fontsize=8)

# Max jumps
ax = axes[0, 1]
max_jumps = df.diff().abs().max()
ax.bar(range(len(signal_order)), max_jumps[mean_cols], color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Signal Index')
ax.set_ylabel('Max Jump')
ax.set_title('Largest Single-Step Change')
ax.set_xticks(range(len(signal_order)))
ax.set_xticklabels(signal_order, rotation=90, fontsize=8)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='High threshold')
ax.legend()

# Example continuity
ax = axes[1, 0]
most_variable_idx = derivatives[mean_cols].argmax()
sig = signal_order[most_variable_idx]
time_slice = df.iloc[:300]
ax.plot(time_slice[f'{sig}_mean'], linewidth=1.5)
ax.scatter(range(len(time_slice)), time_slice[f'{sig}_mean'], s=10, alpha=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized Value')
ax.set_title(f'Most Variable: {sig}')
ax.grid(True, alpha=0.3)

# Flat segments
ax = axes[1, 1]
max_flat = []
for sig in signal_order:
    diff = df[f'{sig}_mean'].diff()
    flat_mask = (diff == 0)
    max_run = flat_mask.astype(int).groupby((flat_mask != flat_mask.shift()).cumsum()).sum().max()
    max_flat.append(max_run)
ax.bar(range(len(signal_order)), max_flat, color='purple', alpha=0.7, edgecolor='black')
ax.set_xlabel('Signal Index')
ax.set_ylabel('Max Flat Segment (s)')
ax.set_title('Longest Constant Value Period')
ax.set_xticks(range(len(signal_order)))
ax.set_xticklabels(signal_order, rotation=90, fontsize=8)

plt.tight_layout()
plt.savefig('results/validation_5_continuity.png', dpi=120)
print("✓ Saved: validation_5_continuity.png")
plt.close()

# ============================================================================
# 6. CNN VIEW
# ============================================================================
print("\n=== Validation 6: CNN Input View ===")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Validation 6: What CNN Sees', fontsize=14, fontweight='bold')

# Full dataset (subsampled)
ax = axes[0]
subsample = max(1, len(df) // 500)
df_sub = df.iloc[::subsample]
im = ax.imshow(df_sub.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('Time (subsampled)')
ax.set_ylabel('Feature Index')
ax.set_title(f'Full Dataset as Image\n{len(df_sub)} × 60')
plt.colorbar(im, ax=ax)
for i in range(1, len(signal_order)):
    ax.axhline(i*4 - 0.5, color='red', linewidth=0.5, alpha=0.5)

# CNN window (50 timesteps)
ax = axes[1]
window = df.iloc[1000:1050]
im = ax.imshow(window.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Feature Index')
ax.set_title('CNN Window (50 timesteps)')
plt.colorbar(im, ax=ax)
ax.set_yticks(range(0, 60, 4))
ax.set_yticklabels(signal_order, fontsize=7)
for i in range(1, len(signal_order)):
    ax.axhline(i*4 - 0.5, color='red', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig('results/validation_6_cnn_view.png', dpi=120)
print("✓ Saved: validation_6_cnn_view.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Duration: {len(df)} seconds = {len(df)/60:.1f} minutes")
print(f"Signals: {len(signal_order)}")

print(f"\n1. NORMALIZATION:")
print(f"   Range: [{all_values.min():.6f}, {all_values.max():.6f}]")
print(f"   Out of bounds: {out_low + out_high}")
print(f"   NaN count: {nan_total}")
print(f"   {'✓ PASS' if (all_values.min() >= 0 and all_values.max() <= 1 and out_low + out_high == 0) else '✗ FAIL'}")

print(f"\n2. FEATURE RELATIONSHIPS:")
print(f"   Min>Mean violations: {len(violations['min>mean'])}")
print(f"   Mean>Max violations: {len(violations['mean>max'])}")
print(f"   Invalid Std: {len(violations['invalid_std'])}")
print(f"   {'✓ PASS' if sum(len(v) for v in violations.values()) == 0 else '✗ FAIL'}")

print(f"\n3. CORRELATIONS:")
print(f"   High correlations (|r|>0.5): {len(high_corr)}")
if high_corr:
    print(f"   Strongest: {high_corr[0][0]} ↔ {high_corr[0][1]} (r={high_corr[0][2]:.3f})")

print(f"\n4. CONTINUITY:")
print(f"   Max derivative: {derivatives[mean_cols].max():.6f}")
print(f"   Max jump: {max_jumps[mean_cols].max():.6f}")
print(f"   Max flat segment: {max(max_flat)} seconds")

print(f"\n5. CNN READINESS:")
ready = (all_values.min() >= 0 and all_values.max() <= 1 and nan_total == 0)
print(f"   {'✓ READY FOR TRAINING' if ready else '✗ NOT READY'}")
print("\n" + "="*70)
print("All validation plots saved to Phase1/results/")
print("="*70)
