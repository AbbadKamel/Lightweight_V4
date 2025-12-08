import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# Configuration
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project Root is one level up
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'Phase0', 'results', 'decoded_frames.csv')
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, 'results', 'signal_statistics_report.md')
PLOT_DIR = os.path.join(SCRIPT_DIR, 'results', 'plots', 'signals')
GRID_PLOT_OUTPUT = os.path.join(SCRIPT_DIR, 'results', 'all_signals_raw_grid.png')

# Ensure directories exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Clean Timestamp
print("Cleaning timestamps...")
df['timestamp'] = df['timestamp'].astype(str).str.replace(r'\.0$', '', regex=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
df.set_index('timestamp', inplace=True)

# Identify Signal Columns (Exclude metadata)
metadata_cols = ['pgn', 'pgn_name']
signal_cols = [c for c in df.columns if c not in metadata_cols]

print(f"Found {len(signal_cols)} potential signals.")

# --- PART 1: Statistics Report & Individual Plots ---
print("\n--- Generating Statistics Report & Individual Plots ---")

report_lines = []
report_lines.append("# Comprehensive Signal Analysis Report")
report_lines.append(f"**Total Rows in Dataset:** {len(df)}")
report_lines.append(f"**Time Range:** {df.index.min()} to {df.index.max()}")
report_lines.append(f"**Duration:** {df.index.max() - df.index.min()}\n")
report_lines.append("| Signal Name | Coverage (%) | Mean | Std Dev | Min | Max | Unique Values | Status |")
report_lines.append("|---|---|---|---|---|---|---|---|")

for col in signal_cols:
    print(f"Analyzing {col}...")
    series = df[col].dropna()
    
    # Statistics
    count = len(series)
    coverage = (count / len(df)) * 100
    
    if count == 0:
        report_lines.append(f"| {col} | 0.00% | - | - | - | - | 0 | **EMPTY** |")
        continue
        
    mean_val = series.mean()
    std_val = series.std()
    min_val = series.min()
    max_val = series.max()
    unique_count = series.nunique()
    
    # Determine Status
    status = "**GOOD**"
    if unique_count == 1:
        status = "CONSTANT (Useless)"
    elif coverage < 0.1:
        status = "RARE (Sparse)"
    
    report_lines.append(f"| {col} | {coverage:.2f}% | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} | {unique_count} | {status} |")
    
    # Individual Plot
    plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values, '.', markersize=2, alpha=0.5, label='Raw Data')
    
    # Add a rolling mean trendline if enough data
    if count > 100:
        trend = series.resample('1s').mean()
        plt.plot(trend.index, trend.values, '-', color='red', linewidth=1, label='1s Trend')
        
    plt.title(f"Signal Evolution: {col} (Coverage: {coverage:.1f}%)")
    plt.ylabel(col)
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save Plot
    plot_path = f"{PLOT_DIR}/{col}.png"
    plt.savefig(plot_path)
    plt.close()

# Write Report
with open(OUTPUT_REPORT, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"Report saved to: {OUTPUT_REPORT}")

# --- PART 2: Grid Plot (All Signals) ---
print("\n--- Generating Grid Plot of All Signals ---")

# Setup Grid (5x4 = 20 plots, enough for 19 signals)
rows = 5
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 25), sharex=False)
axes = axes.flatten()

for i, signal in enumerate(signal_cols):
    ax = axes[i]
    series = df[signal].dropna()
    
    if len(series) > 0:
        # Plot raw data points connected by a fine line
        ax.plot(series.index, series.values, linestyle='-', linewidth=0.2, marker='.', markersize=1, alpha=0.7, color='gray')
        ax.set_title(signal, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format X-Axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    else:
        ax.text(0.5, 0.5, 'NO DATA / EMPTY', ha='center', va='center', color='red', fontweight='bold')
        ax.set_title(signal, fontsize=10, fontweight='bold', color='red')

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(GRID_PLOT_OUTPUT, dpi=150)
plt.close()
print(f"Grid plot saved to: {GRID_PLOT_OUTPUT}")

print("\nAll tasks complete!")
