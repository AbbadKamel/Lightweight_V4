import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'Phase0', 'results', 'decoded_frames.csv')
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, 'results', 'speed_distribution.png')

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Select Speed Signal (using speed_water as primary, but let's check SOG too if needed)
# The user asked for "speed", usually implies Speed Through Water or SOG. 
# Given previous context, speed_water had 0.78% coverage and SOG had 9.35%. 
# SOG is likely the more reliable "speed" here, but I will analyze BOTH to be sure.

signals_to_check = ['speed_water', 'sog']

for signal in signals_to_check:
    print(f"\n--- Analyzing {signal} ---")
    series = df[signal].dropna()
    
    if len(series) == 0:
        print(f"No data for {signal}")
        continue

    # 1. Calculate Mean and Std
    mean_val = series.mean()
    std_val = series.std()
    
    print(f"Mean: {mean_val:.1f}")
    print(f"Std : {std_val:.1f}")
    
    # 2. Value Distribution (Rounded to 1 decimal)
    print(f"\nValue Distribution (Rounded to 1 decimal place):")
    # Round to 1 decimal
    series_rounded = series.round(1)
    # Count occurrences
    distribution = series_rounded.value_counts().sort_index()
    
    # Print top values or full distribution if small
    print(f"{'Value':<10} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 35)
    total_count = len(series)
    for value, count in distribution.items():
        percentage = (count / total_count) * 100
        print(f"{value:<10.1f} | {count:<10} | {percentage:<10.1f}%")

    # 3. Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(series, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribution of {signal}\nMean: {mean_val:.1f}, Std: {std_val:.1f}")
    plt.xlabel("Speed (knots)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SCRIPT_DIR, 'results', f'{signal}_distribution.png'))
    print(f"Plot saved to Phase1/results/{signal}_distribution.png")
