import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project Root is one level up
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'Phase0', 'results', 'decoded_frames.csv')
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, 'results', 'selected_signals_correlation.png')

# The 16 Selected Signals
SELECTED_SIGNALS = [
    'latitude', 'longitude', 'depth', 
    'rudder_angle_order', 'rudder_position', 
    'cog', 'sog', 'speed_water', 
    'yaw', 'pitch', 'roll', 'heading', 
    'variation', 'rate_of_turn', 
    'wind_speed', 'wind_angle'
]

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Clean Timestamp
print("Cleaning timestamps...")
df['timestamp'] = df['timestamp'].astype(str).str.replace(r'\.0$', '', regex=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
df.set_index('timestamp', inplace=True)

# Filter for only selected signals
print(f"Filtering for {len(SELECTED_SIGNALS)} selected signals...")
df_selected = df[SELECTED_SIGNALS]

# Resample to 1s to align data (Correlation requires aligned timestamps)
print("Resampling to 1s to align timestamps for correlation...")
df_resampled = df_selected.resample('1s').mean()

# Calculate Correlation Matrix
print("Calculating correlation matrix...")
corr_matrix = df_resampled.corr()

# Plotting
print("Generating heatmap...")
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle (redundant)

sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r', 
            center=0, 
            vmin=-1, 
            vmax=1,
            square=True, 
            linewidths=0.5, 
            cbar_kws={"shrink": 0.8})

plt.title('Correlation Matrix of Selected Signals (1s Resampled)', fontsize=16, pad=20)
plt.tight_layout()

# Save
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"Correlation heatmap saved to: {OUTPUT_PLOT}")
