import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration
INPUT_FILE = 'Phase0/results/decoded_frames.csv'
OUTPUT_PLOT = 'Phase1/selected_signals_grid.png'

# --- SELECTION RATIONALE ---
# We selected these 16 signals because:
# 1. High Coverage: They appear frequently enough to be useful.
# 2. High Variance: They show meaningful movement (Std > 0).
# 3. Criticality: They represent core navigation (Pos, Speed, Rudder, Wind).
#
# We DROPPED:
# - speed_ground (Empty)
# - deviation (Empty)
# - offset (Constant 0.0)
# ---------------------------

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

print(f"Plotting {len(SELECTED_SIGNALS)} SELECTED signals...")

# Setup Grid (4x4 = 16 plots)
rows = 4
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 20), sharex=False)
axes = axes.flatten()

for i, signal in enumerate(SELECTED_SIGNALS):
    ax = axes[i]
    
    # Get data for this signal
    series = df[signal].dropna()
    
    if len(series) > 0:
        # Plot raw data points connected by a fine line (Blue for Selected)
        ax.plot(series.index, series.values, linestyle='-', linewidth=0.2, marker='.', markersize=1, alpha=0.7, color='blue')
        
        ax.set_title(f"{signal} (SELECTED)", fontsize=10, fontweight='bold', color='green')
        ax.grid(True, alpha=0.3)
        
        # Format X-Axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    else:
        ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center')

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"Grid plot of SELECTED signals saved to: {OUTPUT_PLOT}")
