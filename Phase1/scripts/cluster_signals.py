import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'Phase0', 'results', 'decoded_frames.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
DENDROGRAM_PLOT = os.path.join(OUTPUT_DIR, 'signal_dendrogram.png')
CLUSTERED_HEATMAP = os.path.join(OUTPUT_DIR, 'correlation_matrix_clustered.png')
SIGNAL_ORDER_FILE = os.path.join(OUTPUT_DIR, 'optimal_signal_order.json')

# The 16 Selected Signals
SELECTED_SIGNALS = [
    'latitude', 'longitude', 'depth', 
    'rudder_angle_order', 'rudder_position', 
    'cog', 'sog', 'speed_water', 
    'yaw', 'pitch', 'roll', 'heading', 
    'variation', 'rate_of_turn', 
    'wind_speed', 'wind_angle'
]

print("="*80)
print("SIGNAL CLUSTERING ANALYSIS")
print("="*80)

# Load and prepare data
print("\n1. Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Clean timestamps
df['timestamp'] = df['timestamp'].astype(str).str.replace(r'\.0$', '', regex=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
df.set_index('timestamp', inplace=True)

# Filter for selected signals
df_selected = df[SELECTED_SIGNALS]

# Resample to 1s to align data
print("2. Resampling to 1s for correlation calculation...")
df_resampled = df_selected.resample('1s').mean()

# Calculate correlation matrix
print("3. Calculating correlation matrix...")
corr_matrix = df_resampled.corr()

# Convert correlation to distance
print("4. Converting correlation to distance matrix...")
# Distance = 1 - |correlation|
distance_matrix = 1 - np.abs(corr_matrix.values)

# Ensure distance matrix is valid (symmetric, non-negative)
distance_matrix = np.clip(distance_matrix, 0, None)
np.fill_diagonal(distance_matrix, 0)

# Convert to condensed form for scipy
condensed_dist = squareform(distance_matrix)

# Perform hierarchical clustering
print("5. Performing hierarchical clustering...")
linkage_matrix = linkage(condensed_dist, method='ward')

# Get optimal leaf order
print("6. Determining optimal signal order...")
optimal_indices = leaves_list(linkage_matrix)
clustered_order = [SELECTED_SIGNALS[i] for i in optimal_indices]

# Print results
print("\n" + "="*80)
print("CLUSTERING RESULTS")
print("="*80)

print("\nOptimal Signal Order (Data-Driven Clustering):")
for i, signal in enumerate(clustered_order, 1):
    print(f"  {i:2d}. {signal}")

# Save clustered order
print("\n7. Saving optimal signal order...")
output_data = {
    "optimal_order": clustered_order,
    "description": "Data-driven clustering order based on actual correlations in the dataset. Signals that move together are grouped together for optimal CNN learning."
}

with open(SIGNAL_ORDER_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"   ✓ Saved: {SIGNAL_ORDER_FILE}")

# Plot 1: Dendrogram
print("\n8. Generating dendrogram...")
plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, 
           labels=SELECTED_SIGNALS,
           leaf_rotation=90,
           leaf_font_size=10)
plt.title('Signal Clustering Dendrogram\n(Lower height = more similar signals)', 
          fontsize=14, pad=20)
plt.xlabel('Signal Name', fontsize=12)
plt.ylabel('Distance (Dissimilarity)', fontsize=12)
plt.tight_layout()
plt.savefig(DENDROGRAM_PLOT, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {DENDROGRAM_PLOT}")

# Plot 2: Reordered correlation matrix
print("\n9. Generating clustered correlation heatmap...")
# Reorder correlation matrix based on clustering
corr_clustered = corr_matrix.iloc[optimal_indices, optimal_indices]

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_clustered, dtype=bool))
sns.heatmap(corr_clustered,
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
plt.title('Correlation Matrix (Clustered Order)\nSimilar signals grouped together', 
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(CLUSTERED_HEATMAP, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {CLUSTERED_HEATMAP}")

# Identify clusters by cutting dendrogram at a threshold
from scipy.cluster.hierarchy import fcluster
# Cut at distance threshold to get clusters
max_dist = 0.5  # Adjust this to get more/fewer clusters
cluster_labels = fcluster(linkage_matrix, max_dist, criterion='distance')

print("\n10. Identified Signal Clusters:")
clusters = {}
for signal, label in zip(SELECTED_SIGNALS, cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(signal)

for cluster_id, signals in sorted(clusters.items()):
    print(f"\n   Cluster {cluster_id}:")
    for signal in signals:
        print(f"     - {signal}")

print("\n" + "="*80)
print("CLUSTERING COMPLETE!")
print("="*80)
print("\nNext Steps:")
print("  1. Review the dendrogram plot to understand signal relationships")
print("  2. Review the clustered correlation matrix (blocks along diagonal)")
print("  3. Use the optimal_order when creating the Master Table")
print("\n" + "="*80)
