
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_thresholds():
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    THRESHOLDS_DIR = os.path.join(BASE_DIR, 'thresholds')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Find all threshold files
    files = glob.glob(os.path.join(THRESHOLDS_DIR, "*_thresholds.json"))
    
    data = []
    
    print(f"Loading {len(files)} threshold files...")
    
    for f_path in files:
        filename = os.path.basename(f_path)
        # Parse config name: e.g., "50s_1s_thresholds.json"
        parts = filename.replace('_thresholds.json', '').split('_')
        window_size = parts[0]
        sampling_period = parts[1]
        
        with open(f_path, 'r') as f:
            content = json.load(f)
            # Handle simple format {"threshold": X}
            val = float(content['threshold'])
            
            data.append({
                'Window Size': window_size,
                'Sampling': sampling_period,
                'Threshold (MSE)': val,
                'Config': f"{window_size}_{sampling_period}"
            })
            
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort for consistent plotting order
    # Extract numeric values for sorting
    df['w_num'] = df['Window Size'].str.replace('s','').astype(int)
    df['s_num'] = df['Sampling'].str.replace('s','').astype(int)
    df = df.sort_values(['w_num', 's_num'])
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Bar chart grouped by Window Size, colored by Sampling Period
    ax = sns.barplot(
        data=df, 
        x='Window Size', 
        y='Threshold (MSE)', 
        hue='Sampling',
        palette='viridis'
    )
    
    # Add values on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=10, fontweight='bold')

    plt.title('Phase 3 Detection Thresholds (Max MSE on Validation)', fontsize=16, fontweight='bold')
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.xlabel('Window Size', fontsize=14)
    plt.legend(title='Sampling Period')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'thresholds_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    plot_thresholds()
