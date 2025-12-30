import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_comparison():
    print("Generating comparison plots...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir_b = os.path.join(project_root, 'Phase3', 'results_B_mean_only')
    results_dir_c = os.path.join(project_root, 'Phase3', 'results_C_mean_std')
    
    file_b = os.path.join(results_dir_b, 'canshield_evaluation_results.json')
    file_c = os.path.join(results_dir_c, 'canshield_evaluation_results.json')
    
    if not os.path.exists(file_b) or not os.path.exists(file_c):
        print("Results files not found.")
        return
        
    with open(file_b, 'r') as f:
        results_b = json.load(f)
    with open(file_c, 'r') as f:
        results_c = json.load(f)
        
    # Configurations to plot
    configs = sorted(results_b.keys())
    attack_types = ['plateau', 'continuous', 'playback', 'suppress']
    
    # Create a plot for each configuration
    for config in configs:
        if config not in results_c:
            continue
            
        # Data for this config
        rates_b = [results_b[config][at]['detection_rate'] * 100 for at in attack_types]
        rates_c = [results_c[config][at]['detection_rate'] * 100 for at in attack_types]
        
        x = np.arange(len(attack_types))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, rates_b, width, label='Scenario B (Mean Only)')
        rects2 = ax.bar(x + width/2, rates_c, width, label='Scenario C (Mean + Std)')
        
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title(f'Detection Rate Comparison - {config}')
        ax.set_xticks(x)
        ax.set_xticklabels([at.capitalize() for at in attack_types])
        ax.set_ylim(0, 105)
        ax.legend()
        
        ax.bar_label(rects1, padding=3, fmt='%.1f')
        ax.bar_label(rects2, padding=3, fmt='%.1f')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(project_root, 'Phase3', 'comparison_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(os.path.join(output_dir, f'comparison_{config}.png'))
        plt.close()
        
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_comparison()
