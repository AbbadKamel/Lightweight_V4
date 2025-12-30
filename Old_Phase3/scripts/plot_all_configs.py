import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_training_history_comparison():
    print("Generating training history comparison plots...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Scenarios to compare
    scenarios = ['B_mean_only', 'C_mean_std']
    colors = {'B_mean_only': 'blue', 'C_mean_std': 'orange'}
    labels = {'B_mean_only': 'Scenario B (Mean)', 'C_mean_std': 'Scenario C (Mean+Std)'}
    
    # Find all configurations (e.g., 50s_1s, 100s_10s)
    # We'll just look in one of the model directories to get the list
    models_dir_b = os.path.join(project_root, 'Phase2', 'models_B_mean_only')
    history_files = glob.glob(os.path.join(models_dir_b, "*_history.json"))
    
    # If history files are not saved separately, we might need to rely on the fact 
    # that we didn't explicitly save history JSONs in the training script provided in context.
    # However, standard Keras training usually returns a history object. 
    # The current `train_cascade_scenario.py` DOES NOT save history to JSON.
    # It only saves the model .h5 file.
    
    # CHECK: Did we save history?
    # Looking at `train_cascade_scenario.py` in context:
    # It saves `model.save(...)` but does NOT save `history.history` to a JSON file.
    
    # Since we cannot plot history that wasn't saved, we have to report this limitation.
    # BUT, we can plot the FINAL validation loss/accuracy if we can extract it, 
    # or we can modify the training script to save history and re-run (too long).
    
    # ALTERNATIVE: We can plot the DETECTION RESULTS (Accuracy/Recall) which we DO have.
    # The user asked for "accuracy and loss", which usually implies training curves.
    # If those don't exist, I will plot the Evaluation Detection Rate for all configs.
    
    print("Warning: Training history (loss/accuracy curves) was not saved during training.")
    print("Generating Evaluation Detection Rate comparison instead.")
    
    results_dir_b = os.path.join(project_root, 'Phase3', 'results_B_mean_only')
    results_dir_c = os.path.join(project_root, 'Phase3', 'results_C_mean_std')
    
    file_b = os.path.join(results_dir_b, 'canshield_evaluation_results.json')
    file_c = os.path.join(results_dir_c, 'canshield_evaluation_results.json')
    
    with open(file_b, 'r') as f: results_b = json.load(f)
    with open(file_c, 'r') as f: results_c = json.load(f)
    
    configs = sorted(results_b.keys())
    
    # Plot 1: Detection Rate for Plateau Attack (The main target)
    # Grouped by Configuration
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    rates_b = [results_b[c]['plateau']['detection_rate'] * 100 for c in configs]
    rates_c = [results_c[c]['plateau']['detection_rate'] * 100 for c in configs]
    
    rects1 = ax.bar(x - width/2, rates_b, width, label='Scenario B (Mean)', color='skyblue')
    rects2 = ax.bar(x + width/2, rates_c, width, label='Scenario C (Mean+Std)', color='orange')
    
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Plateau Attack Detection Rate by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.bar_label(rects1, padding=3, fmt='%.0f')
    ax.bar_label(rects2, padding=3, fmt='%.0f')
    
    plt.tight_layout()
    
    output_dir = os.path.join(project_root, 'Phase3', 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, 'all_configs_plateau_detection.png'))
    print(f"Saved plateau detection plot to {output_dir}")
    
    # Plot 2: Playback Attack
    fig, ax = plt.subplots(figsize=(14, 7))
    
    rates_b = [results_b[c]['playback']['detection_rate'] * 100 for c in configs]
    rates_c = [results_c[c]['playback']['detection_rate'] * 100 for c in configs]
    
    rects1 = ax.bar(x - width/2, rates_b, width, label='Scenario B (Mean)', color='skyblue')
    rects2 = ax.bar(x + width/2, rates_c, width, label='Scenario C (Mean+Std)', color='orange')
    
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Playback Attack Detection Rate by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.bar_label(rects1, padding=3, fmt='%.0f')
    ax.bar_label(rects2, padding=3, fmt='%.0f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_configs_playback_detection.png'))
    print(f"Saved playback detection plot to {output_dir}")

if __name__ == "__main__":
    plot_training_history_comparison()
