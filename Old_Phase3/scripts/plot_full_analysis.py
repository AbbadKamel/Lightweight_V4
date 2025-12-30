import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_full_analysis():
    print("Generating full analysis plots...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir_b = os.path.join(project_root, 'Phase3', 'results_B_mean_only')
    results_dir_c = os.path.join(project_root, 'Phase3', 'results_C_mean_std')
    output_dir = os.path.join(project_root, 'Phase3', 'comparison_plots')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_b = os.path.join(results_dir_b, 'canshield_evaluation_results.json')
    file_c = os.path.join(results_dir_c, 'canshield_evaluation_results.json')
    
    with open(file_b, 'r') as f: results_b = json.load(f)
    with open(file_c, 'r') as f: results_c = json.load(f)
    
    configs = sorted(results_b.keys())
    attack_types = ['plateau', 'continuous', 'playback', 'suppress']
    
    # 1. Detection Rate (Recall) for ALL 4 Attacks
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    x = np.arange(len(configs))
    width = 0.35
    
    for i, attack in enumerate(attack_types):
        ax = axes[i]
        rates_b = [results_b[c][attack]['recall'] * 100 for c in configs]
        rates_c = [results_c[c][attack]['recall'] * 100 for c in configs]
        
        rects1 = ax.bar(x - width/2, rates_b, width, label='Scenario B', color='skyblue')
        rects2 = ax.bar(x + width/2, rates_c, width, label='Scenario C', color='orange')
        
        ax.set_title(f'{attack.capitalize()} Attack Detection Rate')
        ax.set_ylabel('Recall (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_detection_rates.png'))
    print("Saved full_detection_rates.png")
    
    # 2. Accuracy Comparison (Average across all attacks)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate average accuracy per config
    acc_b = []
    acc_c = []
    for c in configs:
        avg_b = np.mean([results_b[c][at]['accuracy'] for at in attack_types]) * 100
        avg_c = np.mean([results_c[c][at]['accuracy'] for at in attack_types]) * 100
        acc_b.append(avg_b)
        acc_c.append(avg_c)
        
    rects1 = ax.bar(x - width/2, acc_b, width, label='Scenario B', color='lightgreen')
    rects2 = ax.bar(x + width/2, acc_c, width, label='Scenario C', color='salmon')
    
    ax.set_title('Average Accuracy Comparison (All Attacks)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_accuracy.png'))
    print("Saved average_accuracy.png")
    
    # 3. F1-Score Comparison (Average)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    f1_b = []
    f1_c = []
    for c in configs:
        avg_b = np.mean([results_b[c][at]['f1_score'] for at in attack_types]) * 100
        avg_c = np.mean([results_c[c][at]['f1_score'] for at in attack_types]) * 100
        f1_b.append(avg_b)
        f1_c.append(avg_c)
        
    rects1 = ax.bar(x - width/2, f1_b, width, label='Scenario B', color='lightblue')
    rects2 = ax.bar(x + width/2, f1_c, width, label='Scenario C', color='gold')
    
    ax.set_title('Average F1-Score Comparison')
    ax.set_ylabel('F1-Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_f1_score.png'))
    print("Saved average_f1_score.png")
    
    # 4. Confusion Matrix for Best Config (100s_10s) - Scenario C
    best_config = '100s_10s'
    if best_config in results_c:
        # Aggregate counts across all attacks for this config
        tp = sum(results_c[best_config][at]['tp'] for at in attack_types)
        fn = sum(results_c[best_config][at]['fn'] for at in attack_types)
        # FP and TN are the same for all attacks because they come from the same Normal set
        # But we summed TP/FN across 4 attacks, so we should effectively multiply FP/TN by 4 
        # to represent "Total Classification Events" if we treat each attack test as a separate experiment
        fp = results_c[best_config]['plateau']['fp'] * 4
        tn = results_c[best_config]['plateau']['tn'] * 4
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'], 
                    yticklabels=['Normal', 'Attack'])
        plt.title(f'Aggregated Confusion Matrix - Scenario C ({best_config})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_100s_10s_C.png'))
        print("Saved confusion_matrix_100s_10s_C.png")

if __name__ == "__main__":
    plot_full_analysis()
