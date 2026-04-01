
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SELECTED_MODELS = ['50s_1s', '50s_5s', '100s_10s'] 
# Extended range to find optimal trade-off
STRATEGIES = ['max', '99', '95', '90', '85', '80', '75']

def evaluate_ensemble(models, thresholds_dict, strategy_name, datasets_normal, datasets_attack, n_limit):
    """
    Run Ensemble OR Voting for a specific threshold strategy.
    """
    votes_normal = []
    votes_attack = []
    
    # Collect votes
    for name, model in models.items():
        thresh = thresholds_dict[name][strategy_name]
        
        # Normal
        d_n = datasets_normal[name]
        rec_n = model.predict(d_n, verbose=0)
        mse_n = np.mean(np.square(d_n - rec_n), axis=(1,2,3))
        votes_normal.append((mse_n > thresh).astype(int))
        
        # Attack
        d_a = datasets_attack[name][:n_limit]
        rec_a = model.predict(d_a, verbose=0)
        mse_a = np.mean(np.square(d_a - rec_a), axis=(1,2,3))
        votes_attack.append((mse_a > thresh).astype(int))
        
    # Voting (OR)
    votes_normal = np.array(votes_normal).T # (Samples, Models)
    votes_attack = np.array(votes_attack).T
    
    final_n = np.any(votes_normal, axis=1).astype(int)
    final_a = np.any(votes_attack, axis=1).astype(int)
    
    # Concatenate
    y_true = np.concatenate([np.zeros_like(final_n), np.ones_like(final_a)])
    y_pred = np.concatenate([final_n, final_a])
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0
    }

def main():
    print("="*60)
    print("PHASE 3: THRESHOLD OPTIMIZATION (Bake-off)")
    print("="*60)
    
    # Load Models & Data ONCE
    models = {}
    thresholds_data = {}
    datasets_normal = {}
    datasets_attack = {}
    
    print("Loading resources...")
    for name in SELECTED_MODELS:
        # Load Model
        m_path = os.path.join(config.PHASE2_DIR, "models", f"{name}.h5")
        models[name] = load_model(m_path, compile=False)
        
        # Load Thresholds Data (Dict)
        t_path = os.path.join(config.THRESHOLDS_DIR, f"{name}_thresholds.json")
        with open(t_path) as f:
            thresholds_data[name] = json.load(f)
            
        # Load Data
        parts = name.split('_')
        ts, sp = int(parts[0][:-1]), int(parts[1][:-1])
        
        d_n = np.load(os.path.join(config.TEST_DATA_DIR, f"{ts}s_window", f"sampling_{sp}s", "test.npy"), allow_pickle=True)
        datasets_normal[name] = d_n[:, :, 1:].astype(np.float32).reshape(d_n.shape[0], d_n.shape[1], -1, 1)
        
        d_a = np.load(os.path.join(config.ATTACKS_DIR, f"attacks_{name}.npy"))
        datasets_attack[name] = d_a
        
    n_limit = min([len(datasets_attack[m]) for m in models])
    n_normal_limit = min([len(datasets_normal[m]) for m in models])
    
    # Truncate normal to same size for fair loading?
    # No, evaluate on all available normal.
    for name in models:
        datasets_normal[name] = datasets_normal[name][:n_normal_limit]
        
    results = []
    
    print(f"\nComparing Strategies on {n_limit} Attacks and {n_normal_limit} Normal samples...")
    print("-" * 65)
    print(f"{'Strategy':<10} | {'F1 Score':<10} | {'Recall':<10} | {'Precision':<10} | {'FPR':<10}")
    print("-" * 65)
    
    for strat in STRATEGIES:
        metrics = evaluate_ensemble(models, thresholds_data, strat, datasets_normal, datasets_attack, n_limit)
        metrics['Strategy'] = strat
        results.append(metrics)
        print(f"{strat:<10} | {metrics['F1 Score']:.4f}     | {metrics['Recall']:.4f}     | {metrics['Precision']:.4f}     | {metrics['FPR']:.4f}")

    print("-" * 65)
    
    # Plotting
    import pandas as pd
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    
    # Plot F1, Recall, Precision lines
    plt.plot(df['Strategy'], df['F1 Score'], marker='o', label='F1 Score', linewidth=2)
    plt.plot(df['Strategy'], df['Recall'], marker='s', label='Recall (Sensitivity)', linewidth=2)
    plt.plot(df['Strategy'], df['Precision'], marker='^', label='Precision', linewidth=2, linestyle='--')
    
    plt.title('Threshold Strategy Trade-off', fontsize=14, fontweight='bold')
    plt.xlabel('Threshold Strictness (Percentile)', fontsize=12)
    plt.ylabel('Score (0-1)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(config.FIGURES_DIR, "threshold_optimization_plot.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved plot to: {out_path}")
    
    # Recommendation
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    print(f"\n🏆 Recommended Strategy: {best_f1['Strategy']}")
    print(f"   F1: {best_f1['F1 Score']:.4f}, Recall: {best_f1['Recall']:.4f}, FPR: {best_f1['FPR']:.4f}")

if __name__ == "__main__":
    main()
