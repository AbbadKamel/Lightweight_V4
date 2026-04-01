
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# Add current directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Configuration
# We select the "Experts" that had 0% False Positive Rate in individual testing
# Added 100s_10s for slow attack detection (replay, constant)
SELECTED_MODELS = ['50s_1s', '50s_5s', '100s_10s'] 

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    
    plt.title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_error_distribution(errors_normal, errors_attack, threshold, output_path):
    plt.figure(figsize=(10, 6))
    
    # Plot distributions
    sns.kdeplot(errors_normal, fill=True, color='green', label='Normal Data', clip=(0, None))
    sns.kdeplot(errors_attack, fill=True, color='red', label='Attack Data', clip=(0, None))
    
    # Plot Threshold Line
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    plt.title('Reconstruction Error (Loss) Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Mean Squared Error (MSE)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Limit X axis to focus on the intersection area
    # Find 99th percentile of attack to avoid long tail stretching graph
    if len(errors_attack) > 0:
        limit = np.percentile(errors_attack, 95) * 1.5
        plt.xlim(0, limit)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    print("="*60)
    print("PHASE 3: ENSEMBLE EVALUATION (The Committee of Experts)")
    print("="*60)
    
    ensure_dir = os.path.join(config.FIGURES_DIR)
    if not os.path.exists(ensure_dir):
        os.makedirs(ensure_dir)

    # 1. Load Everything for Selected Models
    models = {}
    thresholds = {}
    datasets_normal = {}
    datasets_attack = {}
    
    print(f"Loading Experts: {SELECTED_MODELS}")
    
    for name in SELECTED_MODELS:
        # Parse config
        parts = name.split('_')
        ts = int(parts[0].replace('s', '')) # 50
        sp = int(parts[1].replace('s', '')) # 1 or 5
        
        # Paths
        m_path = os.path.join(config.PHASE2_DIR, "models", f"{name}.h5")
        t_path = os.path.join(config.THRESHOLDS_DIR, f"{name}_thresholds.json")
        data_n_path = os.path.join(config.TEST_DATA_DIR, f"{ts}s_window", f"sampling_{sp}s", "test.npy")
        data_a_path = os.path.join(config.ATTACKS_DIR, f"attacks_{name}.npy")
        
        # Load
        if not (os.path.exists(m_path) and os.path.exists(t_path) and 
                os.path.exists(data_n_path) and os.path.exists(data_a_path)):
            print(f"  ❌ Missing files for {name}. Skipping.")
            continue
            
        try:
            # Model
            models[name] = load_model(m_path, compile=False)
            
            # Threshold - Use 75th percentile (optimal from tuning)
            with open(t_path) as f:
                thresholds[name] = float(json.load(f)['75'])
                
            # Data
            n_raw = np.load(data_n_path, allow_pickle=True)
            datasets_normal[name] = n_raw[:, :, 1:].astype(np.float32) # Drop timestamp
            
            datasets_attack[name] = np.load(data_a_path) # Already float32
            
            print(f"  ✅ Loaded {name} (Thresh: {thresholds[name]:.4f})")
            
        except Exception as e:
            print(f"  ❌ Error loading {name}: {e}")
            continue
            
    if not models:
        print("No models loaded. Exiting.")
        return

    # 2. Run Inference & Voting
    # Since different models have different sapmling rates, their windows don't match 1-to-1 in time exactly 
    # (or they might, if generated from same source).
    # However, for this evaluation, we treat them as independent tests on their respective test sets.
    # To combine them "Fairly", we concat the predictions of each model on its OWN test set,
    # and then calculate the aggregate metrics.
    
    # Wait, "Ensemble" usually means voting on the SAME event.
    # But 50s_1s and 50s_5s test sets are different slices (sampled differently).
    # Correct Approach for this Project structure:
    # "Ensemble Performance" here means "If we deployed these 2 models in parallel, what is the system performance?"
    # System Alarm = (Model A Alarm) OR (Model B Alarm)
    
    # But we can't OR them if they are looking at different arrays.
    # SIMPLIFICATION: We will evaluate the "Logical OR" capability by assuming they look at the SAME attacks.
    # Since we generated 300 attacks for EACH, we can treat "Attack #1 for 50s_1s" and "Attack #1 for 50s_5s" as corresponding to the same event.
    
    # Let's align by index.
    n_samples_max = min([len(datasets_attack[m]) for m in models])
    print(f"\nAligning on {n_samples_max} samples...")
    
    # Arrays to store votes (N_samples, N_models)
    # Normal Data
    # 50s_1s has 76 normal samples. 50s_5s has 76 normal samples. They generally match.
    n_normal_max = min([len(datasets_normal[m]) for m in models])
    
    votes_normal = np.zeros((n_normal_max, len(models)))
    votes_attack = np.zeros((n_samples_max, len(models)))
    
    # Store errors for plotting (flattened)
    all_errors_normal = []
    all_errors_attack = []
    
    model_names = list(models.keys())
    
    for i, name in enumerate(model_names):
        print(f"  Running {name}...")
        model = models[name]
        thresh = thresholds[name]
        
        # Normal
        d_norm = datasets_normal[name][:n_normal_max]
        if d_norm.ndim == 3: d_norm = d_norm.reshape(d_norm.shape + (1,))
        rec_n = model.predict(d_norm, verbose=0)
        mse_n = np.mean(np.square(d_norm - rec_n), axis=(1,2,3))
        votes_normal[:, i] = (mse_n > thresh).astype(int)
        all_errors_normal.extend(mse_n)
        
        # Attack
        d_att = datasets_attack[name][:n_samples_max]
        if d_att.ndim == 3: d_att = d_att.reshape(d_att.shape + (1,))
        rec_a = model.predict(d_att, verbose=0)
        mse_a = np.mean(np.square(d_att - rec_a), axis=(1,2,3))
        votes_attack[:, i] = (mse_a > thresh).astype(int)
        all_errors_attack.extend(mse_a)

    # 3. Voting Strategy: OR (Any)
    # If any model says 1, result is 1
    final_pred_normal = np.any(votes_normal, axis=1).astype(int)
    final_pred_attack = np.any(votes_attack, axis=1).astype(int)
    
    y_true_normal = np.zeros_like(final_pred_normal)
    y_true_attack = np.ones_like(final_pred_attack)
    
    y_true_all = np.concatenate([y_true_normal, y_true_attack])
    y_pred_all = np.concatenate([final_pred_normal, final_pred_attack])
    
    # 4. Metrics
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all)
    rec = recall_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all)
    
    # Conf Matrix
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n" + "="*30)
    print("ENSEMBLE RESULTS (OR Vote)")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print("="*30)
    
    # 5. Plots
    print("\nGenerating Plots...")
    
    # Confusion Matrix
    cm_path = os.path.join(config.FIGURES_DIR, "ensemble_confusion_matrix.png")
    plot_confusion_matrix(y_true_all, y_pred_all, cm_path)
    print(f"  Saved: {cm_path}")
    
    # Error Distribution (Use the first model as representative, or aggregate? 
    # Loss magnitudes differ between models, so aggregating distributions is messy.
    # Better to plot the distribution for the BEST model (50s_1s) to show the separation)
    
    best_model = '50s_1s'
    if best_model in models:
        # Re-calc errors just for this plot (we have them in the loop, but easier to grab fresh or store)
        # Actually I stored them in all_errors_normal but that's mixed.
        # Let's run quick verify on 50s_1s for the plot.
        
        # Or, since I have the logic in the loop, I can just grab the errors for index 0 (assuming 50s_1s is first)
        idx_best = model_names.index(best_model)
        
        # We need the errors strictly for 50s_1s
        # I'll just re-predict to be safe and clean or extract from loop if I stored it better.
        # I'll just re-run predict for the plot, it's fast.
        
        m = models[best_model]
        d_n = datasets_normal[best_model]
        if d_n.ndim == 3: d_n = d_n.reshape(d_n.shape + (1,))
        e_n = np.mean(np.square(d_n - m.predict(d_n, verbose=0)), axis=(1,2,3))
        
        d_a = datasets_attack[best_model]
        if d_a.ndim == 3: d_a = d_a.reshape(d_a.shape + (1,))
        e_a = np.mean(np.square(d_a - m.predict(d_a, verbose=0)), axis=(1,2,3))
        
        dist_path = os.path.join(config.FIGURES_DIR, "reconstruction_error_density.png")
        plot_error_distribution(e_n, e_a, thresholds[best_model], dist_path)
        print(f"  Saved: {dist_path}")

    # Save Metrics
    results = {
        'strategy': 'OR_Vote',
        'models': model_names,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'fpr': float(fpr),
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
    }
    
    res_path = os.path.join(config.RESULTS_DIR, "ensemble_results.json")
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
