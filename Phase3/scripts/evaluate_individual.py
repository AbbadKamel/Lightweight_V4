
import os
import sys
import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# Add current directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def evaluate_model(model, threshold, normal_data, attack_data):
    """
    Evaluate a single model on mixed data (Normal + Attack).
    """
    # 1. Evaluate on Normal Data (Label = 0)
    # ---------------------------------------
    # Predict
    # Normal data shape: (N, Time, Feat) -> Expand to (N, Time, Feat, 1)
    if normal_data.ndim == 3:
        normal_input = normal_data.reshape(normal_data.shape[0], normal_data.shape[1], normal_data.shape[2], 1)
    else:
        normal_input = normal_data
        
    recon_normal = model.predict(normal_input, verbose=0)
    mse_normal = np.mean(np.square(normal_input - recon_normal), axis=(1, 2, 3))
    
    # Decisions (0 = Normal, 1 = Attack)
    pred_normal = (mse_normal > threshold).astype(int)
    true_normal = np.zeros(len(pred_normal), dtype=int)
    
    # 2. Evaluate on Attack Data (Label = 1)
    # --------------------------------------
    if attack_data.ndim == 3:
        attack_input = attack_data.reshape(attack_data.shape[0], attack_data.shape[1], attack_data.shape[2], 1)
    else:
        attack_input = attack_data

    recon_attack = model.predict(attack_input, verbose=0)
    mse_attack = np.mean(np.square(attack_input - recon_attack), axis=(1, 2, 3))
    
    # Decisions
    pred_attack = (mse_attack > threshold).astype(int)
    true_attack = np.ones(len(pred_attack), dtype=int)
    
    # 3. Combine Results
    # ------------------
    y_true = np.concatenate([true_normal, true_attack])
    y_pred = np.concatenate([pred_normal, pred_attack])
    all_errors = np.concatenate([mse_normal, mse_attack])
    
    # 4. Calculate Metrics
    # --------------------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'samples_normal': int(len(normal_data)),
        'samples_attack': int(len(attack_data)),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return metrics

def main():
    print("="*60)
    print("PHASE 3: INDIVIDUAL MODEL EVALUATION")
    print("="*60)
    
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
        
    # Find all test data files to identify configs
    configs = []
    # Using the manually defined list from our knowledge or scanning
    # Let's scan for robustness (looking for test.npy)
    for time_step in [50, 75, 100]:
        for sampling in [1, 5, 10]:
            path = os.path.join(config.TEST_DATA_DIR, f"{time_step}s_window", f"sampling_{sampling}s", "test.npy")
            if os.path.exists(path):
                configs.append((time_step, sampling, path))
    
    print(f"Found {len(configs)} configurations to evaluate.")
    
    results = {}
    
    for time_step, sampling, test_path in configs:
        config_name = f"{time_step}s_{sampling}s"
        print(f"\nEvaluating {config_name}...")
        
        # 1. Load Threshold (Clean Phase 3)
        thresh_path = os.path.join(config.THRESHOLDS_DIR, f"{config_name}_thresholds.json")
        if not os.path.exists(thresh_path):
            print(f"  ⚠️ Skipping: Threshold file missing ({thresh_path})")
            continue
            
        with open(thresh_path, 'r') as f:
            t_data = json.load(f)
            threshold = float(t_data['threshold'])
            
        print(f"  Threshold: {threshold:.6f}")
        
        # 2. Load Model (Phase 2)
        model_path = os.path.join(config.PHASE2_DIR, "models", f"{config_name}.h5")
        if not os.path.exists(model_path):
            print(f"  ⚠️ Skipping: Model file missing ({model_path})")
            continue
            
        try:
            model = load_model(model_path, compile=False)
        except Exception as e:
            print(f"  ⚠️ Error loading model: {e}")
            continue

        # 3. Load Normal Data
        try:
            d_raw = np.load(test_path, allow_pickle=True)
            normal_data = d_raw[:, :, 1:].astype(np.float32) # Drop timestamp
        except Exception as e:
            print(f"  ⚠️ Error loading normal data: {e}")
            continue
            
        # 4. Load Attack Data (Generated in Step 2)
        attack_path = os.path.join(config.ATTACKS_DIR, f"attacks_{config_name}.npy")
        if not os.path.exists(attack_path):
            print(f"  ⚠️ Skipping: Attack data missing ({attack_path})")
            continue
            
        try:
            attack_data = np.load(attack_path)
            # attack_data is already (N, Time, Feat, 1) or (N, Time, Feat)
        except Exception as e:
            print(f"  ⚠️ Error loading attack data: {e}")
            continue
            
        # 5. Evaluate
        metrics = evaluate_model(model, threshold, normal_data, attack_data)
        
        results[config_name] = metrics
        
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  FPR:      {metrics['fpr']:.4f}")
        
        if metrics['fpr'] > 0.10:
            print("  ❌ HIGH FALSE ALARM RATE")
        elif metrics['f1_score'] > 0.90:
            print("  ✅ EXCELLENT PERFORMANCE")
            
    # Save Results
    out_path = os.path.join(config.RESULTS_DIR, "individual_evaluation_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
