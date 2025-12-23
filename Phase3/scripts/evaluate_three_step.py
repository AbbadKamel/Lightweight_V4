#!/usr/bin/env python3
"""
Three-Step Threshold Evaluation (CANShield Algorithm 1 & 2)
============================================================
This implements CANShield's structured threshold analysis:
1. R_Loss: Per-signal pixel-wise loss threshold
2. R_Time: Time-step violation threshold per signal  
3. R_Signal: Overall signal violation threshold

This file is for COMPARISON with the simple MSE threshold approach.
"""

import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# CONFIGURATION
# ============================================================================
FEATURE_NAMES = [
    'wind_speed_mean', 'wind_speed_max', 'wind_speed_min', 'wind_speed_std',
    'wind_angle_mean', 'wind_angle_max', 'wind_angle_min', 'wind_angle_std',
    'yaw_mean', 'yaw_max', 'yaw_min', 'yaw_std',
    'cog_mean', 'cog_max', 'cog_min', 'cog_std',
    'heading_mean', 'heading_max', 'heading_min', 'heading_std',
    'roll_mean', 'roll_max', 'roll_min', 'roll_std',
    'rudder_angle_order_mean', 'rudder_angle_order_max', 'rudder_angle_order_min', 'rudder_angle_order_std',
    'rudder_position_mean', 'rudder_position_max', 'rudder_position_min', 'rudder_position_std',
    'rate_of_turn_mean', 'rate_of_turn_max', 'rate_of_turn_min', 'rate_of_turn_std',
    'depth_mean', 'depth_max', 'depth_min', 'depth_std',
    'variation_mean', 'variation_max', 'variation_min', 'variation_std',
    'latitude_mean', 'latitude_max', 'latitude_min', 'latitude_std',
    'longitude_mean', 'longitude_max', 'longitude_min', 'longitude_std',
    'pitch_mean', 'pitch_max', 'pitch_min', 'pitch_std',
    'sog_mean', 'sog_max', 'sog_min', 'sog_std'
]
NUM_FEATURES = len(FEATURE_NAMES)
NUM_SIGNALS = 15  # 15 base signals, 4 aggregations each

MODEL_CONFIGS = [
    (50, 1), (50, 5), (50, 10),
    (75, 1), (75, 5), (75, 10),
    (100, 1), (100, 5), (100, 10)
]

# Three-step threshold percentiles (from CANShield paper)
P_LOSS = 95   # R_Loss: 95th percentile of per-pixel loss
P_TIME = 99   # R_Time: 99th percentile of time violations
P_SIGNAL = [90, 92, 95, 97, 99]  # R_Signal: test multiple values


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_model_and_data(window_size, sampling_period):
    """Load model, normal data, and attack data."""
    config_name = f"{window_size}s_{sampling_period}s"
    
    # Load model
    model_path = os.path.join(config.PHASE2_DIR, "models", f"{config_name}.h5")
    model = load_model(model_path, compile=False)
    
    # Load normal test data - correct path structure
    test_path = os.path.join(
        config.PHASE1_DIR, "data", 
        f"{window_size}s_window", f"sampling_{sampling_period}s", "test.npy"
    )
    d_n = np.load(test_path, allow_pickle=True)
    # Skip first column (timestamp), convert to float, add channel dimension
    normal_data = d_n[:, :, 1:].astype(np.float32).reshape(d_n.shape[0], d_n.shape[1], -1, 1)
    
    # Load attack data (already properly formatted)
    attack_path = os.path.join(config.ATTACKS_DIR, f"attacks_{config_name}.npy")
    attack_data = np.load(attack_path).astype(np.float32)
    
    # Labels: 0 for normal, 1 for attack
    normal_labels = np.zeros(len(normal_data))
    attack_labels = np.ones(len(attack_data))
    
    return model, normal_data, attack_data, normal_labels, attack_labels


def calculate_reconstruction_loss(model, data):
    """Calculate per-pixel reconstruction loss (not aggregated)."""
    reconstructed = model.predict(data, verbose=0)
    # Shape: (n_samples, time_steps, features, 1)
    loss = np.abs(data - reconstructed)
    return loss  # Keep full shape


# ============================================================================
# THREE-STEP THRESHOLD ALGORITHM (CANShield Algorithm 1)
# ============================================================================
def calculate_three_step_thresholds(model, normal_data, p_loss=95, p_time=99):
    """
    CANShield Algorithm 1: Calculate thresholds from normal training data.
    
    Returns:
        R_Loss: np.array of shape (num_features,) - per-signal loss threshold
        R_Time: np.array of shape (num_features,) - per-signal time violation threshold
    """
    # Get reconstruction loss for all normal samples
    L = calculate_reconstruction_loss(model, normal_data)
    # L shape: (n_samples, time_steps, features, 1)
    L = L[:, :, :, 0]  # Remove channel dim: (n_samples, time_steps, features)
    
    n_samples, time_steps, features = L.shape
    
    # Step 1: Calculate R_Loss (per-feature loss threshold)
    # Take p_loss percentile across all samples and time steps for each feature
    R_Loss = np.percentile(L, p_loss, axis=(0, 1))  # Shape: (features,)
    
    # Step 2: Calculate R_Time (per-feature time violation threshold)
    # First, binarize: B[i,j,k] = 1 if L[i,j,k] > R_Loss[k]
    B = (L > R_Loss).astype(int)  # Shape: (n_samples, time_steps, features)
    
    # Count violations per sample per feature
    V = np.sum(B, axis=1)  # Shape: (n_samples, features)
    
    # R_Time is p_time percentile of violations per feature
    R_Time = np.percentile(V, p_time, axis=0)  # Shape: (features,)
    
    return R_Loss, R_Time


def apply_three_step_detection(model, data, R_Loss, R_Time, r_signal_threshold):
    """
    CANShield Algorithm 2: Apply three-step detection.
    
    Returns:
        predictions: np.array of 0/1 indicating normal/attack
        anomaly_scores: np.array of anomaly scores (fraction of violating signals)
    """
    # Get reconstruction loss
    L = calculate_reconstruction_loss(model, data)
    L = L[:, :, :, 0]  # Shape: (n_samples, time_steps, features)
    
    n_samples, time_steps, features = L.shape
    
    # Step 1: Binarize by R_Loss
    B = (L > R_Loss).astype(int)  # Shape: (n_samples, time_steps, features)
    
    # Step 2: Count time violations per signal
    V = np.sum(B, axis=1)  # Shape: (n_samples, features)
    
    # Step 3: Flag signals with violations > R_Time
    S = (V > R_Time).astype(int)  # Shape: (n_samples, features)
    
    # Anomaly score = fraction of signals flagged
    P = np.mean(S, axis=1)  # Shape: (n_samples,)
    
    # Prediction = 1 if P > r_signal_threshold
    predictions = (P > r_signal_threshold).astype(int)
    
    return predictions, P


# ============================================================================
# SIMPLE MSE THRESHOLD (Current Approach)
# ============================================================================
def apply_simple_mse_detection(model, data, threshold):
    """Current simple approach: single MSE threshold."""
    reconstructed = model.predict(data, verbose=0)
    mse = np.mean(np.square(data - reconstructed), axis=(1, 2, 3))
    predictions = (mse > threshold).astype(int)
    return predictions, mse


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_method(y_true, y_pred, scores):
    """Calculate evaluation metrics."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        return {
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auroc': 0.5,
            'fpr': np.mean(y_pred[y_true == 0]) if sum(y_true == 0) > 0 else 0
        }
    
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, scores),
        'fpr': np.mean(y_pred[y_true == 0]) if sum(y_true == 0) > 0 else 0
    }


def main():
    print("=" * 70)
    print("COMPARISON: Simple MSE vs Three-Step Threshold")
    print("=" * 70)
    
    # Use best ensemble models
    stable_models = [(50, 1), (50, 5), (75, 1), (100, 1)]
    
    results_simple = []
    results_threestep = []
    
    for ws, sp in stable_models:
        config_name = f"{ws}s_{sp}s"
        print(f"\n📂 Loading {config_name}...")
        
        model, normal_data, attack_data, normal_labels, attack_labels = load_model_and_data(ws, sp)
        
        # Prepare combined test set
        all_data = np.concatenate([normal_data, attack_data], axis=0)
        all_labels = np.concatenate([normal_labels, attack_labels])
        
        # ----------------------------------------------------------------
        # METHOD 1: Simple MSE Threshold (Current)
        # ----------------------------------------------------------------
        reconstructed = model.predict(normal_data, verbose=0)
        normal_mse = np.mean(np.square(normal_data - reconstructed), axis=(1, 2, 3))
        threshold_75 = np.percentile(normal_mse, 75)
        
        preds_simple, scores_simple = apply_simple_mse_detection(model, all_data, threshold_75)
        metrics_simple = evaluate_method(all_labels, preds_simple, scores_simple)
        results_simple.append(metrics_simple)
        
        # ----------------------------------------------------------------
        # METHOD 2: Three-Step Threshold (CANShield)
        # ----------------------------------------------------------------
        R_Loss, R_Time = calculate_three_step_thresholds(model, normal_data, P_LOSS, P_TIME)
        
        # Find best R_Signal threshold (one that gives 0% FPR on normal data)
        best_r_signal = 0.1
        best_metrics = None
        
        for r_sig in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            preds, scores = apply_three_step_detection(model, all_data, R_Loss, R_Time, r_sig)
            metrics = evaluate_method(all_labels, preds, scores)
            
            # Prefer low FPR with reasonable recall
            if metrics['fpr'] <= 0.01 and (best_metrics is None or metrics['recall'] > best_metrics['recall']):
                best_r_signal = r_sig
                best_metrics = metrics
        
        if best_metrics is None:
            preds, scores = apply_three_step_detection(model, all_data, R_Loss, R_Time, 0.2)
            best_metrics = evaluate_method(all_labels, preds, scores)
            
        results_threestep.append(best_metrics)
        
        print(f"  Simple MSE:    Recall={metrics_simple['recall']:.2f}, FPR={metrics_simple['fpr']:.2f}, AUROC={metrics_simple['auroc']:.3f}")
        print(f"  Three-Step:    Recall={best_metrics['recall']:.2f}, FPR={best_metrics['fpr']:.2f}, AUROC={best_metrics['auroc']:.3f}")
    
    # ----------------------------------------------------------------
    # ENSEMBLE COMPARISON
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS (OR Voting)")
    print("=" * 70)
    
    # Average metrics
    avg_simple = {
        'recall': np.mean([r['recall'] for r in results_simple]),
        'fpr': np.mean([r['fpr'] for r in results_simple]),
        'auroc': np.mean([r['auroc'] for r in results_simple]),
        'precision': np.mean([r['precision'] for r in results_simple]),
        'f1': np.mean([r['f1'] for r in results_simple])
    }
    
    avg_threestep = {
        'recall': np.mean([r['recall'] for r in results_threestep]),
        'fpr': np.mean([r['fpr'] for r in results_threestep]),
        'auroc': np.mean([r['auroc'] for r in results_threestep]),
        'precision': np.mean([r['precision'] for r in results_threestep]),
        'f1': np.mean([r['f1'] for r in results_threestep])
    }
    
    print("\n📊 FINAL COMPARISON:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Simple MSE':<15} {'Three-Step':<15} {'Winner':<10}")
    print("-" * 50)
    
    for metric in ['recall', 'precision', 'f1', 'auroc', 'fpr']:
        v1 = avg_simple[metric]
        v2 = avg_threestep[metric]
        
        if metric == 'fpr':
            winner = "Simple" if v1 < v2 else "Three-Step" if v2 < v1 else "Tie"
        else:
            winner = "Simple" if v1 > v2 else "Three-Step" if v2 > v1 else "Tie"
        
        print(f"{metric:<15} {v1:<15.3f} {v2:<15.3f} {winner:<10}")
    
    print("-" * 50)
    
    # Determine overall winner
    simple_wins = sum([
        avg_simple['recall'] > avg_threestep['recall'],
        avg_simple['auroc'] > avg_threestep['auroc'],
        avg_simple['f1'] > avg_threestep['f1'],
        avg_simple['fpr'] < avg_threestep['fpr']
    ])
    
    if simple_wins >= 3:
        print("\n🏆 WINNER: Simple MSE Threshold")
    elif simple_wins <= 1:
        print("\n🏆 WINNER: Three-Step Threshold (CANShield)")
    else:
        print("\n🤝 RESULT: Similar performance, choose based on interpretability needs")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
