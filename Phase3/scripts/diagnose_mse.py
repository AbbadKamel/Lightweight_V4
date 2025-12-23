"""
Diagnostic Script: Check MSE distributions for Normal vs Attack data
"""
import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Focus on 50s_1s as our primary model
MODEL_NAME = '50s_1s'

def main():
    print("="*60)
    print("DIAGNOSTIC: MSE Distribution Analysis")
    print("="*60)
    
    # Load Model
    model_path = os.path.join(config.PHASE2_DIR, "models", f"{MODEL_NAME}.h5")
    print(f"Loading model: {model_path}")
    model = load_model(model_path, compile=False)
    
    # Load Threshold
    thresh_path = os.path.join(config.THRESHOLDS_DIR, f"{MODEL_NAME}_thresholds.json")
    with open(thresh_path) as f:
        thresh_data = json.load(f)
    
    print(f"\nThresholds from file:")
    for k, v in thresh_data.items():
        print(f"  {k}: {v:.6f}")
    
    # Load Normal Test Data
    parts = MODEL_NAME.split('_')
    ts, sp = 50, 1  # 50s_1s
    
    normal_path = os.path.join(config.TEST_DATA_DIR, f"{ts}s_window", f"sampling_{sp}s", "test.npy")
    print(f"\nLoading Normal Data: {normal_path}")
    normal_raw = np.load(normal_path, allow_pickle=True)
    normal_data = normal_raw[:, :, 1:].astype(np.float32)
    print(f"  Shape: {normal_data.shape}")
    
    # Reshape for CNN
    normal_cnn = normal_data.reshape(normal_data.shape[0], normal_data.shape[1], normal_data.shape[2], 1)
    
    # Predict and calculate MSE for Normal
    recon_normal = model.predict(normal_cnn, verbose=0)
    mse_normal = np.mean(np.square(normal_cnn - recon_normal), axis=(1, 2, 3))
    
    print(f"\nNORMAL Data MSE Statistics:")
    print(f"  Min:    {np.min(mse_normal):.6f}")
    print(f"  Max:    {np.max(mse_normal):.6f}")
    print(f"  Mean:   {np.mean(mse_normal):.6f}")
    print(f"  Median: {np.median(mse_normal):.6f}")
    print(f"  Std:    {np.std(mse_normal):.6f}")
    
    # Load Attack Data
    attack_path = os.path.join(config.ATTACKS_DIR, f"attacks_{MODEL_NAME}.npy")
    print(f"\nLoading Attack Data: {attack_path}")
    attack_data = np.load(attack_path)
    print(f"  Shape: {attack_data.shape}")
    
    # Predict and calculate MSE for Attacks
    recon_attack = model.predict(attack_data, verbose=0)
    mse_attack = np.mean(np.square(attack_data - recon_attack), axis=(1, 2, 3))
    
    print(f"\nATTACK Data MSE Statistics:")
    print(f"  Min:    {np.min(mse_attack):.6f}")
    print(f"  Max:    {np.max(mse_attack):.6f}")
    print(f"  Mean:   {np.mean(mse_attack):.6f}")
    print(f"  Median: {np.median(mse_attack):.6f}")
    print(f"  Std:    {np.std(mse_attack):.6f}")
    
    # Compare with thresholds
    threshold_max = thresh_data['threshold']
    threshold_95 = thresh_data.get('95', threshold_max)
    
    n_detected_max = np.sum(mse_attack > threshold_max)
    n_detected_95 = np.sum(mse_attack > threshold_95)
    
    print(f"\nDETECTION ANALYSIS:")
    print(f"  Threshold (Max): {threshold_max:.6f}")
    print(f"  Attacks with MSE > Threshold (Max): {n_detected_max}/{len(mse_attack)} ({100*n_detected_max/len(mse_attack):.1f}%)")
    print(f"  Threshold (95%): {threshold_95:.6f}")
    print(f"  Attacks with MSE > Threshold (95%): {n_detected_95}/{len(mse_attack)} ({100*n_detected_95/len(mse_attack):.1f}%)")
    
    # Key insight
    print(f"\n" + "="*60)
    if np.max(mse_attack) < threshold_max:
        print("⚠️ PROBLEM: Attack MSE values are BELOW the threshold!")
        print("   This means attacks are being reconstructed TOO WELL.")
        print("   Possible causes:")
        print("   1. Attacks are too subtle (low intensity)")
        print("   2. Model is too good at generalizing")
        print("   3. Attack data is not properly modified")
    else:
        print("✅ Some attacks have MSE above threshold - detection should work!")
    print("="*60)

if __name__ == "__main__":
    main()
