"""
Visualize Autoencoder Reconstruction Quality
============================================
This script shows how well the trained autoencoder reconstructs input data.
It displays side-by-side comparison: Original vs Reconstructed
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import sys

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'Phase1' / 'scripts'))
from config import TIME_STEPS, SAMPLING_PERIODS, NUM_FEATURES

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'Phase1' / 'data'
MODEL_DIR = BASE_DIR / 'Phase2' / 'models'
OUTPUT_DIR = BASE_DIR / 'Phase2' / 'visualizations'


def load_model_and_data(window_size: int, sampling_period: int):
    """Load trained model and test data."""
    
    # Load model
    model_name = f"{window_size}s_{sampling_period}s.h5"
    model_path = MODEL_DIR / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_name}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load test data
    data_path = DATA_DIR / f"{window_size}s_window" / f"sampling_{sampling_period}s" / "test.npy"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    print(f"Loading data: {data_path.name}")
    data = np.load(data_path, allow_pickle=True)
    
    # Remove timestamp column (column 0)
    features = data[:, :, 1:].astype(np.float32)
    
    return model, features


def reconstruct_sample(model, sample):
    """Pass a single sample through the autoencoder."""
    
    # Add batch and channel dimensions: (time, features) -> (1, time, features, 1)
    input_data = sample.reshape(1, sample.shape[0], sample.shape[1], 1)
    
    # Get reconstruction
    reconstructed = model.predict(input_data, verbose=0)
    
    # Remove batch and channel dimensions
    reconstructed = reconstructed.squeeze()
    
    return reconstructed


def visualize_single_sample(original, reconstructed, sample_idx, window_size, sampling_period):
    """Create visualization comparing original vs reconstructed."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Autoencoder Reconstruction - Model: {window_size}s_{sampling_period}s (Sample #{sample_idx})', 
                 fontsize=14, fontweight='bold')
    
    # Calculate reconstruction error
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # 1. Heatmap - Original
    ax1 = axes[0, 0]
    im1 = ax1.imshow(original.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Original Input', fontsize=12)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Features (60)')
    plt.colorbar(im1, ax=ax1, label='Normalized Value')
    
    # 2. Heatmap - Reconstructed
    ax2 = axes[0, 1]
    im2 = ax2.imshow(reconstructed.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('Reconstructed Output', fontsize=12)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Features (60)')
    plt.colorbar(im2, ax=ax2, label='Normalized Value')
    
    # 3. Difference Heatmap
    ax3 = axes[1, 0]
    diff = np.abs(original - reconstructed)
    im3 = ax3.imshow(diff.T, aspect='auto', cmap='Reds', vmin=0, vmax=0.3)
    ax3.set_title(f'Absolute Difference (MSE: {mse:.4f}, MAE: {mae:.4f})', fontsize=12)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Features (60)')
    plt.colorbar(im3, ax=ax3, label='|Original - Reconstructed|')
    
    # 4. Line plot - First 5 features over time
    ax4 = axes[1, 1]
    time_steps = np.arange(original.shape[0])
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    feature_names = ['Feat 1', 'Feat 2', 'Feat 3', 'Feat 4', 'Feat 5']
    
    for i in range(5):
        ax4.plot(time_steps, original[:, i], color=colors[i], linestyle='-', 
                 label=f'{feature_names[i]} (Orig)', alpha=0.7)
        ax4.plot(time_steps, reconstructed[:, i], color=colors[i], linestyle='--', 
                 label=f'{feature_names[i]} (Recon)', alpha=0.7)
    
    ax4.set_title('Feature Comparison: Original (solid) vs Reconstructed (dashed)', fontsize=12)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Normalized Value')
    ax4.legend(loc='upper right', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    return fig, mse, mae


def visualize_multiple_features(original, reconstructed, sample_idx, window_size, sampling_period):
    """Detailed comparison of all 60 features for a single time step."""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'Feature-by-Feature Comparison - Model: {window_size}s_{sampling_period}s (Sample #{sample_idx})', 
                 fontsize=14, fontweight='bold')
    
    # Take middle time step
    mid_t = original.shape[0] // 2
    orig_slice = original[mid_t, :]
    recon_slice = reconstructed[mid_t, :]
    
    features = np.arange(60)
    
    # Bar chart comparison
    ax1 = axes[0]
    width = 0.35
    ax1.bar(features - width/2, orig_slice, width, label='Original', alpha=0.7, color='blue')
    ax1.bar(features + width/2, recon_slice, width, label='Reconstructed', alpha=0.7, color='orange')
    ax1.set_title(f'All 60 Features at Time Step {mid_t}', fontsize=12)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.set_xlim(-1, 60)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Error per feature
    ax2 = axes[1]
    errors = np.abs(orig_slice - recon_slice)
    ax2.bar(features, errors, color='red', alpha=0.7)
    ax2.set_title(f'Reconstruction Error per Feature (Mean: {np.mean(errors):.4f})', fontsize=12)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Absolute Error')
    ax2.set_xlim(-1, 60)
    ax2.axhline(y=np.mean(errors), color='black', linestyle='--', label=f'Mean Error: {np.mean(errors):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


def main():
    """Main function to run visualization."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configuration - you can change these
    window_size = 50  # Options: 50, 75, 100
    sampling_period = 1  # Options: 1, 5, 10
    sample_index = 0  # Which test sample to visualize
    
    print("=" * 60)
    print("AUTOENCODER RECONSTRUCTION VISUALIZATION")
    print("=" * 60)
    
    # Load model and data
    model, features = load_model_and_data(window_size, sampling_period)
    
    print(f"\nData shape: {features.shape}")
    print(f"Number of test samples available: {features.shape[0]}")
    
    # Get sample
    if sample_index >= features.shape[0]:
        sample_index = 0
        print(f"Adjusted sample_index to {sample_index}")
    
    original = features[sample_index]
    print(f"\nProcessing sample #{sample_index}")
    print(f"Sample shape: {original.shape}")
    
    # Reconstruct
    reconstructed = reconstruct_sample(model, original)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Calculate metrics
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    print(f"\n📊 RECONSTRUCTION METRICS:")
    print(f"   MSE (Mean Squared Error): {mse:.6f}")
    print(f"   MAE (Mean Absolute Error): {mae:.6f}")
    print(f"   Reconstruction Quality: {(1-mse)*100:.2f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Main visualization
    fig1, _, _ = visualize_single_sample(original, reconstructed, sample_index, 
                                          window_size, sampling_period)
    save_path1 = OUTPUT_DIR / f'reconstruction_{window_size}s_{sampling_period}s_sample{sample_index}.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path1}")
    
    # Feature detail visualization
    fig2 = visualize_multiple_features(original, reconstructed, sample_index,
                                        window_size, sampling_period)
    save_path2 = OUTPUT_DIR / f'features_{window_size}s_{sampling_period}s_sample{sample_index}.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path2}")
    
    # Show plots
    plt.show()
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
