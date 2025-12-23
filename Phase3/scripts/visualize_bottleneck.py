"""
Visualize CNN Autoencoder Bottleneck
=====================================
Shows what the compressed representation looks like inside the model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Setup paths
PHASE1_DIR = "/home/abbad241/Desktop/PhD/Journals_Articles_Papers/Next paper/Lightweight_IA_V_3/Phase1"
PHASE2_DIR = "/home/abbad241/Desktop/PhD/Journals_Articles_Papers/Next paper/Lightweight_IA_V_3/Phase2"
PHASE3_DIR = "/home/abbad241/Desktop/PhD/Journals_Articles_Papers/Next paper/Lightweight_IA_V_3/Phase3"

def visualize_bottleneck():
    """Visualize the bottleneck representation of the autoencoder."""
    
    print("=" * 70)
    print("VISUALIZING CNN AUTOENCODER BOTTLENECK")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(PHASE2_DIR, "models", "50s_1s.h5")
    print(f"\n📂 Loading model: {model_path}")
    full_model = load_model(model_path, compile=False)
    
    # Print model architecture
    print("\n📋 MODEL ARCHITECTURE:")
    print("-" * 70)
    full_model.summary()
    print("-" * 70)
    
    # Load test data first
    data_path = os.path.join(PHASE1_DIR, "data", "50s_window", "sampling_1s", "test.npy")
    print(f"\n📂 Loading test data: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    # Remove timestamp column and reshape
    data = data[:, :, 1:].astype(np.float32)  # Remove timestamp
    data = data.reshape(data.shape[0], data.shape[1], -1, 1)  # Add channel
    print(f"   Data shape: {data.shape}")
    
    # Take a few samples
    samples = data[:3]  # First 3 windows
    
    # Get layer outputs using a custom function
    # Find the bottleneck layer index (last max_pooling before decoder)
    bottleneck_idx = None
    for i, layer in enumerate(full_model.layers):
        if 'max_pooling2d' in layer.name:
            bottleneck_idx = i
    
    print(f"\n🎯 Bottleneck layer: {full_model.layers[bottleneck_idx].name} (index {bottleneck_idx})")
    
    # Create a function to get intermediate outputs
    @tf.function
    def get_bottleneck_output(model, x, layer_idx):
        """Get output from a specific layer."""
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == layer_idx:
                return x
        return x
    
    # Compute bottleneck outputs by running through layers manually
    print("\n🔄 Computing bottleneck representations...")
    
    x = samples
    for i, layer in enumerate(full_model.layers):
        x = layer(x)
        if i == bottleneck_idx:
            bottleneck_output = x.numpy() if hasattr(x, 'numpy') else x
            break
    
    print(f"   Bottleneck output shape: {bottleneck_output.shape}")
    
    # Get full reconstruction
    reconstruction = full_model.predict(samples, verbose=0)
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("CNN Autoencoder: Input → Bottleneck → Reconstruction", 
                 fontsize=16, fontweight='bold')
    
    for sample_idx in range(3):
        # ---- Input ----
        ax1 = fig.add_subplot(3, 4, sample_idx*4 + 1)
        input_img = samples[sample_idx, :, :, 0]
        im1 = ax1.imshow(input_img, aspect='auto', cmap='viridis')
        ax1.set_title(f"Sample {sample_idx+1}: INPUT\n{input_img.shape[0]}×{input_img.shape[1]} = {input_img.size} values", fontsize=10)
        ax1.set_xlabel("Features (signals)")
        ax1.set_ylabel("Time (seconds)")
        plt.colorbar(im1, ax=ax1)
        
        # ---- Bottleneck (average of all channels) ----
        ax2 = fig.add_subplot(3, 4, sample_idx*4 + 2)
        bn = bottleneck_output[sample_idx]  # (H, W, C)
        n_channels = bn.shape[-1]
        
        # Average all channels
        bn_avg = np.mean(bn, axis=-1)
        im2 = ax2.imshow(bn_avg, aspect='auto', cmap='plasma')
        ax2.set_title(f"BOTTLENECK (avg of {n_channels} channels)\n{bn.shape[0]}×{bn.shape[1]}×{bn.shape[2]} = {bn.size} values", fontsize=10)
        ax2.set_xlabel("Compressed Width")
        ax2.set_ylabel("Compressed Time")
        plt.colorbar(im2, ax=ax2)
        
        # ---- All 16 bottleneck channels in a grid ----
        ax3 = fig.add_subplot(3, 4, sample_idx*4 + 3)
        
        # Arrange channels in a 4x4 grid
        channels_grid = np.zeros((bn.shape[0]*4, bn.shape[1]*4))
        for ch in range(min(16, n_channels)):
            row = ch // 4
            col = ch % 4
            channels_grid[row*bn.shape[0]:(row+1)*bn.shape[0], 
                          col*bn.shape[1]:(col+1)*bn.shape[1]] = bn[:, :, ch]
        
        im3 = ax3.imshow(channels_grid, aspect='auto', cmap='plasma')
        ax3.set_title(f"All {n_channels} Bottleneck Channels\n(Each cell is one feature map)", fontsize=10)
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3)
        
        # ---- Reconstruction ----
        ax4 = fig.add_subplot(3, 4, sample_idx*4 + 4)
        recon_img = reconstruction[sample_idx, :, :, 0]
        im4 = ax4.imshow(recon_img, aspect='auto', cmap='viridis')
        ax4.set_title(f"RECONSTRUCTION\n{recon_img.shape[0]}×{recon_img.shape[1]} = {recon_img.size} values", fontsize=10)
        ax4.set_xlabel("Features (signals)")
        ax4.set_ylabel("Time (seconds)")
        plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    
    output_path = os.path.join(PHASE3_DIR, "figures", "bottleneck_visualization.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()
    
    # ========================================================================
    # COMPRESSION STATISTICS
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 COMPRESSION ANALYSIS")
    print("=" * 70)
    
    input_size = samples[0, :, :, 0].size
    bottleneck_size = bottleneck_output[0].size
    
    print(f"\n   INPUT:      {samples[0].shape} = {input_size} values")
    print(f"   BOTTLENECK: {bottleneck_output[0].shape} = {bottleneck_size} values")
    print(f"\n   COMPRESSION RATIO: {input_size/bottleneck_size:.2f}x")
    print(f"   Information reduced to: {(bottleneck_size/input_size)*100:.1f}%")
    
    # MSE between input and reconstruction
    mse = np.mean((samples[0] - reconstruction[0])**2)
    print(f"\n   Reconstruction MSE: {mse:.6f}")
    print(f"   (Lower = better reconstruction quality)")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    visualize_bottleneck()
