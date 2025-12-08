"""
Phase 2: CNN Autoencoder Model Architecture
============================================
Based on CANShield paper (Section IV.C & V.B.1)

Architecture:
- Encoder: 3 blocks of Conv2D → LeakyReLU → MaxPooling2D
- Decoder: 3 blocks of Conv2D → LeakyReLU → UpSampling2D
- Output: Conv2D (Sigmoid) → Cropping2D

Training Parameters (from paper):
- Optimizer: Adam (lr=0.0002, beta1=0.5, beta2=0.99)
- Loss: Mean Squared Error (MSE)
- Batch Size: 128
- Epochs: 100
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    LeakyReLU, 
    MaxPooling2D, 
    UpSampling2D, 
    ZeroPadding2D, 
    Cropping2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


def get_new_autoencoder(time_step: int, num_signals: int) -> Sequential:
    """
    Create a new CNN Autoencoder model.
    
    Args:
        time_step: Number of time steps (window size / sampling period)
                   e.g., 50s window with 1s sampling = 50 time steps
        num_signals: Number of signals (features) in the data
                     For our N2K data = 19 signals
    
    Returns:
        Compiled Keras Sequential model
    
    Architecture (from paper Section V.B.1):
        "We used a five-layer network, where the convolutional layers have 
        3×3 filters, and the numbers of filters in each layer are 32, 16, 16, 
        32, and 1. We utilized leakyRelu as the activation function with a 
        parameter of 0.2, except for the output layer, which has a sigmoid 
        activation function."
    """
    
    in_shape = (time_step, num_signals, 1)
    
    autoencoder = Sequential()
    
    # ==================== ENCODER ====================
    # Zero padding to handle edge cases during convolution
    autoencoder.add(ZeroPadding2D((2, 2), input_shape=in_shape))
    
    # Encoder Block 1: 32 filters, 5x5 kernel
    autoencoder.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    
    # Encoder Block 2: 16 filters, 5x5 kernel
    autoencoder.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    
    # Encoder Block 3: 16 filters, 3x3 kernel
    autoencoder.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    
#bottleneck here !!


    # ==================== DECODER ====================
    # Decoder Block 1: 16 filters, 3x3 kernel
    autoencoder.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))
    
    # Decoder Block 2: 16 filters, 5x5 kernel
    autoencoder.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))
    
    # Decoder Block 3: 32 filters, 5x5 kernel
    autoencoder.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))
    
    # ==================== OUTPUT ====================
    # Calculate cropping needed to match input shape
    temp_shape = autoencoder.output_shape
    diff_h = temp_shape[1] - in_shape[0]
    top = int(diff_h / 2)
    bottom = diff_h - top
    diff_w = temp_shape[2] - in_shape[1]
    left = int(diff_w / 2)
    right = diff_w - left
    
    # Final convolution with sigmoid activation
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    
    # Crop to match original input dimensions
    autoencoder.add(Cropping2D(cropping=((top, bottom), (left, right))))
    
    return autoencoder


def compile_autoencoder(autoencoder: Sequential) -> Sequential:
    """
    Compile the autoencoder with optimizer and loss function.
    
    From paper Section V.B.1:
        "We use the Adam optimizer with a learning rate of 0.0002"
    """
    opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.99)
    autoencoder.compile(
        loss=MeanSquaredError(),
        optimizer=opt,
        metrics=['accuracy']
    )
    return autoencoder


def create_autoencoder(time_step: int, num_signals: int) -> Sequential:
    """
    Create and compile a new autoencoder model.
    
    Args:
        time_step: Number of time steps in input
        num_signals: Number of signals (features)
    
    Returns:
        Compiled autoencoder ready for training
    """
    autoencoder = get_new_autoencoder(time_step, num_signals)
    autoencoder = compile_autoencoder(autoencoder)
    return autoencoder


# ==================== VERIFICATION ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: CNN Autoencoder Model Verification")
    print("=" * 60)
    
    # Test with different configurations
    configs = [
        # (time_step, num_signals, description)
        (50, 19, "50s window, 1s sampling"),
        (10, 19, "50s window, 5s sampling"),
        (5, 19, "50s window, 10s sampling"),
    ]
    
    for time_step, num_signals, desc in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {desc}")
        print(f"Input shape: ({time_step}, {num_signals}, 1)")
        print("=" * 60)
        
        model = create_autoencoder(time_step, num_signals)
        
        # Verify input/output shapes match
        input_shape = model.input_shape[1:]
        output_shape = model.output_shape[1:]
        
        print(f"Input shape:  {input_shape}")
        print(f"Output shape: {output_shape}")
        
        if input_shape == output_shape:
            print("✅ Input and Output shapes MATCH!")
        else:
            print("❌ ERROR: Shapes do not match!")
        
        # Print model summary for first config only
        if time_step == 50:
            print("\nModel Summary:")
            model.summary()
    
    print("\n" + "=" * 60)
    print("✅ Model architecture verification complete!")
    print("=" * 60)
