"""
CNN Autoencoder Windowing Configuration for NMEA 2000 IDS
Based on CANShield methodology adapted for maritime environment

Reference: CANShield paper - Multi-scale temporal window approach
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_NAME = "nmea2000_maritime"
DATA_DIR = "results/master_table_final.csv"
SCALER_DIR = "results/scaler_params.json"
SIGNAL_ORDER_FILE = "results/optimal_signal_order.json"

# Signal configuration
NUM_SIGNALS = 15  # wind_speed, wind_angle, yaw, cog, heading, roll, rudder_angle_order, 
                  # rudder_position, rate_of_turn, depth, variation, latitude, longitude, 
                  # pitch, sog
NUM_FEATURES = 60  # 15 signals × 4 aggregations (mean, max, min, std)

# ============================================================================
# MULTI-SCALE TEMPORAL WINDOWS (CANShield approach)
# ============================================================================
# CANShield uses multiple window sizes to capture different temporal patterns:
# - Small windows: Detect rapid attacks (spike injection, sudden changes)
# - Large windows: Detect slow attacks (gradual drift, replay attacks)

# Window sizes (in timesteps/seconds)
# CANShield original: [25, 50, 75, 100] timesteps
# Our adaptation for NMEA 2000 (slower update rates):
TIME_STEPS = [50, 75, 100]  # Window sizes in seconds
WINDOW_SIZES = TIME_STEPS   # Alias for consistency with train_cascade.py
# 50s = Detect rapid maneuvers/attacks
# 75s = Medium-term patterns (course changes)
# 100s = Long-term patterns (navigation drift)

# For quick testing during development:
TIME_STEPS_QUICK = [50]

# Default window size for single-model training
DEFAULT_TIME_STEP = 50
DEFAULT_WINDOW_SIZE = DEFAULT_TIME_STEP  # Alias

# ============================================================================
# MULTI-SCALE SAMPLING PERIODS (Temporal resolution)
# ============================================================================
# CANShield samples data at different rates to create multiple views:
# - 1s: Original resolution (capture all detail)
# - 5s: Reduced resolution (smoother, less noise)
# - 10s: Coarse resolution (long-term trends only)

# Sampling periods (downsampling factor)
# CANShield original: [1, 5, 10, 20, 50]
# Our adaptation:
SAMPLING_PERIODS = [1, 5, 10]
# 1s = No downsampling, use all data (5,095 samples)
# 5s = Downsample by 5 (1,019 samples)
# 10s = Downsample by 10 (509 samples)

# For quick testing:
SAMPLING_PERIODS_QUICK = [1]

# Default sampling period
DEFAULT_SAMPLING_PERIOD = 1

# ============================================================================
# WINDOW SLIDING STRATEGY (Overlap control)
# ============================================================================
# Window step controls how much windows overlap:
# - step=1: Windows overlap by (window_size-1), maximum training data
# - step=10: Windows overlap by (window_size-10), 10x faster training
# - step=window_size: No overlap, fastest but loses temporal continuity

# Training phase: Use overlapping windows for more training samples
WINDOW_STEP_TRAIN = 10  # Generate window every 10 seconds
                        # For 50s window: 80% overlap (40s shared between windows)
                        # Total training windows ≈ 5,095 / 10 = 509 windows

# Validation phase: Less overlap needed
WINDOW_STEP_VALID = 10  # Same as training for consistency

# Testing phase: Can use full overlap for precise detection
WINDOW_STEP_TEST = 10  # Use same for now, can reduce to 1 for real-time testing

# ============================================================================
# TRAIN/VALIDATION/TEST SPLIT
# ============================================================================
# CANShield uses temporal split (not random) to preserve time series structure

# Split ratios
TRAIN_RATIO = 0.70  # 70% for training (first 3,566 seconds ≈ 59 minutes)
VALID_RATIO = 0.15  # 15% for validation (534 seconds ≈ 9 minutes)
TEST_RATIO = 0.15   # 15% for testing (995 seconds ≈ 17 minutes)

# Or use absolute time splits
TRAIN_SAMPLES = int(5095 * 0.70)  # 3,566 samples
VALID_SAMPLES = int(5095 * 0.15)  # 764 samples  
TEST_SAMPLES = int(5095 * 0.15)   # 764 samples

# ============================================================================
# CNN AUTOENCODER ARCHITECTURE
# ============================================================================
# CANShield uses separate autoencoder for each (time_step, sampling_period) pair

# Number of models to train
# Full approach: len(TIME_STEPS) × len(SAMPLING_PERIODS) = 3 × 3 = 9 models
# Quick approach: 1 model (50s window, 1s sampling)
NUM_MODELS_FULL = len(TIME_STEPS) * len(SAMPLING_PERIODS)  # 9 models
NUM_MODELS_QUICK = 1

# Input shape for each model
# For time_step=50, sampling_period=1:
#   Input: (50 timesteps, 60 features) = 50×60 matrix
# For time_step=50, sampling_period=5:
#   Input: (50 timesteps, 60 features) but data sampled every 5s

# ============================================================================
# TRAINING CONFIGURATION (CANShield paper Section V.B.1)
# ============================================================================
# These values match the CANShield paper exactly

# Optimizer: Adam with specific hyperparameters
LEARNING_RATE = 0.0002   # CANShield: "learning rate of 0.0002"
ADAM_BETA_1 = 0.5        # CANShield paper
ADAM_BETA_2 = 0.99       # CANShield paper

# Training parameters
BATCH_SIZE = 128         # CANShield paper
MAX_EPOCHS = 100         # With early stopping (CANShield uses 500 without)
EARLY_STOPPING_PATIENCE = 10  # Stop if val_loss doesn't improve for 10 epochs

# Learning rate reduction on plateau
REDUCE_LR_PATIENCE = 5   # Reduce LR if no improvement for 5 epochs
REDUCE_LR_FACTOR = 0.5   # Multiply LR by 0.5
REDUCE_LR_MIN = 1e-6     # Minimum learning rate

# Dynamic batch size for small datasets
MIN_BATCH_SIZE = 8       # Fallback for configs with very few samples

# Save best model based on validation loss
SAVE_BEST_MODEL = True

def get_effective_batch_size(num_samples: int) -> int:
    """
    Get the effective batch size based on available samples.
    
    For configurations with very few samples (e.g., 100s/10s = 32 samples),
    we reduce batch size to ensure proper training.
    
    Args:
        num_samples: Number of training samples
    
    Returns:
        Effective batch size to use
    """
    if num_samples >= BATCH_SIZE:
        return BATCH_SIZE
    elif num_samples >= BATCH_SIZE // 2:
        return BATCH_SIZE // 2  # 64
    elif num_samples >= BATCH_SIZE // 4:
        return BATCH_SIZE // 4  # 32
    else:
        return max(MIN_BATCH_SIZE, num_samples // 2)

# ============================================================================
# DETECTION THRESHOLDS (CANShield approach)
# ============================================================================
# CANShield uses percentile-based thresholds on reconstruction error

# Loss factors (percentile of reconstruction error on normal data)
# Higher = more sensitive (more false positives)
# Lower = less sensitive (more false negatives)
LOSS_FACTORS = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.99]
DEFAULT_LOSS_FACTOR = 95  # 95th percentile

# Time factors (percentage of window that must be anomalous)
# How many timesteps in window must exceed threshold
TIME_FACTORS = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.99]
DEFAULT_TIME_FACTOR = 99  # 99% of timesteps must be anomalous

# Signal factors (percentage of signals that must be anomalous)
# How many of 15 signals must show anomaly
SIGNAL_FACTORS = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.99]
DEFAULT_SIGNAL_FACTOR = 95  # 95% of signals must be anomalous

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
MODEL_DIR = "../../Phase2/models/"
RESULTS_DIR = "../../Phase2/results/"
LOGS_DIR = "../../Phase2/logs/"

# Model naming convention
# Format: model_{time_step}s_{sampling_period}s.h5
# Example: model_50s_1s.h5 (50 second window, 1 second sampling)

# ============================================================================
# COMPUTATIONAL EFFICIENCY
# ============================================================================
# Percentage of samples to use (for quick experiments)
PER_OF_SAMPLES = 1.00  # 1.00 = 100% of data
                       # 0.10 = 10% for quick testing

# GPU configuration
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory

# Parallel training
TRAIN_MODELS_PARALLEL = False  # Set True if training multiple models simultaneously

# ============================================================================
# EXAMPLE WINDOW GENERATION
# ============================================================================
"""
For time_step=50, sampling_period=1, window_step=10:

Original data: 5,095 samples (seconds)

Window generation:
- Window 0: samples [0:50]     (0-49 seconds)
- Window 1: samples [10:60]    (10-59 seconds)  ← 40s overlap with window 0
- Window 2: samples [20:70]    (20-69 seconds)  ← 40s overlap with window 1
- ...
- Window N: samples [5040:5090] (last complete 50s window)

Total windows: (5095 - 50) / 10 + 1 ≈ 505 windows

Each window shape: (50 timesteps, 60 features)

For time_step=50, sampling_period=5, window_step=10:
- First, downsample data: 5,095 → 1,019 samples (take every 5th sample)
- Then create windows from downsampled data
- Window 0: downsampled_samples[0:50]
- Window 1: downsampled_samples[10:60]
- ...
Total windows: (1019 - 50) / 10 + 1 ≈ 97 windows

Each window still shape: (50 timesteps, 60 features)
But each timestep represents 5 seconds of real time
"""

# ============================================================================
# SUMMARY
# ============================================================================
def print_config_summary():
    print("="*80)
    print("NMEA 2000 CNN AUTOENCODER CONFIGURATION")
    print("="*80)
    print(f"\n📊 DATASET:")
    print(f"   Signals: {NUM_SIGNALS}")
    print(f"   Features: {NUM_FEATURES} ({NUM_SIGNALS} signals × 4 aggregations)")
    print(f"   Total samples: 5,095 seconds (84.9 minutes)")
    
    print(f"\n🪟 WINDOWING STRATEGY:")
    print(f"   Window sizes: {TIME_STEPS} seconds")
    print(f"   Sampling periods: {SAMPLING_PERIODS} seconds")
    print(f"   Window step (training): {WINDOW_STEP_TRAIN} seconds")
    print(f"   Overlap: {((DEFAULT_TIME_STEP - WINDOW_STEP_TRAIN) / DEFAULT_TIME_STEP * 100):.0f}%")
    
    print(f"\n🤖 MODELS:")
    print(f"   Total models to train: {NUM_MODELS_FULL}")
    for ts in TIME_STEPS:
        for sp in SAMPLING_PERIODS:
            print(f"   - Model {ts}s window × {sp}s sampling")
    
    print(f"\n📈 TRAINING:")
    print(f"   Train/Valid/Test split: {TRAIN_RATIO}/{VALID_RATIO}/{TEST_RATIO}")
    print(f"   Train samples: {TRAIN_SAMPLES} ({TRAIN_SAMPLES/60:.1f} min)")
    print(f"   Valid samples: {VALID_SAMPLES} ({VALID_SAMPLES/60:.1f} min)")
    print(f"   Test samples: {TEST_SAMPLES} ({TEST_SAMPLES/60:.1f} min)")
    print(f"   Max epochs: {MAX_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    print(f"\n🎯 DETECTION:")
    print(f"   Loss factor: {DEFAULT_LOSS_FACTOR}th percentile")
    print(f"   Time factor: {DEFAULT_TIME_FACTOR}% of timesteps")
    print(f"   Signal factor: {DEFAULT_SIGNAL_FACTOR}% of signals")
    
    # Calculate expected windows
    for ts in TIME_STEPS:
        for sp in SAMPLING_PERIODS:
            downsampled_len = 5095 // sp
            num_windows = (downsampled_len - ts) // WINDOW_STEP_TRAIN + 1
            print(f"\n   Model {ts}s×{sp}s: ≈{num_windows} training windows")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_config_summary()
