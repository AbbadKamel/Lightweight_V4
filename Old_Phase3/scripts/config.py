"""
Phase 3: Attack Detection & Evaluation Configuration
=====================================================
Central configuration for detection thresholds, attack generation,
and evaluation metrics.

Based on CANShield methodology adapted for NMEA2000 maritime environment.
"""

import os
import json

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "../..")

# Input paths
PHASE1_DATA_DIR = os.path.join(PROJECT_ROOT, "Phase1/data")
PHASE1_RESULTS_DIR = os.path.join(PROJECT_ROOT, "Phase1/results")
PHASE2_MODELS_DIR = os.path.join(PROJECT_ROOT, "Phase2/models")

# Output paths
PHASE3_DIR = os.path.join(PROJECT_ROOT, "Phase3")
THRESHOLDS_DIR = os.path.join(PHASE3_DIR, "thresholds")
ATTACKS_DIR = os.path.join(PHASE3_DIR, "attacks")
RESULTS_DIR = os.path.join(PHASE3_DIR, "results")
FIGURES_DIR = os.path.join(PHASE3_DIR, "figures")

# Scaler params for denormalization and attack constraints
SCALER_PARAMS_FILE = os.path.join(PHASE1_RESULTS_DIR, "scaler_params.json")

# ============================================================================
# MODEL CONFIGURATIONS (from Phase 2)
# ============================================================================
WINDOW_SIZES = [50, 75, 100]  # seconds
SAMPLING_PERIODS = [1, 5, 10]  # seconds

# All 9 model configurations
MODEL_CONFIGS = [(ws, sp) for ws in WINDOW_SIZES for sp in SAMPLING_PERIODS]

# Number of features (15 signals × 4 aggregations)
NUM_SIGNALS = 15
NUM_FEATURES = 60

# Signal names (in order)
SIGNAL_NAMES = [
    'wind_speed', 'wind_angle', 'yaw', 'cog', 'heading',
    'roll', 'rudder_angle_order', 'rudder_position', 'rate_of_turn',
    'depth', 'variation', 'latitude', 'longitude', 'pitch', 'sog'
]

# Feature names (60 total)
AGGREGATIONS = ['mean', 'max', 'min', 'std']
FEATURE_NAMES = [f"{sig}_{agg}" for sig in SIGNAL_NAMES for agg in AGGREGATIONS]

# Helper: indices for a signal's aggregations (mean, max, min, std)
def signal_feature_indices(signal_name: str) -> list:
    base = SIGNAL_NAMES.index(signal_name) * len(AGGREGATIONS)
    return list(range(base, base + len(AGGREGATIONS)))

# ============================================================================
# THRESHOLD CONFIGURATION
# ============================================================================
# Percentiles to calculate (we'll evaluate different thresholds)
THRESHOLD_PERCENTILES = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.9]

# Default percentile for detection
DEFAULT_THRESHOLD_PERCENTILE = 95

# Reconstruction error aggregation percentile (top-k style to catch localized anomalies)
ERROR_AGGREGATION_PERCENTILE = 98

# ============================================================================
# ATTACK CONFIGURATION
# ============================================================================
# Attack types to generate
ATTACK_TYPES = [
    'spike',      # Sudden value jump
    'constant',   # Freeze signal at fixed value
    'replay',     # Repeat old data segment
    'drift',      # Gradual deviation over time
    'noise',      # Add random noise
    'scaling'     # Scale values by factor
]

# Signals to attack (critical navigation signals)
ATTACK_TARGET_SIGNALS = [
    'heading',    # Compass heading - critical for navigation
    'cog',        # Course over ground
    'sog',        # Speed over ground
    'depth',      # Water depth - safety critical
    'latitude',   # GPS position
    'longitude',  # GPS position
    'rudder_position',  # Steering
]

# Flattened feature indices for all critical signals (all 4 aggregations each)
CRITICAL_FEATURE_INDICES = sorted({
    idx for sig in ATTACK_TARGET_SIGNALS for idx in signal_feature_indices(sig)
})

# Attack intensities (percentage of signal range)
ATTACK_INTENSITIES = {
    'low': 0.1,      # 10% of signal range
    'medium': 0.3,   # 30% of signal range
    'high': 0.5,     # 50% of signal range
    'extreme': 1.0   # 100% of signal range
}

# Physical constraints for realistic attacks (max change per second)
# These are in NORMALIZED units [0, 1]
PHYSICAL_CONSTRAINTS = {
    'heading': 0.03,      # ~10° per second (normalized)
    'cog': 0.03,
    'sog': 0.05,          # ~0.5 m/s per second acceleration
    'depth': 0.02,        # Depth changes slowly
    'latitude': 0.001,    # ~10m per second
    'longitude': 0.001,
    'roll': 0.05,
    'pitch': 0.05,
    'yaw': 0.03,
    'rudder_position': 0.1,
    'rudder_angle_order': 0.1,
    'rate_of_turn': 0.1,
    'wind_speed': 0.1,
    'wind_angle': 0.1,
    'variation': 0.001,   # Almost constant
}

# Attack duration (percentage of window)
ATTACK_DURATIONS = {
    'short': 0.2,    # 20% of window
    'medium': 0.5,   # 50% of window
    'long': 0.8,     # 80% of window
    'full': 1.0      # 100% of window
}

# Number of attack samples to generate per configuration
NUM_ATTACK_SAMPLES = 50

# ============================================================================
# ENSEMBLE VOTING CONFIGURATION
# ============================================================================
# Voting strategies
VOTING_STRATEGIES = ['majority', 'unanimous', 'any']

# Default voting strategy
DEFAULT_VOTING_STRATEGY = 'majority'

# Minimum votes for detection (for weighted voting)
MIN_VOTES_FOR_DETECTION = 5  # Out of 9 models

# ============================================================================
# EVALUATION METRICS
# ============================================================================
# Metrics to compute
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'specificity',
    'fpr',  # False Positive Rate
    'fnr',  # False Negative Rate
    'auc_roc',
    'auc_pr'
]

# Success criteria
SUCCESS_CRITERIA = {
    'min_recall': 0.90,      # Detect at least 90% of attacks
    'max_fpr': 0.05,         # Less than 5% false alarms
    'min_f1': 0.85,          # F1 score above 0.85
    'min_auc': 0.90          # AUC above 0.90
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_scaler_params():
    """Load scaler parameters for denormalization."""
    with open(SCALER_PARAMS_FILE, 'r') as f:
        return json.load(f)

def get_model_path(window_size: int, sampling_period: int) -> str:
    """Get path to trained model."""
    return os.path.join(PHASE2_MODELS_DIR, f"{window_size}s_{sampling_period}s.h5")

def get_test_data_path(window_size: int, sampling_period: int) -> str:
    """Get path to test data."""
    return os.path.join(
        PHASE1_DATA_DIR,
        f"{window_size}s_window",
        f"sampling_{sampling_period}s",
        "test.npy"
    )

def get_threshold_path(window_size: int, sampling_period: int) -> str:
    """Get path to save/load thresholds."""
    return os.path.join(THRESHOLDS_DIR, f"{window_size}s_{sampling_period}s_thresholds.json")

def get_attack_data_path(attack_type: str, window_size: int, sampling_period: int) -> str:
    """Get path to save/load attack data."""
    return os.path.join(ATTACKS_DIR, f"{attack_type}_{window_size}s_{sampling_period}s.npy")

def ensure_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [THRESHOLDS_DIR, ATTACKS_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================
def print_config():
    """Print Phase 3 configuration summary."""
    print("=" * 80)
    print("PHASE 3: ATTACK DETECTION & EVALUATION CONFIGURATION")
    print("=" * 80)
    
    print(f"\n📁 PATHS:")
    print(f"   Models: {PHASE2_MODELS_DIR}")
    print(f"   Test data: {PHASE1_DATA_DIR}")
    print(f"   Thresholds: {THRESHOLDS_DIR}")
    print(f"   Attacks: {ATTACKS_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    
    print(f"\n🤖 MODELS: {len(MODEL_CONFIGS)}")
    for ws, sp in MODEL_CONFIGS:
        print(f"   - {ws}s_{sp}s")
    
    print(f"\n📊 THRESHOLDS:")
    print(f"   Percentiles: {THRESHOLD_PERCENTILES}")
    print(f"   Default: {DEFAULT_THRESHOLD_PERCENTILE}th percentile")
    
    print(f"\n⚔️ ATTACKS:")
    print(f"   Types: {ATTACK_TYPES}")
    print(f"   Target signals: {ATTACK_TARGET_SIGNALS}")
    print(f"   Samples per config: {NUM_ATTACK_SAMPLES}")
    
    print(f"\n🗳️ ENSEMBLE:")
    print(f"   Voting strategies: {VOTING_STRATEGIES}")
    print(f"   Default: {DEFAULT_VOTING_STRATEGY}")
    
    print(f"\n✅ SUCCESS CRITERIA:")
    for metric, value in SUCCESS_CRITERIA.items():
        print(f"   {metric}: {value}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_config()
