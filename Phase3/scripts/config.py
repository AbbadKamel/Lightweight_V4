
"""
Phase 3 Configuration: Attack Detection & Evaluation
=====================================================
Central configuration for Phase 3 (Attack Generation & Evaluation).
re-implemented based on Project Standards.
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Phase3
PHASE1_DIR = os.path.join(os.path.dirname(BASE_DIR), "Phase1")
PHASE2_DIR = os.path.join(os.path.dirname(BASE_DIR), "Phase2")

# Input Data
TEST_DATA_DIR = os.path.join(PHASE1_DIR, "data")
SCALER_PARAMS_PATH = os.path.join(PHASE1_DIR, "results", "scaler_params.json")

# Output Data
ATTACKS_DIR = os.path.join(BASE_DIR, "attacks")
THRESHOLDS_DIR = os.path.join(BASE_DIR, "thresholds")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# ============================================================================
# ATTACK DEFINITIONS (Use ALL of them)
# ============================================================================
ATTACK_TYPES = [
    'spike',      # Sudden value jump (Spoofing)
    'constant',   # Freeze signal (Jamming)
    'replay',     # Repeat old data (Hacker)
    'drift',      # Gradual change (Man-in-the-Middle)
    'noise',      # Interference
    'scaling'     # Calibration Hack
]

# TARGET SIGNALS (Critical Navigation Data Only)
# We don't attack "Wind Angle" because that's annoying but not fatal.
# We attack "Heading" because that crashes the boat.
ATTACK_TARGET_SIGNALS = [
    'heading',    # Critical: Steering
    'depth',      # Critical: Grounding
    'latitude',   # Critical: GPS Spoofing
    'longitude',  # Critical: GPS Spoofing
    'sog',        # Critical: Speed
    'cog',        # Critical: Course
    'rudder_position' # Critical: Control
]

# ============================================================================
# PHYSICAL CONSTRAINTS (The Laws of Physics)
# ============================================================================
# Max normalized change per second (0.0 to 1.0)
PHYSICAL_CONSTRAINTS = {
    'heading': 0.03,      # Max 10 degrees/sec
    'depth': 0.02,        # Depth changes slowly
    'latitude': 0.001,    # GPS doesn't jump
    'longitude': 0.001,
    'sog': 0.05,
    'rudder_position': 0.1
}

# ============================================================================
# GENERATION SETTINGS
# ============================================================================
# How many attack samples to generate per simple/type
# Total = 6 Types * 50 Samples = 300 Attacks per Model
NUM_ATTACK_SAMPLES = 50 

# Attack Intensity Levels (INCREASED for detectability)
# Original was too subtle (attacks reconstructed too well)
ATTACK_INTENSITIES = {
    'low': 0.5,        # Was 0.1 - now 50% of signal range
    'medium': 1.0,     # Was 0.3 - now 100% of signal range
    'high': 2.0,       # Was 0.5 - now 200% of signal range
    'extreme': 3.0     # Was 1.0 - now 300% of signal range
}
