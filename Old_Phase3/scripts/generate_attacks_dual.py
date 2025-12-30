"""
Phase 3 - Step 2: Generate Two Attack Scenarios
================================================
Generate two different attack datasets for comparison:

SCENARIO A: "Aggressive Attacks" (Easy to Detect)
- High intensity (50-100% of signal range)
- Long duration (50-100% of window)
- Focus on critical signals (heading, GPS, depth)
- All 6 attack types

SCENARIO B: "Stealthy Attacks" (Hard to Detect)
- Low to medium intensity (10-30% of signal range)
- Short to medium duration (20-50% of window)
- Respects physical constraints more strictly
- Focus on realistic attacks (drift, replay, constant)

Both scenarios use the same test data as base, ensuring fair comparison.

Usage:
    python generate_attacks_dual.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIGS, SIGNAL_NAMES, FEATURE_NAMES, NUM_FEATURES,
    PHYSICAL_CONSTRAINTS, get_test_data_path, ensure_dirs,
    ATTACKS_DIR
)

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIO_A = {
    'name': 'aggressive',
    'description': 'High intensity, long duration attacks (easy to detect)',
    'attack_types': ['spike', 'constant', 'replay', 'drift', 'noise', 'scaling'],
    'target_signals': ['heading', 'cog', 'sog', 'depth', 'latitude', 'longitude'],
    'intensity_range': (0.5, 1.0),      # 50-100% of signal range
    'duration_range': (0.5, 1.0),       # 50-100% of window
    'scale_factors': [0.3, 0.5, 1.5, 2.0, 3.0],
    'respect_physics': False,           # Ignore physical constraints
}

SCENARIO_B = {
    'name': 'stealthy',
    'description': 'Low intensity, short duration attacks (hard to detect)',
    'attack_types': ['drift', 'replay', 'constant', 'noise'],  # More realistic
    'target_signals': ['heading', 'cog', 'sog', 'depth'],      # Navigation focus
    'intensity_range': (0.1, 0.3),      # 10-30% of signal range
    'duration_range': (0.2, 0.5),       # 20-50% of window
    'scale_factors': [0.85, 0.9, 1.1, 1.15],
    'respect_physics': True,            # Respect physical constraints
}


class AttackGenerator:
    """Generate realistic NMEA2000 attacks."""
    
    def __init__(self):
        self.signal_names = SIGNAL_NAMES
        self.feature_names = FEATURE_NAMES
    
    def get_signal_indices(self, signal_name: str) -> List[int]:
        """Get feature indices for a signal (mean, max, min, std)."""
        base_idx = self.signal_names.index(signal_name) * 4
        return list(range(base_idx, base_idx + 4))
    
    def generate_spike(self, data: np.ndarray, signal: str, 
                       intensity: float, duration: float) -> Tuple[np.ndarray, Dict]:
        """Sudden value jump attack."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        # Random position
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        # Apply spike
        direction = np.random.choice([-1, 1])
        for idx in indices:
            original = attacked[start, idx]
            attacked[start:end, idx] = np.clip(original + direction * intensity, 0, 1)
        
        return attacked, {
            'type': 'spike', 'signal': signal, 'intensity': intensity,
            'duration': duration, 'start': start, 'end': end
        }
    
    def generate_constant(self, data: np.ndarray, signal: str,
                          intensity: float, duration: float) -> Tuple[np.ndarray, Dict]:
        """Freeze signal at fixed value."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0, 1 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        # Freeze at current value
        for idx in indices:
            attacked[start:end, idx] = attacked[start, idx]
        
        return attacked, {
            'type': 'constant', 'signal': signal, 'intensity': 0,
            'duration': duration, 'start': start, 'end': end
        }
    
    def generate_replay(self, data: np.ndarray, signal: str,
                        intensity: float, duration: float) -> Tuple[np.ndarray, Dict]:
        """Replay old data segment."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        segment_len = max(3, int(duration * time_steps))
        
        # Copy from start to later position
        source = 0
        target = min(time_steps - segment_len, 
                     int(np.random.uniform(0.4, 0.8) * time_steps))
        
        for idx in indices:
            attacked[target:target+segment_len, idx] = data[source:source+segment_len, idx]
        
        return attacked, {
            'type': 'replay', 'signal': signal, 'intensity': 0,
            'duration': duration, 'source': source, 'target': target
        }
    
    def generate_drift(self, data: np.ndarray, signal: str,
                       intensity: float, duration: float,
                       respect_physics: bool = False) -> Tuple[np.ndarray, Dict]:
        """Gradual value deviation."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int((1 - duration) * time_steps / 2)
        end = time_steps
        length = end - start
        
        # Get max rate if respecting physics
        if respect_physics:
            max_rate = PHYSICAL_CONSTRAINTS.get(signal, 0.1)
            max_drift = min(intensity, max_rate * length)
        else:
            max_drift = intensity
        
        # Create linear drift
        direction = np.random.choice([-1, 1])
        drift = np.linspace(0, direction * max_drift, length)
        
        for idx in indices:
            attacked[start:end, idx] = np.clip(attacked[start:end, idx] + drift, 0, 1)
        
        return attacked, {
            'type': 'drift', 'signal': signal, 'intensity': intensity,
            'duration': duration, 'start': start, 'max_drift': float(max_drift)
        }
    
    def generate_noise(self, data: np.ndarray, signal: str,
                       intensity: float, duration: float) -> Tuple[np.ndarray, Dict]:
        """Add random noise."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0, 1 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        for idx in indices:
            noise = np.random.normal(0, intensity, end - start)
            attacked[start:end, idx] = np.clip(attacked[start:end, idx] + noise, 0, 1)
        
        return attacked, {
            'type': 'noise', 'signal': signal, 'intensity': intensity,
            'duration': duration, 'start': start, 'end': end
        }
    
    def generate_scaling(self, data: np.ndarray, signal: str,
                         scale_factor: float, duration: float) -> Tuple[np.ndarray, Dict]:
        """Scale values by factor."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0, 1 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        for idx in indices:
            attacked[start:end, idx] = np.clip(attacked[start:end, idx] * scale_factor, 0, 1)
        
        return attacked, {
            'type': 'scaling', 'signal': signal, 'scale': scale_factor,
            'duration': duration, 'start': start, 'end': end
        }
    
    def generate_attack(self, data: np.ndarray, attack_type: str, signal: str,
                        intensity: float, duration: float,
                        scale_factor: float = 1.5,
                        respect_physics: bool = False) -> Tuple[np.ndarray, Dict]:
        """Generate attack of specified type."""
        if attack_type == 'spike':
            return self.generate_spike(data, signal, intensity, duration)
        elif attack_type == 'constant':
            return self.generate_constant(data, signal, intensity, duration)
        elif attack_type == 'replay':
            return self.generate_replay(data, signal, intensity, duration)
        elif attack_type == 'drift':
            return self.generate_drift(data, signal, intensity, duration, respect_physics)
        elif attack_type == 'noise':
            return self.generate_noise(data, signal, intensity, duration)
        elif attack_type == 'scaling':
            return self.generate_scaling(data, signal, scale_factor, duration)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")


def load_normal_data(window_size: int, sampling_period: int) -> np.ndarray:
    """Load normal test data."""
    data_path = get_test_data_path(window_size, sampling_period)
    data = np.load(data_path, allow_pickle=True)
    return data[:, :, 1:].astype(np.float32)  # Remove timestamp


def generate_scenario(
    scenario: Dict,
    window_size: int,
    sampling_period: int,
    generator: AttackGenerator
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate attacks for a scenario.
    
    Strategy: Generate ONE attack per normal test window.
    This gives balanced dataset: same #attacks as #normal samples.
    
    Returns:
        attack_data: Array of attacked windows
        labels: Array of 1s (all attacks)
        metadata: List of attack info
    """
    # Load normal data
    normal_data = load_normal_data(window_size, sampling_period)
    n_windows = normal_data.shape[0]
    
    print(f"\n  Scenario: {scenario['name']}")
    print(f"  Normal windows: {n_windows}")
    print(f"  Attack types: {scenario['attack_types']}")
    print(f"  Target signals: {scenario['target_signals']}")
    print(f"  Intensity range: {scenario['intensity_range']}")
    print(f"  Duration range: {scenario['duration_range']}")
    
    attack_samples = []
    metadata_list = []
    
    for i in range(n_windows):
        # Use each normal window once
        base_window = normal_data[i].copy()
        
        # Random attack parameters from scenario
        attack_type = np.random.choice(scenario['attack_types'])
        target_signal = np.random.choice(scenario['target_signals'])
        
        intensity = np.random.uniform(*scenario['intensity_range'])
        duration = np.random.uniform(*scenario['duration_range'])
        scale_factor = np.random.choice(scenario['scale_factors'])
        
        # Generate attack
        attacked, meta = generator.generate_attack(
            base_window, attack_type, target_signal,
            intensity=intensity, duration=duration,
            scale_factor=scale_factor,
            respect_physics=scenario['respect_physics']
        )
        
        meta['window_idx'] = i
        meta['scenario'] = scenario['name']
        
        attack_samples.append(attacked)
        metadata_list.append(meta)
    
    # Convert to array
    attack_array = np.array(attack_samples, dtype=np.float32)
    labels = np.ones(n_windows, dtype=np.int32)
    
    print(f"  Generated: {len(attack_samples)} attacks")
    
    return attack_array, labels, metadata_list


def save_scenario_data(
    scenario_name: str,
    window_size: int,
    sampling_period: int,
    attack_data: np.ndarray,
    normal_data: np.ndarray,
    attack_metadata: List[Dict]
):
    """Save attack data and combined dataset."""
    config_name = f"{window_size}s_{sampling_period}s"
    
    # Create scenario directory
    scenario_dir = os.path.join(ATTACKS_DIR, f"scenario_{scenario_name}")
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Reshape for CNN: add channel dimension
    attack_cnn = attack_data.reshape(-1, attack_data.shape[1], NUM_FEATURES, 1)
    normal_cnn = normal_data.reshape(-1, normal_data.shape[1], NUM_FEATURES, 1)
    
    # Save attack data only
    attack_path = os.path.join(scenario_dir, f"{config_name}_attacks.npy")
    np.save(attack_path, attack_cnn)
    
    # Save combined (normal + attack) with labels
    combined_data = np.concatenate([normal_cnn, attack_cnn], axis=0)
    labels = np.concatenate([
        np.zeros(len(normal_cnn), dtype=np.int32),  # Normal = 0
        np.ones(len(attack_cnn), dtype=np.int32)     # Attack = 1
    ])
    
    combined_path = os.path.join(scenario_dir, f"{config_name}_combined.npy")
    labels_path = os.path.join(scenario_dir, f"{config_name}_labels.npy")
    np.save(combined_path, combined_data)
    np.save(labels_path, labels)
    
    # Save metadata
    meta_path = os.path.join(scenario_dir, f"{config_name}_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'scenario': scenario_name,
            'config': config_name,
            'n_normal': len(normal_cnn),
            'n_attacks': len(attack_cnn),
            'n_total': len(combined_data),
            'attack_shape': list(attack_cnn.shape),
            'attacks': attack_metadata
        }, f, indent=2)
    
    print(f"  Saved: {attack_path}")
    print(f"  Saved: {combined_path} (shape: {combined_data.shape})")
    print(f"  Saved: {labels_path} (0s: {len(normal_cnn)}, 1s: {len(attack_cnn)})")
    
    return {
        'attack_path': attack_path,
        'combined_path': combined_path,
        'labels_path': labels_path,
        'n_normal': len(normal_cnn),
        'n_attacks': len(attack_cnn)
    }


def main():
    """Generate both attack scenarios for all model configurations."""
    print("=" * 80)
    print("PHASE 3 - STEP 2: GENERATE DUAL ATTACK SCENARIOS")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    
    print("\n" + "-" * 80)
    print("SCENARIO A: AGGRESSIVE ATTACKS (Easy to Detect)")
    print("-" * 80)
    print(f"  Attack types: {SCENARIO_A['attack_types']}")
    print(f"  Intensity: {SCENARIO_A['intensity_range'][0]*100:.0f}%-{SCENARIO_A['intensity_range'][1]*100:.0f}%")
    print(f"  Duration: {SCENARIO_A['duration_range'][0]*100:.0f}%-{SCENARIO_A['duration_range'][1]*100:.0f}% of window")
    
    print("\n" + "-" * 80)
    print("SCENARIO B: STEALTHY ATTACKS (Hard to Detect)")
    print("-" * 80)
    print(f"  Attack types: {SCENARIO_B['attack_types']}")
    print(f"  Intensity: {SCENARIO_B['intensity_range'][0]*100:.0f}%-{SCENARIO_B['intensity_range'][1]*100:.0f}%")
    print(f"  Duration: {SCENARIO_B['duration_range'][0]*100:.0f}%-{SCENARIO_B['duration_range'][1]*100:.0f}% of window")
    
    # Ensure directories
    ensure_dirs()
    
    # Initialize generator
    generator = AttackGenerator()
    
    # Track results
    all_results = {'scenario_A': {}, 'scenario_B': {}}
    
    # Process each model configuration
    for window_size, sampling_period in MODEL_CONFIGS:
        config_name = f"{window_size}s_{sampling_period}s"
        print(f"\n{'='*70}")
        print(f"CONFIGURATION: {config_name}")
        print("=" * 70)
        
        # Load normal data once
        normal_data = load_normal_data(window_size, sampling_period)
        
        # Generate Scenario A
        print("\n  [SCENARIO A: Aggressive]")
        attack_A, _, meta_A = generate_scenario(
            SCENARIO_A, window_size, sampling_period, generator
        )
        result_A = save_scenario_data(
            'A_aggressive', window_size, sampling_period,
            attack_A, normal_data, meta_A
        )
        all_results['scenario_A'][config_name] = result_A
        
        # Generate Scenario B
        print("\n  [SCENARIO B: Stealthy]")
        attack_B, _, meta_B = generate_scenario(
            SCENARIO_B, window_size, sampling_period, generator
        )
        result_B = save_scenario_data(
            'B_stealthy', window_size, sampling_period,
            attack_B, normal_data, meta_B
        )
        all_results['scenario_B'][config_name] = result_B
    
    # Print summary
    print("\n" + "=" * 80)
    print("ATTACK GENERATION SUMMARY")
    print("=" * 80)
    
    print("\nScenario A (Aggressive):")
    print(f"{'Config':<15} {'Normal':>8} {'Attacks':>8} {'Total':>8}")
    print("-" * 45)
    total_A = 0
    for config, info in all_results['scenario_A'].items():
        print(f"{config:<15} {info['n_normal']:>8} {info['n_attacks']:>8} {info['n_normal']+info['n_attacks']:>8}")
        total_A += info['n_attacks']
    
    print("\nScenario B (Stealthy):")
    print(f"{'Config':<15} {'Normal':>8} {'Attacks':>8} {'Total':>8}")
    print("-" * 45)
    total_B = 0
    for config, info in all_results['scenario_B'].items():
        print(f"{config:<15} {info['n_normal']:>8} {info['n_attacks']:>8} {info['n_normal']+info['n_attacks']:>8}")
        total_B += info['n_attacks']
    
    print(f"\nTotal attacks generated:")
    print(f"  Scenario A (Aggressive): {total_A}")
    print(f"  Scenario B (Stealthy):   {total_B}")
    print(f"  Combined:                {total_A + total_B}")
    
    print(f"\nData saved to:")
    print(f"  {os.path.join(ATTACKS_DIR, 'scenario_A_aggressive/')}")
    print(f"  {os.path.join(ATTACKS_DIR, 'scenario_B_stealthy/')}")
    
    print(f"\nCompleted: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
