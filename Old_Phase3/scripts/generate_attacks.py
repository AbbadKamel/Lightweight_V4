"""
Phase 3 - Step 2: Generate Synthetic Attacks
=============================================
Generate realistic attack data based on NMEA2000 constraints.

Attack Types:
1. Spike - Sudden value jump (sensor spoofing)
2. Constant - Freeze signal at fixed value (sensor failure/jamming)
3. Replay - Repeat old data segment (replay attack)
4. Drift - Gradual deviation over time (man-in-the-middle)
5. Noise - Add random noise (interference)
6. Scaling - Scale values by factor (calibration attack)

All attacks respect:
- NMEA2000 value ranges (from scaler_params.json)
- Physical constraints (realistic rate of change)
- Data remains in normalized [0, 1] range

Usage:
    python generate_attacks.py
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
    MODEL_CONFIGS, ATTACK_TYPES, ATTACK_TARGET_SIGNALS, ATTACK_INTENSITIES,
    ATTACK_DURATIONS, PHYSICAL_CONSTRAINTS, NUM_ATTACK_SAMPLES,
    SIGNAL_NAMES, AGGREGATIONS, FEATURE_NAMES, NUM_FEATURES,
    get_test_data_path, get_attack_data_path, load_scaler_params, ensure_dirs
)


class N2KAttackGenerator:
    """
    Generate realistic NMEA2000 attack data.
    
    Attacks are applied to normalized data and respect:
    - Valid value ranges (0-1 normalized)
    - Physical rate-of-change constraints
    - Signal correlations where appropriate
    """
    
    def __init__(self):
        """Initialize attack generator with scaler parameters."""
        self.scaler_params = load_scaler_params()
        self.signal_names = SIGNAL_NAMES
        self.feature_names = FEATURE_NAMES
        self.aggregations = AGGREGATIONS
        
    def get_signal_indices(self, signal_name: str) -> List[int]:
        """
        Get feature indices for a signal (mean, max, min, std).
        
        Args:
            signal_name: Base signal name (e.g., 'heading')
        
        Returns:
            List of 4 indices for the signal's aggregations
        """
        base_idx = self.signal_names.index(signal_name) * 4
        return list(range(base_idx, base_idx + 4))
    
    def generate_spike_attack(
        self,
        data: np.ndarray,
        target_signal: str,
        intensity: float = 0.5,
        duration: float = 0.2,
        spike_position: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate spike injection attack.
        
        Simulates sensor spoofing where a value suddenly jumps
        to an incorrect value.
        
        Args:
            data: Normal window data (time_steps, features)
            target_signal: Signal to attack
            intensity: Spike magnitude (fraction of range, 0-1)
            duration: Attack duration (fraction of window)
            spike_position: Position in window (0-1), random if None
        
        Returns:
            Attacked data and attack metadata
        """
        attacked = data.copy()
        time_steps = data.shape[0]
        
        # Get signal indices
        signal_indices = self.get_signal_indices(target_signal)
        
        # Determine attack timing
        if spike_position is None:
            spike_position = np.random.uniform(0.1, 0.9 - duration)
        
        start_idx = int(spike_position * time_steps)
        attack_length = max(1, int(duration * time_steps))
        end_idx = min(start_idx + attack_length, time_steps)
        
        # Generate spike value
        # Direction: randomly up or down
        direction = np.random.choice([-1, 1])
        spike_offset = direction * intensity
        
        # Apply spike to all aggregations of the signal
        for idx in signal_indices:
            original_value = attacked[start_idx, idx]
            spike_value = np.clip(original_value + spike_offset, 0, 1)
            attacked[start_idx:end_idx, idx] = spike_value
        
        metadata = {
            'attack_type': 'spike',
            'target_signal': target_signal,
            'intensity': intensity,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'direction': int(direction),
            'affected_features': signal_indices
        }
        
        return attacked, metadata
    
    def generate_constant_attack(
        self,
        data: np.ndarray,
        target_signal: str,
        duration: float = 0.5,
        start_position: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate constant injection attack (signal freeze).
        
        Simulates sensor failure or jamming where values get stuck.
        
        Args:
            data: Normal window data (time_steps, features)
            target_signal: Signal to attack
            duration: Attack duration (fraction of window)
            start_position: Position in window (0-1), random if None
        
        Returns:
            Attacked data and attack metadata
        """
        attacked = data.copy()
        time_steps = data.shape[0]
        
        # Get signal indices
        signal_indices = self.get_signal_indices(target_signal)
        
        # Determine attack timing
        if start_position is None:
            start_position = np.random.uniform(0, 1 - duration)
        
        start_idx = int(start_position * time_steps)
        attack_length = max(1, int(duration * time_steps))
        end_idx = min(start_idx + attack_length, time_steps)
        
        # Freeze at the value at start of attack
        for idx in signal_indices:
            freeze_value = attacked[start_idx, idx]
            attacked[start_idx:end_idx, idx] = freeze_value
        
        metadata = {
            'attack_type': 'constant',
            'target_signal': target_signal,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'affected_features': signal_indices
        }
        
        return attacked, metadata
    
    def generate_replay_attack(
        self,
        data: np.ndarray,
        target_signal: str,
        duration: float = 0.3,
        replay_offset: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate replay attack.
        
        Simulates recording and replaying old sensor data.
        Most realistic attack - uses actual data patterns.
        
        Args:
            data: Normal window data (time_steps, features)
            target_signal: Signal to attack
            duration: Segment duration (fraction of window)
            replay_offset: How far back to copy from, random if None
        
        Returns:
            Attacked data and attack metadata
        """
        attacked = data.copy()
        time_steps = data.shape[0]
        
        # Get signal indices
        signal_indices = self.get_signal_indices(target_signal)
        
        # Calculate segment length
        segment_length = max(3, int(duration * time_steps))
        
        # Determine source and target positions
        # Source: early in window, Target: later in window
        if replay_offset is None:
            replay_offset = np.random.randint(segment_length + 5, time_steps - segment_length)
        
        source_start = np.random.randint(0, min(segment_length, time_steps - segment_length - 5))
        target_start = min(source_start + replay_offset, time_steps - segment_length)
        
        # Copy old data to new position
        for idx in signal_indices:
            attacked[target_start:target_start + segment_length, idx] = \
                data[source_start:source_start + segment_length, idx]
        
        metadata = {
            'attack_type': 'replay',
            'target_signal': target_signal,
            'duration': duration,
            'source_start': source_start,
            'target_start': target_start,
            'segment_length': segment_length,
            'affected_features': signal_indices
        }
        
        return attacked, metadata
    
    def generate_drift_attack(
        self,
        data: np.ndarray,
        target_signal: str,
        intensity: float = 0.3,
        duration: float = 0.8
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate gradual drift attack.
        
        Simulates man-in-the-middle attack with subtle value deviation.
        Respects physical rate-of-change constraints.
        
        Args:
            data: Normal window data (time_steps, features)
            target_signal: Signal to attack
            intensity: Final drift magnitude (fraction of range)
            duration: Attack duration (fraction of window)
        
        Returns:
            Attacked data and attack metadata
        """
        attacked = data.copy()
        time_steps = data.shape[0]
        
        # Get signal indices
        signal_indices = self.get_signal_indices(target_signal)
        
        # Get physical constraint for this signal
        max_rate = PHYSICAL_CONSTRAINTS.get(target_signal, 0.1)
        
        # Calculate drift profile
        start_idx = int((1 - duration) * time_steps / 2)  # Start in middle portion
        end_idx = time_steps
        attack_length = end_idx - start_idx
        
        # Direction: randomly up or down
        direction = np.random.choice([-1, 1])
        
        # Create gradual drift (linear ramp)
        # Limit by physical constraint
        max_total_drift = min(intensity, max_rate * attack_length)
        drift_profile = np.linspace(0, direction * max_total_drift, attack_length)
        
        # Apply drift
        for idx in signal_indices:
            original = attacked[start_idx:end_idx, idx]
            attacked[start_idx:end_idx, idx] = np.clip(original + drift_profile, 0, 1)
        
        metadata = {
            'attack_type': 'drift',
            'target_signal': target_signal,
            'intensity': intensity,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'direction': int(direction),
            'max_drift': float(max_total_drift),
            'affected_features': signal_indices
        }
        
        return attacked, metadata
    
    def generate_noise_attack(
        self,
        data: np.ndarray,
        target_signal: str,
        intensity: float = 0.1,
        duration: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate noise injection attack.
        
        Simulates electromagnetic interference or sensor malfunction.
        
        Args:
            data: Normal window data (time_steps, features)
            target_signal: Signal to attack
            intensity: Noise standard deviation (fraction of range)
            duration: Attack duration (fraction of window)
        
        Returns:
            Attacked data and attack metadata
        """
        attacked = data.copy()
        time_steps = data.shape[0]
        
        # Get signal indices
        signal_indices = self.get_signal_indices(target_signal)
        
        # Determine attack timing
        start_idx = int(np.random.uniform(0, 1 - duration) * time_steps)
        attack_length = max(1, int(duration * time_steps))
        end_idx = min(start_idx + attack_length, time_steps)
        
        # Generate noise
        for idx in signal_indices:
            noise = np.random.normal(0, intensity, end_idx - start_idx)
            attacked[start_idx:end_idx, idx] = np.clip(
                attacked[start_idx:end_idx, idx] + noise, 0, 1
            )
        
        metadata = {
            'attack_type': 'noise',
            'target_signal': target_signal,
            'intensity': intensity,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'affected_features': signal_indices
        }
        
        return attacked, metadata
    
    def generate_scaling_attack(
        self,
        data: np.ndarray,
        target_signal: str,
        scale_factor: float = 1.5,
        duration: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate scaling attack (calibration manipulation).
        
        Simulates calibration attack where values are multiplied
        by a factor.
        
        Args:
            data: Normal window data (time_steps, features)
            target_signal: Signal to attack
            scale_factor: Multiplication factor
            duration: Attack duration (fraction of window)
        
        Returns:
            Attacked data and attack metadata
        """
        attacked = data.copy()
        time_steps = data.shape[0]
        
        # Get signal indices
        signal_indices = self.get_signal_indices(target_signal)
        
        # Determine attack timing
        start_idx = int(np.random.uniform(0, 1 - duration) * time_steps)
        attack_length = max(1, int(duration * time_steps))
        end_idx = min(start_idx + attack_length, time_steps)
        
        # Apply scaling
        for idx in signal_indices:
            attacked[start_idx:end_idx, idx] = np.clip(
                attacked[start_idx:end_idx, idx] * scale_factor, 0, 1
            )
        
        metadata = {
            'attack_type': 'scaling',
            'target_signal': target_signal,
            'scale_factor': scale_factor,
            'duration': duration,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'affected_features': signal_indices
        }
        
        return attacked, metadata
    
    def generate_attack(
        self,
        data: np.ndarray,
        attack_type: str,
        target_signal: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate attack of specified type.
        
        Args:
            data: Normal window data
            attack_type: Type of attack
            target_signal: Signal to attack
            **kwargs: Attack-specific parameters
        
        Returns:
            Attacked data and metadata
        """
        attack_generators = {
            'spike': self.generate_spike_attack,
            'constant': self.generate_constant_attack,
            'replay': self.generate_replay_attack,
            'drift': self.generate_drift_attack,
            'noise': self.generate_noise_attack,
            'scaling': self.generate_scaling_attack
        }
        
        if attack_type not in attack_generators:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return attack_generators[attack_type](data, target_signal, **kwargs)


def load_normal_data(window_size: int, sampling_period: int) -> np.ndarray:
    """Load normal test data for a configuration."""
    data_path = get_test_data_path(window_size, sampling_period)
    data = np.load(data_path, allow_pickle=True)
    # Remove timestamp column
    return data[:, :, 1:].astype(np.float32)


def generate_attacks_for_config(
    window_size: int,
    sampling_period: int,
    generator: N2KAttackGenerator,
    num_samples: int = NUM_ATTACK_SAMPLES
) -> Dict[str, Dict]:
    """
    Generate all attack types for a configuration.
    
    Args:
        window_size: Window size in seconds
        sampling_period: Sampling period in seconds
        generator: Attack generator instance
        num_samples: Number of samples per attack type
    
    Returns:
        Dictionary with attack data and metadata
    """
    config_name = f"{window_size}s_{sampling_period}s"
    print(f"\n{'='*60}")
    print(f"Generating attacks for: {config_name}")
    print("=" * 60)
    
    # Load normal data
    normal_data = load_normal_data(window_size, sampling_period)
    n_windows = normal_data.shape[0]
    print(f"  Normal windows available: {n_windows}")
    
    results = {}
    
    for attack_type in ATTACK_TYPES:
        print(f"\n  Generating {attack_type} attacks...")
        
        attack_samples = []
        attack_metadata = []
        labels = []  # 1 = attack, 0 = normal
        
        for i in range(num_samples):
            # Select random normal window as base
            window_idx = np.random.randint(0, n_windows)
            normal_window = normal_data[window_idx].copy()
            
            # Select random target signal
            target_signal = np.random.choice(ATTACK_TARGET_SIGNALS)
            
            # Select random intensity and duration
            intensity_name = np.random.choice(list(ATTACK_INTENSITIES.keys()))
            intensity = ATTACK_INTENSITIES[intensity_name]
            
            duration_name = np.random.choice(list(ATTACK_DURATIONS.keys()))
            duration = ATTACK_DURATIONS[duration_name]
            
            # Generate attack
            if attack_type == 'scaling':
                # Special case: scale factor instead of intensity
                scale_factor = np.random.choice([0.5, 0.7, 1.3, 1.5, 2.0])
                attacked, meta = generator.generate_attack(
                    normal_window, attack_type, target_signal,
                    scale_factor=scale_factor, duration=duration
                )
            else:
                attacked, meta = generator.generate_attack(
                    normal_window, attack_type, target_signal,
                    intensity=intensity, duration=duration
                )
            
            # Add additional metadata
            meta['source_window_idx'] = int(window_idx)
            meta['intensity_level'] = intensity_name
            meta['duration_level'] = duration_name
            
            attack_samples.append(attacked)
            attack_metadata.append(meta)
            labels.append(1)  # Attack label
        
        # Convert to arrays
        attack_array = np.array(attack_samples, dtype=np.float32)
        
        # Save attack data
        attack_path = get_attack_data_path(attack_type, window_size, sampling_period)
        
        # Reshape for CNN: add channel dimension
        attack_cnn = attack_array.reshape(-1, attack_array.shape[1], NUM_FEATURES, 1)
        np.save(attack_path, attack_cnn)
        
        # Save metadata
        meta_path = attack_path.replace('.npy', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'attack_type': attack_type,
                'config': config_name,
                'n_samples': num_samples,
                'samples': attack_metadata
            }, f, indent=2)
        
        print(f"    Generated {num_samples} samples → {attack_path}")
        
        results[attack_type] = {
            'n_samples': num_samples,
            'path': attack_path,
            'shape': attack_cnn.shape
        }
    
    return results


def main():
    """Generate attacks for all configurations."""
    print("=" * 80)
    print("PHASE 3 - STEP 2: GENERATE SYNTHETIC ATTACKS")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nAttack types: {ATTACK_TYPES}")
    print(f"Target signals: {ATTACK_TARGET_SIGNALS}")
    print(f"Samples per attack: {NUM_ATTACK_SAMPLES}")
    
    # Ensure output directories exist
    ensure_dirs()
    
    # Initialize generator
    generator = N2KAttackGenerator()
    
    # Generate for all configurations
    all_results = {}
    for window_size, sampling_period in MODEL_CONFIGS:
        try:
            results = generate_attacks_for_config(
                window_size, sampling_period, generator
            )
            config_name = f"{window_size}s_{sampling_period}s"
            all_results[config_name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("ATTACK GENERATION SUMMARY")
    print("=" * 80)
    
    total_attacks = 0
    for config, attacks in all_results.items():
        print(f"\n{config}:")
        for attack_type, info in attacks.items():
            print(f"  {attack_type}: {info['n_samples']} samples, shape={info['shape']}")
            total_attacks += info['n_samples']
    
    print(f"\nTotal attack samples generated: {total_attacks}")
    print(f"Attack data saved to: {os.path.dirname(get_attack_data_path('spike', 50, 1))}")
    print(f"\nCompleted: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
