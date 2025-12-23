
import os
import sys
import json
import numpy as np
import glob
from datetime import datetime

# Add current directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

class N2KAttackGenerator:
    """
    Generates realistic NMEA 2000 attacks based on physical constraints.
    """
    def __init__(self):
        self.scaler_params = self._load_scaler_params()
        # Signal Setup
        self.signal_names = [
            'wind_speed', 'wind_angle', 'yaw', 'cog', 'heading',
            'roll', 'rudder_angle_order', 'rudder_position', 'rate_of_turn',
            'depth', 'variation', 'latitude', 'longitude', 'pitch', 'sog'
        ]
        self.aggregations = ['mean', 'max', 'min', 'std']
        
    def _load_scaler_params(self):
        try:
            with open(config.SCALER_PARAMS_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load scaler params: {e}")
            return {}

    def get_signal_indices(self, signal_name):
        """Get the 4 feaure indices (mean, max, min, std) for a signal."""
        try:
            base_idx = self.signal_names.index(signal_name) * 4
            return list(range(base_idx, base_idx + 4))
        except ValueError:
            return []

    # =========================================================================
    # ATTACK IMPLEMENTATIONS
    # =========================================================================
    
    def generate_spike(self, data, target_signal, intensity, duration):
        """Sudden value jump (Spoofing)."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(target_signal)
        
        # Random position
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        # Direction
        direction = np.random.choice([-1, 1])
        offset = direction * intensity
        
        for idx in indices:
            val = attacked[start, idx]
            # NO CLIPPING - allow out-of-range values for anomaly detection
            attacked[start:end, idx] = val + offset
            
        return attacked

    def generate_constant(self, data, target_signal, intensity, duration):
        """Freeze signal (Jamming/Failure). Higher intensity = freeze at extreme values."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(target_signal)
        
        start = int(np.random.uniform(0, 1 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        for idx in indices:
            # Use intensity to determine freeze value
            # Low intensity: freeze at current value (stealthy but hard to detect)
            # Medium intensity: freeze at abnormal but valid value
            # High intensity: freeze at extreme value (0 or 1 in normalized space)
            if intensity >= 1.0:  # High+ intensity
                # Freeze at extreme values (0 or 1)
                val = 0.0 if np.random.random() > 0.5 else 1.0
            elif intensity >= 0.5:  # Medium intensity  
                # Freeze at a value offset from current
                current_val = attacked[start, idx]
                offset = 0.3 * (1 if np.random.random() > 0.5 else -1)
                val = np.clip(current_val + offset, 0, 1)
            else:  # Low intensity
                val = attacked[start, idx]  # Freeze at current (original behavior)
            
            attacked[start:end, idx] = val
            
        return attacked

    def generate_drift(self, data, target_signal, intensity, duration):
        """Gradual change (Man-in-the-Middle)."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(target_signal)
        
        start = int((1 - duration) * time_steps / 2) # Start middle-ish
        end = time_steps
        length = end - start
        
        direction = np.random.choice([-1, 1])
        
        # Limit by Physics
        max_rate = config.PHYSICAL_CONSTRAINTS.get(target_signal, 0.1)
        max_drift = min(intensity, max_rate * length)
        
        ramp = np.linspace(0, direction * max_drift, length)
        
        for idx in indices:
            original = attacked[start:end, idx]
            # NO CLIPPING - allow drift to go outside [0,1] range
            attacked[start:end, idx] = original + ramp
            
        return attacked

    def generate_noise(self, data, target_signal, intensity, duration):
        """Random noise interference."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(target_signal)
        
        start = int(np.random.uniform(0, 1 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        for idx in indices:
            noise = np.random.normal(0, intensity * 0.5, length) 
            # NO CLIPPING - noise should create out-of-range values
            attacked[start:end, idx] = attacked[start:end, idx] + noise
            
        return attacked

    def generate_scaling(self, data, target_signal, intensity, duration):
        """Calibration attack (multiply values)."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(target_signal)
        
        start = int(np.random.uniform(0, 1 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        factor = 1.0 + (intensity if np.random.random() > 0.5 else -intensity/2)
        
        for idx in indices:
            # NO CLIPPING - scaling should create out-of-range values
            attacked[start:end, idx] = attacked[start:end, idx] * factor
            
        return attacked

    def generate_replay(self, data, target_signal, intensity, duration):
        """Replay attack - higher intensity adds temporal anomalies."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(target_signal)
        
        segment_len = max(3, int(duration * time_steps))
        
        # High intensity: reverse temporal order (creates obvious discontinuity)
        if intensity >= 1.0:
            if time_steps < segment_len:
                return attacked
            source_start = np.random.randint(0, time_steps - segment_len)
            for idx in indices:
                segment = attacked[source_start:source_start+segment_len, idx].copy()
                attacked[source_start:source_start+segment_len, idx] = segment[::-1]  # Reverse
            return attacked
        
        # Medium intensity: replay with time shift (creates phase mismatch)
        elif intensity >= 0.5:
            if time_steps < segment_len * 2:
                return attacked
            source_start = 0
            target_start = time_steps - segment_len
            for idx in indices:
                # Copy beginning to end (creates temporal discontinuity)
                attacked[target_start:target_start+segment_len, idx] = \
                    data[source_start:source_start+segment_len, idx] + 0.1 * np.random.randn()
            return attacked
        
        # Low intensity: original behavior (copy earlier to later)
        if time_steps < segment_len * 2:
            return attacked
            
        source_start = np.random.randint(0, time_steps - segment_len * 2 + 1)
        target_start = np.random.randint(source_start + segment_len, time_steps - segment_len + 1)
        
        for idx in indices:
            attacked[target_start:target_start+segment_len, idx] = \
                data[source_start:source_start+segment_len, idx]
                
        return attacked

    def generate(self, data, attack_type, target_signal, intensity_name):
        intensity = config.ATTACK_INTENSITIES[intensity_name]
        # Random duration from list
        duration = 0.5 # default
        
        # Map types to functions
        methods = {
            'spike': self.generate_spike,
            'constant': self.generate_constant,
            'drift': self.generate_drift,
            'noise': self.generate_noise,
            'scaling': self.generate_scaling,
            'replay': self.generate_replay
        }
        
        if attack_type in methods:
            return methods[attack_type](data, target_signal, intensity, duration)
        else:
            print(f"Unknown attack: {attack_type}")
            return data

def main():
    print("="*60)
    print("PHASE 3: GENERATING SYNTHETIC ATTACKS")
    print("="*60)
    
    if not os.path.exists(config.ATTACKS_DIR):
        os.makedirs(config.ATTACKS_DIR)

    generator = N2KAttackGenerator()
    
    # 1. Find all Test Data files
    # We look in Phase1/data/50s_window/sampling_1s/test.npy etc
    
    # Manually defined configs from Phase 1 structure
    # Or scan directories. simpler to iterate known configs if defined.
    # But let's scan to be robust.
    
    configs_found = []
    
    # Walk Phase 1 Data
    for time_step in [50, 75, 100]:
        for sampling in [1, 5, 10]:
            path = os.path.join(config.TEST_DATA_DIR, f"{time_step}s_window", f"sampling_{sampling}s", "test.npy")
            if os.path.exists(path):
                configs_found.append((time_step, sampling, path))

    print(f"Found {len(configs_found)} model configurations.")

    for time_step, sampling, data_path in configs_found:
        config_name = f"{time_step}s_{sampling}s"
        print(f"\nProcessing {config_name}...")
        
        # Load Normal Data
        # [samples, time, features + timestamp]
        try:
            data_raw = np.load(data_path, allow_pickle=True)
            # Drop timestamp (col 0)
            data_normal = data_raw[:, :, 1:].astype(np.float32)
            n_samples = data_normal.shape[0]
        except Exception as e:
            print(f"  Error loading {data_path}: {e}")
            continue

        if n_samples == 0:
            print("  No samples found.")
            continue

        # Generate Attacks
        all_attacks = []
        all_labels = [] # 1 for attack
        all_metadata = []
        
        for attack_type in config.ATTACK_TYPES:
            print(f"  - Generating {attack_type}...", end="", flush=True)
            count = 0
            
            for _ in range(config.NUM_ATTACK_SAMPLES):
                # Pick random normal window
                idx = np.random.randint(0, n_samples)
                window = data_normal[idx].copy()
                
                # Pick random target
                target = np.random.choice(config.ATTACK_TARGET_SIGNALS)
                intensity_name = np.random.choice(list(config.ATTACK_INTENSITIES.keys()))
                
                # Generate
                attacked_window = generator.generate(window, attack_type, target, intensity_name)
                
                all_attacks.append(attacked_window)
                all_labels.append(1)
                all_metadata.append({
                    'attack_type': attack_type,
                    'target_signal': target,
                    'intensity': intensity_name,
                    'origin_idx': int(idx)
                })
                count += 1
            print(f" Done ({count})")

        # Save merged attacks for this config
        # Shape: (N, Time, Feat)
        attacks_np = np.array(all_attacks, dtype=np.float32)
        
        # Reshape for CNN (add channel)
        # (N, Time, Feat, 1)
        attacks_cnn = attacks_np.reshape(attacks_np.shape[0], attacks_np.shape[1], attacks_np.shape[2], 1)
        
        save_path = os.path.join(config.ATTACKS_DIR, f"attacks_{config_name}.npy")
        np.save(save_path, attacks_cnn)
        
        # Save metadata
        meta_path = os.path.join(config.ATTACKS_DIR, f"attacks_{config_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
            
        print(f"  Saved {len(all_attacks)} attacks to {save_path}")

    print("\nAttack Generation Complete.")

if __name__ == "__main__":
    main()
