
import os
import sys
import json
import numpy as np
import argparse

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIGS, SIGNAL_NAMES,
    PHYSICAL_CONSTRAINTS, ensure_dirs
)

# ============================================================================
# CANSHIELD ATTACK DEFINITIONS
# ============================================================================
# Based on the CANShield paper (2023) and SynCAN dataset
# 1. Flooding: (Not applicable to signal-level time-series, this is a bus-level attack)
# 2. Suppress: Signal stops updating (becomes constant or 0)
# 3. Plateau: Signal freezes at a specific value for a duration
# 4. Continuous: Signal drifts away from true value (slowly)
# 5. Playback: Replay past valid data

ATTACK_CONFIG = {
    'plateau': {
        'description': 'Signal freezes at a specific value (Masquerade/Injection)',
        'duration_range': (0.1, 0.4), # 10% to 40% of the window
        'intensity': None # Value is taken from the start of the attack
    },
    'continuous': {
        'description': 'Signal drifts away from true value (Masquerade/Injection)',
        'duration_range': (0.2, 0.6), # 20% to 60% of the window
        'intensity_range': (0.1, 0.3) # Drift magnitude (0.1 to 0.3 normalized units)
    },
    'playback': {
        'description': 'Replay past valid data (Masquerade/Replay)',
        'duration_range': (0.2, 0.5), # 20% to 50% of the window
        'intensity': None
    },
    'suppress': {
        'description': 'Signal stops updating (DoS/Injection)',
        'duration_range': (0.1, 0.3), # 10% to 30% of the window
        'value': 0.0 # Or last known value
    }
}

class AttackGeneratorCANShield:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.signal_names = SIGNAL_NAMES 
        
    def get_signal_indices(self, signal_name: str) -> list:
        """Get feature indices for a signal based on available features."""
        indices = []
        for i, feat in enumerate(self.feature_names):
            if feat.startswith(f"{signal_name}_"):
                indices.append(i)
        return indices

    def apply_attack(self, data, attack_type, signal):
        """Apply a specific attack type to a signal."""
        config = ATTACK_CONFIG.get(attack_type)
        if not config:
            raise ValueError(f"Unknown attack type: {attack_type}")
            
        duration = np.random.uniform(*config['duration_range'])
        
        if attack_type == 'plateau':
            return self.generate_plateau(data, signal, duration)
        elif attack_type == 'continuous':
            intensity = np.random.uniform(*config['intensity_range'])
            return self.generate_continuous(data, signal, intensity, duration)
        elif attack_type == 'playback':
            return self.generate_playback(data, signal, duration)
        elif attack_type == 'suppress':
            return self.generate_suppress(data, signal, duration)
            
    def generate_plateau(self, data, signal, duration):
        """Plateau Attack: Signal freezes at a specific value."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        # Freeze at the value of the start index
        freeze_values = attacked[start, indices]
        
        for i, idx in enumerate(indices):
            attacked[start:end, idx] = freeze_values[i]
            
        return attacked, {'type': 'plateau', 'signal': signal, 'start': start, 'end': end}

    def generate_continuous(self, data, signal, intensity, duration):
        """Continuous Attack: Signal drifts away from true value."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        actual_length = end - start
        
        # Linear drift
        drift = np.linspace(0, intensity, actual_length)
        direction = np.random.choice([-1, 1])
        
        for i, idx in enumerate(indices):
            attacked[start:end, idx] = np.clip(attacked[start:end, idx] + direction * drift, 0, 1)
            
        return attacked, {'type': 'continuous', 'signal': signal, 'start': start, 'end': end}

    def generate_playback(self, data, signal, duration):
        """Playback Attack: Replay past valid data."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        length = max(1, int(duration * time_steps))
        start = int(np.random.uniform(0, time_steps - length))
        end = start + length
        
        # Replay from a different random segment in the SAME window (simplified replay)
        # In a real scenario, this would come from a buffer of past windows
        replay_start = int(np.random.uniform(0, time_steps - length))
        
        # Ensure replay segment doesn't overlap perfectly with target (trivial)
        while abs(replay_start - start) < 5:
             replay_start = int(np.random.uniform(0, time_steps - length))

        replay_segment = data[replay_start:replay_start+length, indices]
        attacked[start:end, indices] = replay_segment
        
        return attacked, {'type': 'playback', 'signal': signal, 'start': start, 'end': end}

    def generate_suppress(self, data, signal, duration):
        """Suppress Attack: Signal stops updating (becomes 0 or constant)."""
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        # Suppress to 0 (assuming normalized data, 0 is min)
        for idx in indices:
            attacked[start:end, idx] = 0.0
            
        return attacked, {'type': 'suppress', 'signal': signal, 'start': start, 'end': end}


def generate_attacks_canshield(scenario_name):
    print(f"\n{'='*80}")
    print(f"GENERATING CANSHIELD ATTACKS FOR SCENARIO: {scenario_name}")
    print(f"{'='*80}\n")
    
    # 1. Load Config
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phase1_config_path = os.path.join(project_root, 'Phase1', 'results', f'config_{scenario_name}.json')
    
    with open(phase1_config_path, 'r') as f:
        scenario_config = json.load(f)
    
    feature_names = scenario_config['feature_names']
    
    # 2. Setup Paths
    PHASE1_DATA_DIR = os.path.join(project_root, 'Phase1', f'data_{scenario_name}')
    PHASE3_ATTACKS_DIR = os.path.join(project_root, 'Phase3', f'attacks_{scenario_name}')
    
    if not os.path.exists(PHASE3_ATTACKS_DIR):
        os.makedirs(PHASE3_ATTACKS_DIR)
    
    # 3. Initialize Generator
    generator = AttackGeneratorCANShield(feature_names)
    
    # 4. Generate Attacks for each config
    # Hardcoded from Phase1/scripts/config.py since they are not in the scenario config
    TIME_STEPS = [50, 75, 100]
    SAMPLING_PERIODS = [1, 5, 10]
    
    for window_size in TIME_STEPS:
        for sampling_period in SAMPLING_PERIODS:
            config_name = f"{window_size}s_{sampling_period}s"
            print(f"Processing {config_name}...")
            
            # Load Test Data
            data_dir = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s")
            test_path = os.path.join(data_dir, "test.npy")
            
            if not os.path.exists(test_path):
                print(f"  Test data not found: {test_path}")
                continue
                
            test_data = np.load(test_path, allow_pickle=True)
            
            # Data is already clean (no timestamps) from Phase1/scripts/prepare_training_datasets_scenario.py
            test_signals = test_data.astype(np.float32)
            
            # Generate attacks for each type
            for attack_type in ATTACK_CONFIG.keys():
                attack_dir = os.path.join(PHASE3_ATTACKS_DIR, f"canshield_{attack_type}")
                if not os.path.exists(attack_dir):
                    os.makedirs(attack_dir)
                
                attacked_samples = []
                metadata = []
                
                # Generate 1 attack per signal per sample (limited to avoid explosion)
                # We will generate N attacks where N = number of test samples
                # Randomly selecting signal and sample
                
                for i in range(len(test_signals)):
                    sample = test_signals[i]
                    
                    # Pick a random signal to attack
                    signal = np.random.choice(generator.signal_names)
                    
                    try:
                        attacked_sample, meta = generator.apply_attack(sample, attack_type, signal)
                        
                        # Add to list
                        attacked_samples.append(attacked_sample)
                        metadata.append(meta)
                    except Exception as e:
                        print(f"Error generating {attack_type} on {signal}: {e}")
                        print(f"Sample shape: {sample.shape}")
                        continue

                # Save
                save_path = os.path.join(attack_dir, f"{config_name}.npy")
                meta_path = os.path.join(attack_dir, f"{config_name}_meta.json")
                
                np.save(save_path, np.array(attacked_samples))
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                print(f"    Saved {len(attacked_samples)} {attack_type} attacks")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        generate_attacks_canshield(scenario)
    else:
        print("Please provide scenario name")
