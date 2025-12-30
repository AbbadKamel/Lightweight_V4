
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

SCENARIO_A = {
    'name': 'aggressive',
    'description': 'High intensity, long duration attacks (easy to detect)',
    'attack_types': ['spike', 'constant', 'replay', 'drift', 'noise', 'scaling'],
    'target_signals': ['heading', 'cog', 'sog', 'depth', 'latitude', 'longitude'],
    'intensity_range': (0.5, 1.0),
    'duration_range': (0.5, 1.0),
    'scale_factors': [0.3, 0.5, 1.5, 2.0, 3.0],
    'respect_physics': False,
}

SCENARIO_B = {
    'name': 'stealthy',
    'description': 'Low intensity, short duration attacks (hard to detect)',
    'attack_types': ['drift', 'replay', 'constant', 'noise'],
    'target_signals': ['heading', 'cog', 'sog', 'depth'],
    'intensity_range': (0.1, 0.3),
    'duration_range': (0.2, 0.5),
    'scale_factors': [0.85, 0.9, 1.1, 1.15],
    'respect_physics': True,
}

class AttackGeneratorScenario:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.signal_names = SIGNAL_NAMES # From config (15 signals)
        
    def get_signal_indices(self, signal_name: str) -> list:
        """Get feature indices for a signal based on available features."""
        indices = []
        for i, feat in enumerate(self.feature_names):
            if feat.startswith(f"{signal_name}_"):
                indices.append(i)
        return indices

    def generate_spike(self, data, signal, intensity, duration):
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        direction = np.random.choice([-1, 1])
        for idx in indices:
            original = attacked[start, idx]
            attacked[start:end, idx] = np.clip(original + direction * intensity, 0, 1)
            
        return attacked, {'type': 'spike', 'signal': signal, 'start': start, 'end': end}

    def generate_constant(self, data, signal, intensity, duration):
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        val = np.random.uniform(0, 1)
        for idx in indices:
            attacked[start:end, idx] = val
            
        return attacked, {'type': 'constant', 'signal': signal, 'start': start, 'end': end}

    def generate_replay(self, data, signal, intensity, duration):
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        length = max(1, int(duration * time_steps))
        start = int(np.random.uniform(0, time_steps - length))
        end = start + length
        
        # Replay from random past segment
        replay_start = int(np.random.uniform(0, time_steps - length))
        replay_segment = data[replay_start:replay_start+length, indices]
        
        attacked[start:end, indices] = replay_segment
        
        return attacked, {'type': 'replay', 'signal': signal, 'start': start, 'end': end}

    def generate_drift(self, data, signal, intensity, duration):
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        actual_length = end - start
        
        drift = np.linspace(0, intensity, actual_length)
        direction = np.random.choice([-1, 1])
        
        for i, idx in enumerate(indices):
            attacked[start:end, idx] = np.clip(attacked[start:end, idx] + direction * drift, 0, 1)
            
        return attacked, {'type': 'drift', 'signal': signal, 'start': start, 'end': end}

    def generate_noise(self, data, signal, intensity, duration):
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        actual_length = end - start
        
        noise = np.random.normal(0, intensity, (actual_length, len(indices)))
        attacked[start:end, indices] = np.clip(attacked[start:end, indices] + noise, 0, 1)
        
        return attacked, {'type': 'noise', 'signal': signal, 'start': start, 'end': end}

    def generate_scaling(self, data, signal, intensity, duration):
        attacked = data.copy()
        time_steps = data.shape[0]
        indices = self.get_signal_indices(signal)
        
        start = int(np.random.uniform(0.1, 0.9 - duration) * time_steps)
        length = max(1, int(duration * time_steps))
        end = min(start + length, time_steps)
        
        factor = 1.0 + intensity if np.random.random() > 0.5 else 1.0 - intensity
        attacked[start:end, indices] = np.clip(attacked[start:end, indices] * factor, 0, 1)
        
        return attacked, {'type': 'scaling', 'signal': signal, 'start': start, 'end': end}

    def apply_attack(self, data, attack_type, signal, intensity, duration):
        if attack_type == 'spike':
            return self.generate_spike(data, signal, intensity, duration)
        elif attack_type == 'constant':
            return self.generate_constant(data, signal, intensity, duration)
        elif attack_type == 'replay':
            return self.generate_replay(data, signal, intensity, duration)
        elif attack_type == 'drift':
            return self.generate_drift(data, signal, intensity, duration)
        elif attack_type == 'noise':
            return self.generate_noise(data, signal, intensity, duration)
        elif attack_type == 'scaling':
            return self.generate_scaling(data, signal, intensity, duration)
        return data, {}

def generate_attacks_scenario(scenario_name):
    print("="*80)
    print(f"GENERATING ATTACKS FOR SCENARIO: {scenario_name}")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phase1_config_path = os.path.join(project_root, 'Phase1', 'results', f'config_{scenario_name}.json')
    
    with open(phase1_config_path, 'r') as f:
        scenario_config = json.load(f)
    
    FEATURE_NAMES = scenario_config['feature_names']
    NUM_FEATURES = scenario_config['num_features']
    
    PHASE1_DATA_DIR = os.path.join(project_root, 'Phase1', f'data_{scenario_name}')
    PHASE3_ATTACKS_DIR = os.path.join(project_root, 'Phase3', f'attacks_{scenario_name}')
    
    generator = AttackGeneratorScenario(FEATURE_NAMES)
    
    for window_size, sampling_period in MODEL_CONFIGS:
        config_name = f"{window_size}s_{sampling_period}s"
        print(f"\nProcessing {config_name}...")
        
        # Load Test Data (Normal)
        data_dir = os.path.join(PHASE1_DATA_DIR, f"{window_size}s_window", f"sampling_{sampling_period}s")
        test_path = os.path.join(data_dir, "test.npy")
        
        if not os.path.exists(test_path):
            print(f"Data not found: {test_path}")
            continue
            
        data = np.load(test_path, allow_pickle=True)
        signals = data[:, :, 1:].astype(np.float32) # (samples, time, features)
        
        # Generate for both A and B scenarios
        for scenario_def in [SCENARIO_A, SCENARIO_B]:
            s_name = scenario_def['name']
            print(f"  Generating {s_name} attacks...")
            
            out_dir = os.path.join(PHASE3_ATTACKS_DIR, f"scenario_{s_name}")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            attacked_samples = []
            labels = []
            
            # Generate 1 attack per normal sample (balanced)
            for i in range(len(signals)):
                sample = signals[i]
                
                # 50% chance of attack
                if np.random.random() > 0.5:
                    # Attack
                    attack_type = np.random.choice(scenario_def['attack_types'])
                    signal = np.random.choice(scenario_def['target_signals'])
                    intensity = np.random.uniform(*scenario_def['intensity_range'])
                    duration = np.random.uniform(*scenario_def['duration_range'])
                    
                    attacked, meta = generator.apply_attack(sample, attack_type, signal, intensity, duration)
                    attacked_samples.append(attacked)
                    labels.append(1)
                else:
                    # Normal
                    attacked_samples.append(sample)
                    labels.append(0)
            
            # Save
            X_combined = np.array(attacked_samples)
            # Reshape for CNN
            X_combined = X_combined.reshape(-1, X_combined.shape[1], NUM_FEATURES, 1)
            y_combined = np.array(labels)
            
            np.save(os.path.join(out_dir, f"{config_name}_combined.npy"), X_combined)
            np.save(os.path.join(out_dir, f"{config_name}_labels.npy"), y_combined)
            
            print(f"    Saved {len(X_combined)} samples to {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        generate_attacks_scenario(scenario)
    else:
        print("Please provide scenario name")
