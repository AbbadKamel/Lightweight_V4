
import pandas as pd
import numpy as np
import json
import os
import argparse
import sys

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # Up 2 levels from Phase1/scripts

INPUT_FILE = os.path.join(PROJECT_ROOT, 'Phase0', 'results', 'decoded_frames.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results') # Phase1/results
SIGNAL_ORDER_FILE = os.path.join(OUTPUT_DIR, 'optimal_signal_order.json')

def create_scenario_dataset(scenario_name):
    print("="*80)
    print(f"CREATING MASTER TABLE FOR SCENARIO: {scenario_name}")
    print("="*80)

    # Define features based on scenario
    if scenario_name == 'A_baseline':
        aggregations = ['mean', 'max', 'min', 'std']
    elif scenario_name == 'B_mean_only':
        aggregations = ['mean']
    elif scenario_name == 'C_mean_std':
        aggregations = ['mean', 'std']
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    print(f"Selected aggregations: {aggregations}")

    # Load optimal order
    with open(SIGNAL_ORDER_FILE, 'r') as f:
        order_data = json.load(f)
        OPTIMAL_ORDER = order_data['clustered_order']

    # Load raw data
    print("Loading raw dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Clean timestamps
    df['timestamp'] = df['timestamp'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    df.set_index('timestamp', inplace=True)

    # Filter signals
    df_selected = df[OPTIMAL_ORDER]

    # Resample
    print("Resampling...")
    agg_dict = {}
    for signal in OPTIMAL_ORDER:
        for agg in aggregations:
            agg_dict[f'{signal}_{agg}'] = (signal, agg)

    df_resampled = df_selected.resample('1s').agg(**agg_dict)

    # Reorder columns
    ordered_columns = []
    for signal in OPTIMAL_ORDER:
        for agg in aggregations:
            ordered_columns.append(f'{signal}_{agg}')

    df_final = df_resampled[ordered_columns]

    # Handle NaN
    df_final = df_final.fillna(method='ffill', limit=2)
    df_final = df_final.fillna(df_final.mean())

    # Normalize
    print("Normalizing...")
    scaler_params = {}
    for col in df_final.columns:
        min_val = df_final[col].min()
        max_val = df_final[col].max()
        
        if max_val == min_val:
            df_final[col] = 0.0
        else:
            df_final[col] = (df_final[col] - min_val) / (max_val - min_val)
            
        scaler_params[col] = {'min': float(min_val), 'max': float(max_val)}

    # Save
    output_file = os.path.join(OUTPUT_DIR, f'master_table_{scenario_name}.csv')
    df_final.to_csv(output_file)
    print(f"Saved dataset to: {output_file}")
    
    # Save scaler params
    scaler_file = os.path.join(OUTPUT_DIR, f'scaler_params_{scenario_name}.json')
    with open(scaler_file, 'w') as f:
        json.dump(scaler_params, f, indent=4)

    # Update Config Info (Save to a separate file that config.py can read or we pass it along)
    config_info = {
        'num_features': len(df_final.columns),
        'feature_names': list(df_final.columns),
        'dataset_path': output_file
    }
    config_file = os.path.join(OUTPUT_DIR, f'config_{scenario_name}.json')
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=4)
        
    return config_info

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    else:
        scenario = 'B_mean_only' # Default to the new one
    
    create_scenario_dataset(scenario)
