import pandas as pd
import numpy as np
import json
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'Phase0', 'results', 'decoded_frames.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
SIGNAL_ORDER_FILE = os.path.join(OUTPUT_DIR, 'optimal_signal_order.json')
OUTPUT_TABLE = os.path.join(OUTPUT_DIR, 'master_table_final.csv')
SCALER_PARAMS_FILE = os.path.join(OUTPUT_DIR, 'scaler_params.json')

print("="*80)
print("CREATING FINAL MASTER TABLE (15 SIGNALS)")
print("="*80)

print("\n1. Loading optimal signal order...")
with open(SIGNAL_ORDER_FILE, 'r') as f:
    order_data = json.load(f)
    OPTIMAL_ORDER = order_data['clustered_order']

print(f"   Using {len(OPTIMAL_ORDER)} signals (speed_water removed):")
for i, sig in enumerate(OPTIMAL_ORDER, 1):
    print(f"   {i:2d}. {sig}")

# Load raw data
print("\n2. Loading raw dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"   Loaded {len(df):,} messages")

# Clean timestamps
print("\n3. Cleaning timestamps...")
df['timestamp'] = df['timestamp'].astype(str).str.replace(r'\.0$', '', regex=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
df.set_index('timestamp', inplace=True)
print(f"   Time range: {df.index.min()} to {df.index.max()}")
duration_seconds = (df.index.max() - df.index.min()).total_seconds()
print(f"   Duration: {duration_seconds:.0f} seconds ({duration_seconds/60:.1f} minutes)")

# Filter to selected signals only
print("\n4. Filtering to 15 selected signals...")
df_selected = df[OPTIMAL_ORDER]

# Resample to 1-second buckets
print("\n5. Resampling to 1-second buckets (Mean, Max, Min, Std)...")

# Create aggregation dictionary
agg_dict = {}
for signal in OPTIMAL_ORDER:
    agg_dict[f'{signal}_mean'] = (signal, 'mean')
    agg_dict[f'{signal}_max'] = (signal, 'max')
    agg_dict[f'{signal}_min'] = (signal, 'min')
    agg_dict[f'{signal}_std'] = (signal, 'std')

# Perform resampling
df_resampled = df_selected.resample('1s').agg(**agg_dict)

print(f"   Created {len(df_resampled):,} rows")
print(f"   Created {len(df_resampled.columns)} columns (15 signals × 4 features)")

# Reorder columns
print("\n6. Reordering columns by clustered order...")
ordered_columns = []
for signal in OPTIMAL_ORDER:
    ordered_columns.extend([
        f'{signal}_mean',
        f'{signal}_max',
        f'{signal}_min',
        f'{signal}_std'
    ])

df_final = df_resampled[ordered_columns]

# Check for NaN
print("\n7. Checking for NaN values...")
nan_count = df_final.isna().sum().sum()
if nan_count > 0:
    print(f"   WARNING: {nan_count} NaN values detected")
    # Fill NaN with forward-fill (limited to 2 seconds)
    print("   Applying forward-fill (max 2 seconds)...")
    df_final = df_final.fillna(method='ffill', limit=2)
    
    # Check remaining NaN
    remaining_nan = df_final.isna().sum().sum()
    if remaining_nan > 0:
        print(f"   {remaining_nan} NaN remaining after forward-fill")
        # Fill remaining with column mean
        print("   Filling remaining NaN with column mean...")
        df_final = df_final.fillna(df_final.mean())
else:
    print("   ✓ No NaN values - data is complete!")

# Normalize to [0, 1] using Min-Max scaling
print("\n8. Normalizing to [0, 1] using Min-Max scaling...")
scaler_params = {}

for col in df_final.columns:
    min_val = df_final[col].min()
    max_val = df_final[col].max()
    
    # Save scaler parameters
    scaler_params[col] = {
        'min': float(min_val),
        'max': float(max_val)
    }
    
    # Apply scaling
    if max_val > min_val:
        df_final[col] = (df_final[col] - min_val) / (max_val - min_val)
    else:
        # Constant column (shouldn't happen, but handle it)
        df_final[col] = 0.5

print("   ✓ All values scaled to [0, 1]")

# Verify normalization
print("\n9. Verifying normalization...")
overall_min = df_final.min().min()
overall_max = df_final.max().max()
print(f"   Global min: {overall_min:.6f}")
print(f"   Global max: {overall_max:.6f}")

if overall_min < 0 or overall_max > 1:
    print("   WARNING: Values outside [0, 1] range!")
else:
    print("   ✓ All values within [0, 1]")

# Save final table
print("\n10. Saving final master table...")
df_final.to_csv(OUTPUT_TABLE)
print(f"   ✓ Saved: {OUTPUT_TABLE}")

# Save scaler parameters
with open(SCALER_PARAMS_FILE, 'w') as f:
    json.dump(scaler_params, f, indent=2)
print(f"   ✓ Saved: {SCALER_PARAMS_FILE}")

# Save final statistics
stats = {
    "total_rows": int(len(df_final)),
    "total_columns": int(len(df_final.columns)),
    "total_signals": len(OPTIMAL_ORDER),
    "features_per_signal": 4,
    "duration_seconds": float(duration_seconds),
    "signals": OPTIMAL_ORDER,
    "removed_signals": ["speed_water"],
    "normalization": "Min-Max scaling to [0, 1]",
    "value_range": {
        "min": float(overall_min),
        "max": float(overall_max)
    }
}

stats_file = os.path.join(OUTPUT_DIR, 'master_table_final_stats.json')
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"   ✓ Saved: {stats_file}")

print("\n" + "="*80)
print("FINAL MASTER TABLE COMPLETE!")
print("="*80)
print(f"\nOutput: {OUTPUT_TABLE}")
print(f"Shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
print(f"Signals: {len(OPTIMAL_ORDER)}")
print(f"Features per signal: 4 (mean, max, min, std)")
print(f"Total features: {len(OPTIMAL_ORDER) * 4}")
print(f"Value range: [{overall_min:.6f}, {overall_max:.6f}]")
print("\n✓ Ready for CNN training!")
print("="*80)
