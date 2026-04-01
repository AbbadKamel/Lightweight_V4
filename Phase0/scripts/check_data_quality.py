import pandas as pd
import numpy as np
import os

def check_data_quality():
    print("="*80)
    print("DATA QUALITY CHECK - Phase 0")
    print("="*80)

    file_path = 'Phase0/results/decoded_frames_realtime.csv'
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime for analysis
    # Assuming today's date for calculation purposes
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    
    # 1. OVERVIEW
    n_rows = len(df)
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()
    duration = end_time - start_time
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"  - Total Rows: {n_rows:,}")
    print(f"  - Start Time: {start_time.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"  - End Time:   {end_time.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"  - Duration:   {duration}")

    # 2. SAMPLING RATE ANALYSIS
    print(f"\n⏱️ SAMPLING RATE (Time between updates)")
    time_diffs = df['datetime'].diff().dropna().dt.total_seconds()
    
    mean_diff = time_diffs.mean()
    median_diff = time_diffs.median()
    max_diff = time_diffs.max()
    min_diff = time_diffs.min()
    
    print(f"  - Mean Interval:   {mean_diff*1000:.2f} ms")
    print(f"  - Median Interval: {median_diff*1000:.2f} ms")
    print(f"  - Min Interval:    {min_diff*1000:.2f} ms")
    print(f"  - Max Gap:         {max_diff:.4f} s")
    
    # Check for large gaps (> 1 second)
    gaps = time_diffs[time_diffs > 1.0]
    if len(gaps) > 0:
        print(f"  ⚠️  Found {len(gaps)} gaps larger than 1 second!")
        print(f"      Largest gap: {gaps.max():.2f}s")
    else:
        print(f"  ✓ No gaps larger than 1 second.")

    # 3. COMPLETENESS (NaN Analysis)
    print(f"\n📉 COMPLETENESS (After Forward-Fill)")
    print(f"  (Note: NaNs here mean the sensor hadn't reported yet at the start of the file)")
    
    key_columns = ['latitude', 'longitude', 'sog', 'heading', 'depth', 'wind_speed', 'rudder_position']
    
    print(f"  {'Column':<20} {'Missing':<10} {'% Complete':<10}")
    print(f"  {'-'*45}")
    
    for col in key_columns:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct_complete = 100 * (1 - n_missing / n_rows)
            print(f"  {col:<20} {n_missing:<10,} {pct_complete:>9.2f}%")
        else:
            print(f"  {col:<20} NOT FOUND")

    # 4. PHYSICAL VALIDITY CHECKS
    print(f"\nphysics PHYSICAL VALIDITY CHECKS")
    
    def check_range(col, min_val, max_val, unit=""):
        if col not in df.columns: return
        vals = df[col].dropna()
        if len(vals) == 0: return
        
        curr_min = vals.min()
        curr_max = vals.max()
        outliers = vals[(vals < min_val) | (vals > max_val)]
        
        status = "✓ OK" if len(outliers) == 0 else f"⚠️ {len(outliers)} outliers"
        print(f"  - {col:<15} Range: [{curr_min:.2f}, {curr_max:.2f}] {unit} | Expect: [{min_val}, {max_val}] | {status}")

    check_range('sog', 0, 50, "knots")
    check_range('heading', 0, 360, "deg")
    check_range('wind_speed', 0, 100, "m/s")
    check_range('depth', 0, 1000, "m")
    check_range('rudder_position', -90, 90, "deg")
    check_range('pitch', -45, 45, "deg")
    check_range('roll', -45, 45, "deg")
    
    # Position check
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat_outliers = df[(df['latitude'] < -90) | (df['latitude'] > 90)]
        lon_outliers = df[(df['longitude'] < -180) | (df['longitude'] > 180)]
        if len(lat_outliers) + len(lon_outliers) == 0:
            print(f"  - Position        ✓ Valid Lat/Lon ranges")
        else:
            print(f"  ⚠️  Found {len(lat_outliers)} invalid Lats and {len(lon_outliers)} invalid Lons")

    # 5. STATIONARITY
    print(f"\n⚓ MOVEMENT ANALYSIS")
    if 'sog' in df.columns:
        stopped = df[df['sog'] < 0.1]
        moving = df[df['sog'] >= 0.1]
        pct_stopped = len(stopped) / n_rows * 100
        print(f"  - Time Stopped (<0.1 kn): {len(stopped):,} rows ({pct_stopped:.1f}%)")
        print(f"  - Time Moving:            {len(moving):,} rows ({100-pct_stopped:.1f}%)")
        if len(moving) > 0:
            print(f"  - Max Speed:              {moving['sog'].max():.2f} knots")
            print(f"  - Avg Speed (moving):     {moving['sog'].mean():.2f} knots")

    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

if __name__ == "__main__":
    check_data_quality()
