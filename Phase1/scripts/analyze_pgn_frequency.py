import pandas as pd
import numpy as np

# rudder ===> Simard RF25
# Heading ===> fureno SCX20
# rateofturn ===> fureno SCX20
# attitude ===> We don't know !
# 
# 
# 

def parse_timestamp(ts):
    """Convert timestamp string to seconds since start"""
    try:
        # Format: HH:MM:SS.mmm.0
        ts_clean = ts.replace('.0', '')
        parts = ts_clean.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except:
        return None

print("Loading data...")
df = pd.read_csv('/home/abbad241/Desktop/PhD/Journals_Articles_Papers/Next paper/Lightweight_IA_V_3/Phase0/results/decoded_frames.csv')

# Parse timestamps
df['time_sec'] = df['timestamp'].apply(parse_timestamp)
df = df.dropna(subset=['time_sec'])

# Calculate duration
start_time = df['time_sec'].min()
end_time = df['time_sec'].max()
duration = end_time - start_time

print(f"Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"Total Messages: {len(df)}")
print("-" * 100)
print(f"{'PGN Name (ID)':<35} | {'Interval (ms)':<15} | {'Count':<8} | {'Signals (Columns)'}")
print("-" * 100)

# Group by PGN
pgn_groups = df.groupby(['pgn', 'pgn_name'])

for (pgn, pgn_name), group in pgn_groups:
    count = len(group)
    frequency = count / duration
    interval_ms = (1 / frequency) * 1000 if frequency > 0 else 0
    
    # Find columns that are not null for this PGN
    # We exclude metadata columns
    metadata_cols = ['timestamp', 'pgn', 'pgn_name', 'time_sec']
    potential_signals = [c for c in df.columns if c not in metadata_cols]
    
    active_signals = []
    for col in potential_signals:
        # Check if this column has any non-null values in this group
        if group[col].notna().any():
            active_signals.append(col)
            
    signals_str = ", ".join(active_signals)
    
    print(f"{pgn_name} ({pgn})".ljust(35) + f" | {interval_ms:>10.1f} ms   | {count:>8} | {signals_str}")

print("-" * 100)
