import pandas as pd
import numpy as np

print("Loading raw data...")
df = pd.read_csv('Phase0/results/decoded_frames.csv')

# 1. Convert timestamp to datetime object for precise handling
# The format is HH:MM:SS.mmm.0
def parse_real_time(ts):
    try:
        # Remove the trailing .0 if it exists
        clean_ts = str(ts)
        if clean_ts.endswith('.0'):
            clean_ts = clean_ts[:-2]
        return pd.to_datetime(clean_ts, format='%H:%M:%S.%f').time()
    except:
        return None

print("Parsing timestamps...")
# We'll use a full datetime for sorting/diffs, assuming current date (doesn't matter for relative time)
df['datetime'] = pd.to_datetime(df['timestamp'].astype(str).str.replace(r'\.0$', '', regex=True), format='%H:%M:%S.%f', errors='coerce')

# Drop rows with invalid time
df = df.dropna(subset=['datetime'])

# Sort by time to be sure
df = df.sort_values('datetime')

print("Pivoting data...")

# 2. The Strategy: "Forward Fill" (Propagation)
# We want one row per unique timestamp.
# If at 15:05:30.021 we receive Position, we keep the last known Wind, Depth, Speed, etc.

# First, we need to identify what columns belong to what data type
# We'll create a wide format where every unique timestamp is a row
# and every possible sensor value is a column.

# Get all unique sensor columns (excluding metadata like pgn, pgn_name, timestamp)
sensor_cols = [c for c in df.columns if c not in ['timestamp', 'pgn', 'pgn_name', 'datetime']]

# Create a new dataframe with just timestamp and all sensor columns
# We group by timestamp and take the 'first' non-null value for each column at that exact instant
# (In case multiple PGNs arrive at the EXACT same millisecond)
df_wide = df.groupby('datetime')[sensor_cols].first()

# Now the magic: Forward Fill
# This propagates the last valid observation forward to the next timestamp
print("Aligning sensors (Forward Fill)...")
df_aligned = df_wide.ffill()

# Reset index to make datetime a column again
df_aligned = df_aligned.reset_index()

# Format the timestamp back to string if you prefer, or keep as object
df_aligned['timestamp'] = df_aligned['datetime'].dt.strftime('%H:%M:%S.%f').str[:-3] # Keep milliseconds

# Reorder columns: Timestamp first
cols = ['timestamp'] + [c for c in df_aligned.columns if c != 'timestamp' and c != 'datetime']
df_final = df_aligned[cols]

# Filter to keep only rows where we actually have a Position update?
# Or keep ALL rows (high frequency)?
# User asked for "reprend le meme temps reel", so we keep all unique timestamps.
# However, to make it readable, maybe we only want rows where SOMETHING changed?
# The ffill() makes it so every row has values.

print(f"Generated {len(df_final)} aligned rows.")
print("Saving to Phase0/results/decoded_frames_realtime.csv...")
df_final.to_csv('Phase0/results/decoded_frames_realtime.csv', index=False)
print("Done!")
