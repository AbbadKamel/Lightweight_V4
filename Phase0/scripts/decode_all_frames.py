"""
Decode ALL NMEA2000 frames - Full dataset analysis
Process all 299 Frame files (~3 million frames)

HOW TO RUN:
    cd /path/to/Lightweight_IA_V_3
    python3 Phase0/scripts/decode_all_frames.py

OUTPUT:
    - Phase0/results/decoded_frames.csv (all decoded messages)
    - Phase0/results/full_dataset_analysis.json (PGN/signal statistics)
    - Phase0/results/signal_statistics.json (value ranges)
"""
import sys
import os
import glob
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
from tqdm import tqdm
sys.path.append('decode_N2K')
from n2k_decoder import N2KDecoder


def load_and_decode_file(filepath, decoder):
    """Load and decode a single file"""
    frames_processed = 0
    decoded_messages = []
    
    try:
        with open(filepath, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                frames_processed += 1
                parts = line.strip().split('\t')
                
                if len(parts) >= 8:
                    try:
                        frame_data = {
                            'Index': int(parts[0]),
                            'Timestamp': parts[2].strip(),
                            'ID': parts[3].strip(),
                            'Data': parts[7].strip().replace(' ', '')
                        }
                        
                        # Format data for decoder
                        data_formatted = ' '.join([frame_data['Data'][i:i+2] 
                                                   for i in range(0, len(frame_data['Data']), 2)])
                        
                        # Decode
                        signals = decoder.decode_message(frame_data['ID'], data_formatted)
                        
                        if signals:
                            pgn = decoder._extract_pgn(frame_data['ID'])
                            pgn_name = decoder.pgn_handlers.get(pgn, f'Unknown_{pgn}')
                            
                            decoded_messages.append({
                                'timestamp': frame_data['Timestamp'],
                                'pgn': pgn,
                                'pgn_name': pgn_name,
                                'signals': signals
                            })
                    except:
                        continue
    except Exception as e:
        print(f"  ⚠️  Error reading {filepath}: {e}")
    
    return frames_processed, decoded_messages


def main():
    """Main batch processing"""
    
    print("\n" + "="*80)
    print("FULL DATASET DECODING - ALL NMEA2000 FRAMES")
    print("="*80)
    
    # Initialize decoder
    decoder = N2KDecoder()
    print(f"\n✓ Decoder initialized with {len(decoder.pgn_handlers)} PGN definitions")
    
    # Find all frame files and sort numerically by frame range
    frame_files = glob.glob("NMEA2000/Frame*.txt")
    
    # Extract numeric range from filename for proper sorting
    def extract_range(filename):
        import re
        match = re.search(r'Frame\((\d+)', filename)
        return int(match.group(1)) if match else 0
    
    frame_files = sorted(frame_files, key=extract_range)
    print(f"✓ Found {len(frame_files)} frame files")
    
    # Statistics
    total_frames = 0
    total_decoded = 0
    all_pgn_counts = Counter()
    all_signal_counts = Counter()
    signal_values = defaultdict(list)
    pgn_names_map = {}
    
    print(f"\n{'='*80}")
    print(f"Processing files...")
    print(f"{'='*80}\n")
    
    # Process each file
    for i, filepath in enumerate(tqdm(frame_files, desc="Decoding files")):
        frames_count, decoded = load_and_decode_file(filepath, decoder)
        
        total_frames += frames_count
        total_decoded += len(decoded)
        
        # Accumulate statistics
        for msg in decoded:
            all_pgn_counts[msg['pgn']] += 1
            pgn_names_map[msg['pgn']] = msg['pgn_name']
            
            for signal_name, signal_value in msg['signals'].items():
                full_name = f"{msg['pgn_name']}.{signal_name}"
                all_signal_counts[full_name] += 1
                
                # Sample values (don't store all 3M to save memory)
                if len(signal_values[full_name]) < 10000 and signal_value is not None:
                    if isinstance(signal_value, (int, float)) and not np.isnan(signal_value):
                        signal_values[full_name].append(signal_value)
        
        # Progress every 50 files
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(frame_files)} files | "
                  f"Frames: {total_frames:,} | Decoded: {total_decoded:,}")
    
    print(f"\n{'='*80}")
    print(f"DECODING COMPLETE")
    print(f"{'='*80}")
    print(f"Total frames processed: {total_frames:,}")
    print(f"Total messages decoded: {total_decoded:,}")
    print(f"Decode success rate: {(total_decoded/total_frames*100):.2f}%")
    
    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"PGN ANALYSIS - FULL DATASET")
    print(f"{'='*80}")
    print(f"\nUnique PGNs: {len(all_pgn_counts)}")
    print(f"\nAll PGNs sorted by frequency:")
    print(f"{'Rank':<6} {'PGN':<10} {'Name':<35} {'Count':<15} {'Percentage':<10}")
    print(f"{'-'*85}")
    
    for rank, (pgn, count) in enumerate(all_pgn_counts.most_common(), 1):
        name = pgn_names_map.get(pgn, 'Unknown')
        percentage = (count / total_decoded) * 100
        print(f"{rank:<6} {pgn:<10} {name:<35} {count:<15,} {percentage:>6.2f}%")
    
    print(f"\n{'='*80}")
    print(f"SIGNAL ANALYSIS - FULL DATASET")
    print(f"{'='*80}")
    print(f"\nUnique signals: {len(all_signal_counts)}")
    print(f"\nAll signals sorted by frequency:")
    print(f"{'Rank':<6} {'Signal Name':<50} {'Count':<15} {'Percentage':<10}")
    print(f"{'-'*90}")
    
    for rank, (signal, count) in enumerate(all_signal_counts.most_common(), 1):
        percentage = (count / total_decoded) * 100
        print(f"{rank:<6} {signal:<50} {count:<15,} {percentage:>6.2f}%")
    
    print(f"\n{'='*80}")
    print(f"SIGNAL VALUE STATISTICS (sampled)")
    print(f"{'='*80}")
    print(f"{'Signal Name':<50} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'Samples':<10}")
    print(f"{'-'*110}")
    
    for signal, count in all_signal_counts.most_common():
        if signal in signal_values and len(signal_values[signal]) > 0:
            values = signal_values[signal]
            print(f"{signal:<50} {min(values):<12.3f} {max(values):<12.3f} "
                  f"{np.mean(values):<12.3f} {np.std(values):<12.3f} {len(values):<10,}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"SAVING ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    # Save comprehensive report
    report = {
        'total_frames': total_frames,
        'total_decoded': total_decoded,
        'decode_rate': total_decoded / total_frames,
        'unique_pgns': len(all_pgn_counts),
        'unique_signals': len(all_signal_counts),
        'pgn_distribution': {pgn: {'name': pgn_names_map.get(pgn, 'Unknown'), 
                                    'count': count,
                                    'percentage': count/total_decoded*100}
                             for pgn, count in all_pgn_counts.items()},
        'signal_distribution': {signal: {'count': count,
                                         'percentage': count/total_decoded*100}
                                for signal, count in all_signal_counts.items()}
    }
    
    with open('full_dataset_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved: full_dataset_analysis.json")
    
    # Save signal statistics
    signal_stats = {}
    for signal in all_signal_counts.keys():
        if signal in signal_values and len(signal_values[signal]) > 0:
            values = signal_values[signal]
            signal_stats[signal] = {
                'min': float(min(values)),
                'max': float(max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'samples': len(values)
            }
    
    with open('signal_statistics.json', 'w') as f:
        json.dump(signal_stats, f, indent=2)
    print(f"✓ Saved: signal_statistics.json")
    
    # NEW: Save all decoded messages as CSV
    print(f"\n{'='*80}")
    print(f"CREATING CSV FILE")
    print(f"{'='*80}")
    print(f"Re-processing all files to save complete decoded data...")
    
    csv_records = []
    for filepath in tqdm(frame_files, desc="Creating CSV"):
        _, decoded = load_and_decode_file(filepath, decoder)
        for msg in decoded:
            row = {
                'timestamp': msg['timestamp'],
                'pgn': msg['pgn'],
                'pgn_name': msg['pgn_name']
            }
            row.update(msg['signals'])
            csv_records.append(row)
    
    decoded_df = pd.DataFrame(csv_records)
    csv_path = 'Phase0/results/decoded_frames.csv'
    decoded_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path} ({len(decoded_df):,} rows, {len(decoded_df.columns)} columns)")
    
    print(f"\n{'='*80}")
    print(f"FULL ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nKey Results:")
    print(f"  - Processed: {total_frames:,} frames")
    print(f"  - Decoded: {total_decoded:,} messages")
    print(f"  - Found: {len(all_pgn_counts)} unique PGNs")
    print(f"  - Found: {len(all_signal_counts)} unique signals")
    print(f"  - CSV file: {csv_path}")
    print(f"\nNow you can review ALL the data before making any parameter decisions!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
