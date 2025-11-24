#!/usr/bin/env python3
"""
Decode raw CAN frame dumps captured from the NEAC setup.

This script walks through the "Frame brute cantest" directory, reassembles
NMEA 2000 fast-packet payloads, decodes all supported PGNs using the
`N2KDecoder`, and exports two artefacts:

1. results/aggregated_brute_frames.csv  -> Timestamp, CAN ID, PGN, payload hex
2. results/decoded_brute_frames.csv     -> Timestamp + decoded signal columns
"""
import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

# Ensure we can import from src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# Option 3: Use absolute import with src
from src.preprocessing.n2k_decoder import N2KDecoder  # noqa: E402


def iter_frame_files(frame_dir: Path, limit: Optional[int] = None) -> Iterable[Tuple[Path, pd.DataFrame]]:
    """
    Yield (path, dataframe) pairs for every Frame*.txt file inside frame_dir.
    """
    files = sorted(frame_dir.glob("Frame*.txt"))
    if limit is not None:
        files = files[:limit]
    
    for path in files:
        df = pd.read_csv(path, sep="\t", engine="python")
        if "Unnamed: 8" in df.columns:
            df = df.drop(columns=["Unnamed: 8"])
        yield path, df


def main():
    parser = argparse.ArgumentParser(description="Decode raw CAN frame dumps into N2K signals.")
    parser.add_argument(
        "-i", "--input-dir",
        default="Frame brute cantest",
        type=Path,
        help="Directory containing Frame*.txt files (default: Frame brute cantest)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=Path("results"),
        type=Path,
        help="Directory used to store aggregated / decoded CSV outputs (default: results)",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Optional limit on the number of Frame*.txt files to process",
    )
    parser.add_argument(
        "--pgn",
        action="append",
        type=int,
        default=None,
        help="Only decode the provided PGN (can be repeated).",
    )
    parser.add_argument(
        "--supported-only",
        action="store_true",
        help="Skip messages without a decoder handler.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Process only files whose name contains this substring.",
    )
    
    args = parser.parse_args()
    frame_dir: Path = args.input_dir
    
    if not frame_dir.exists() or not frame_dir.is_dir():
        raise SystemExit(f"Input directory not found: {frame_dir}")
    
    decoder = N2KDecoder()
    
    aggregated_records = []
    decoded_records = []
    total_frames = 0
    total_messages = 0
    
    files_iter = iter_frame_files(frame_dir, limit=None if args.pattern else args.file_limit)
    processed_files = 0
    
    for file_path, frame_df in files_iter:
        if args.pattern and args.pattern not in file_path.name:
            continue
        processed_files += 1
        if args.pattern is None and args.file_limit and processed_files > args.file_limit:
            break
        print(f"📥 Processing {file_path.name} ({len(frame_df):,} frames)")
        total_frames += len(frame_df)
        
        for timestamp, can_id, payload in decoder._iter_complete_messages(frame_df):
            pgn = decoder._extract_pgn(can_id)

            if args.pgn and pgn not in args.pgn:
                continue
            if args.supported_only and pgn not in decoder.pgn_handlers:
                continue

            payload_hex = " ".join(f"{byte:02X}" for byte in payload)
            
            aggregated_records.append(
                {
                    "Timestamp": timestamp,
                    "ID": can_id,
                    "PGN": pgn,
                    "Payload": payload_hex,
                }
            )
            
            signals = decoder.decode_message(can_id, payload_hex)
            if signals or not args.supported_only:
                signals["Timestamp"] = timestamp
                decoded_records.append(signals)
            total_messages += 1
    
    if not aggregated_records:
        raise SystemExit("No messages were reconstructed. Check input directory contents.")
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_df = pd.DataFrame(aggregated_records)
    aggregated_path = output_dir / "aggregated_brute_frames.csv"
    aggregated_df.to_csv(aggregated_path, index=False)
    
    decoded_df = pd.DataFrame(decoded_records)
    if not decoded_df.empty:
        decoded_df = decoded_df.ffill()
    decoded_path = output_dir / "decoded_brute_frames.csv"
    decoded_df.to_csv(decoded_path, index=False)
    
    print("\n✅ Decoding complete!")
    print(f"   Frames processed : {total_frames:,}")
    print(f"   Messages decoded : {total_messages:,}")
    print(f"   Aggregated CSV   : {aggregated_path}")
    print(f"   Decoded CSV      : {decoded_path}")
    print("\nℹ️  Note: Manufacturer-specific PGNs (e.g., 653xx range) are logged in the aggregated file.")
    print("    You can correlate those payloads manually or extend N2KDecoder with custom handlers.")


if __name__ == "__main__":
    main()
