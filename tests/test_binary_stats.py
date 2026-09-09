#!/usr/bin/env python
"""
Test script for binary statistics computation.
Usage: python test_binary_stats.py --device 011_08 --process raw --hash <file_hash>
"""
import json
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.parquet_data_manager import ParquetExperimentalDataManager
from src.analysis.entropy_evaluation_pipeline import EntropyEvaluationPipeline


def load_ea_binaries_path():
    """Load the SP800-90B EntropyAssessment binaries path from config.json"""
    config_path = Path(__file__).parents[1] / 'config' / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(
            "Config file not found. Please copy config.template.json to "
            "config.json and fill in the required values."
        )

    with open(config_path) as f:
        config = json.load(f)

    ea_binaries_path = config.get('ea_binaries', {}).get('path')
    if not ea_binaries_path:
        raise ValueError("ea_binaries.path not found in config.json")

    return ea_binaries_path



def main():
    parser = argparse.ArgumentParser(description="Test binary statistics computation")
    parser.add_argument("--device", required=True, help="Device ID (e.g., '011_08')")
    parser.add_argument("--process", default="raw", help="Process type (e.g., 'raw', 'xor02')")
    parser.add_argument("--hash", help="Specific file hash (if not provided, uses first .bin file found)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path=load_ea_binaries_path()
    )
    
    # Construct path to binary file
    bin_dir = Path("bin") / args.device / args.process
    
    if not bin_dir.exists():
        print(f"Error: Directory {bin_dir} does not exist")
        return
    
    # Find binary file
    if args.hash:
        bin_file = bin_dir / f"{args.hash}.bin"
        if not bin_file.exists():
            print(f"Error: File {bin_file} does not exist")
            return
    else:
        # Find first .bin file
        bin_files = list(bin_dir.glob("*.bin"))
        if not bin_files:
            print(f"Error: No .bin files found in {bin_dir}")
            return
        bin_file = bin_files[0]
        print(f"Using first .bin file found: {bin_file.name}")
    
    # Test the binary statistics computation
    print(f"\nComputing binary statistics for: {bin_file}")
    print("="*80)
    
    result = pipeline._get_binary_statistics(
        device_id=args.device,
        process_name=args.process,
        file_hash=bin_file.stem
    )
    
    if result and result.get('mean') is not None:
        print(f"\nResults:")
        print(f"  Mean (P(X=1)): {result['mean']:.6f}")
        print(f"  Variance: {result['variance']:.6f}")
        print(f"\n  Correlation coefficients:")
        for key, corr in result.items():
            if key.startswith('corr_lag_') and corr is not None:
                lag = int(key.removeprefix('corr_lag_'))
                print(f"    Lag {lag:8d}: {corr:+.6f}")
    else:
        print("Failed to compute statistics")

if __name__ == "__main__":
    main()