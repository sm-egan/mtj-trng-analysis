#!/usr/bin/env python
"""
Test script to verify binary statistics are included in collect_entropy_results.
Usage: python test_collect_with_stats.py --device 011_08
"""
import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parquet_data_manager import ParquetExperimentalDataManager
from src.analysis.entropy_evaluation_pipeline import EntropyEvaluationPipeline

def load_config():
    """Load configuration from config file"""
    config_path = project_root / 'config' / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(
            "Config file not found. Please copy config.template.json to "
            "config.json and fill in the required values."
        )
    
    with open(config_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Test binary statistics in entropy results collection")
    parser.add_argument("--device", required=True, help="Device ID (e.g., '011_08')")
    parser.add_argument("--process", help="Specific process (e.g., 'raw')")
    parser.add_argument("--output", help="Output CSV file (optional)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    ea_binaries_path = config.get('ea_binaries', {}).get('path')
    if not ea_binaries_path:
        raise ValueError("ea_binaries.path not found in config.json")
    
    # Initialize pipeline
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path=ea_binaries_path
    )
    
    # Test with binary statistics enabled
    print(f"Collecting entropy results with binary statistics for device {args.device}...")
    
    # Custom correlation lags for testing
    test_lags = [1, 2, 4, 8, 16, 32]  # Fewer lags for testing
    
    df = pipeline.collect_entropy_results(
        device_id=args.device,
        process_names=[args.process] if args.process else None,
        include_binary_stats=True,
        correlation_lags=test_lags
    )
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Display first few rows with key columns
    print(f"\nFound {len(df)} results")
    print("\nFirst few rows with binary statistics:")
    
    # Select columns to display
    display_cols = ['device', 'processing', 'switching_probability', 
                   'mean', 'variance', 'corr_lag_1', 'corr_lag_2', 'corr_lag_4']
    display_cols = [col for col in display_cols if col in df.columns]
    
    print(df[display_cols].head())
    
    # Check if binary stats were computed
    if 'mean' in df.columns:
        non_null_means = df['mean'].notna().sum()
        print(f"\nBinary statistics computed for {non_null_means}/{len(df)} files")
        
        # Compare mean and switching_probability (should be equal)
        if 'switching_probability' in df.columns:
            df_with_both = df[df['mean'].notna() & df['switching_probability'].notna()]
            if len(df_with_both) > 0:
                diff = (df_with_both['mean'] - df_with_both['switching_probability']).abs()
                print(f"\nMean vs switching_probability difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
                if diff.max() < 0.001:
                    print("✓ Mean and switching_probability match!")
    
    # Save if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")
    
    # Test with binary statistics disabled
    print("\n" + "="*80)
    print("Testing with binary statistics disabled...")
    df_no_stats = pipeline.collect_entropy_results(
        device_id=args.device,
        process_names=[args.process] if args.process else None,
        include_binary_stats=False
    )
    
    print(f"Columns without binary stats: {len(df_no_stats.columns)}")
    print(f"Columns with binary stats: {len(df.columns)}")
    print(f"Difference: {len(df.columns) - len(df_no_stats.columns)} columns")

if __name__ == "__main__":
    main()