# Script: scripts/analysis/extract_entropy_results.py
#!/usr/bin/env python
"""
Extract entropy test results to CSV format.
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parents[1]
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
    parser = argparse.ArgumentParser(description="Extract entropy test results to CSV")
    parser.add_argument("--device", help="Specific device ID (e.g., '07_08')")
    parser.add_argument("--processes", nargs="+", help="Specific processes (e.g., raw xor02)")
    parser.add_argument("--filename", help="Output CSV filename (default: entropy_results_YYYYMMDD.csv)")

    lags_group = parser.add_mutually_exclusive_group()
    lags_group.add_argument("--lags", nargs="+", type=int,
                            help="Explicit correlation lags (e.g., --lags 1 2 4 8 16)")
    lags_group.add_argument("--lags-expr", dest="lags_expr",
                            help="NumPy expression for lags, evaluated with np available "
                                 "(e.g., --lags-expr \"np.arange(0, 21)\" or "
                                 "\"(2**np.arange(0,21)).tolist()\")")

    args = parser.parse_args()

    # Resolve correlation lags
    correlation_lags = None
    if args.lags:
        correlation_lags = args.lags
    elif args.lags_expr:
        try:
            result = eval(args.lags_expr, {"np": np, "__builtins__": {}})
            correlation_lags = [int(x) for x in np.asarray(result).ravel()]
        except Exception as e:
            parser.error(f"Could not evaluate --lags-expr: {e}")
    
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
    
    # Default output filename with date
    if not args.filename:
        date_str = datetime.now().strftime("%Y%m%d")
        output_file = project_root / 'esv' / f"entropy_results_{date_str}.csv"
    else:
        output_file = args.filename
    
    print(f"Extracting entropy results...")
    print(f"Device: {args.device if args.device else 'All devices'}")
    print(f"Processes: {args.processes if args.processes else 'All processes'}")
    print(f"Correlation lags: {correlation_lags if correlation_lags else 'Default (powers of 2, 2^0 to 2^20)'}")
    print(f"Output file: {output_file}")

    # Collect results
    df = pipeline.collect_entropy_results(
        device_id=args.device,
        process_names=args.processes,
        output_csv=output_file,
        correlation_lags=correlation_lags
    )
    
    print(f"\nExtracted {len(df)} results to {output_file}")
    
    # Show summary
    if len(df) > 0:
        print("\nSummary by device:")
        print(df['device'].value_counts())
        
        print("\nSummary by processing type:")
        print(df['processing'].value_counts())
        
        # Count pass/fail
        if 'iid_pass' in df.columns:
            print(f"\nIID test results:")
            print(f"  Passed: {df['iid_pass'].sum()}")
            print(f"  Failed: {(df['iid_pass'] == False).sum()}")
        
        if 'restart_pass' in df.columns and df['restart_pass'].notna().any():
            print(f"\nRestart test results:")
            print(f"  Passed: {df['restart_pass'].sum()}")
            print(f"  Failed: {(df['restart_pass'] == False).sum()}")


if __name__ == "__main__":
    main()