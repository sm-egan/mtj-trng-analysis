#!/usr/bin/env python
"""
Test script for Step 2 of restart test implementation.
Tests the restart test planning functionality.
"""
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from src.parquet_data_manager import ParquetExperimentalDataManager
from src.analysis.entropy_evaluation_pipeline import EntropyEvaluationPipeline


def print_restart_test_plan(tasks):
    """
    Pretty-print the restart test plan for verification.
    
    Args:
        tasks: List of restart test tasks from create_restart_test_plan()
    """
    if not tasks:
        print("No restart tests to run.")
        return
    
    # Group by device and process for cleaner output
    grouped = defaultdict(lambda: defaultdict(list))
    
    for task in tasks:
        grouped[task['device_id']][task['process_name']].append(task)
    
    print(f"\nRestart Test Plan: {len(tasks)} total tests")
    print("=" * 80)
    
    for device_id in sorted(grouped.keys()):
        print(f"\nDevice: {device_id}")
        for process_name in sorted(grouped[device_id].keys()):
            print(f"  Process: {process_name}")
            for task in grouped[device_id][process_name]:
                print(f"    {task['file_hash'][:12]}...")
                print(f"      Test: ea_restart {task['test_flag']} [file] 1 {task['h_original']:.6f}")
                print(f"      Files: {task['num_restart_files']} restart files")
                print(f"      Source: {task['source_test']} test results")
    
    # Summary
    iid_count = sum(1 for t in tasks if t['test_flag'] == '-i')
    non_iid_count = sum(1 for t in tasks if t['test_flag'] == '-n')
    
    print(f"\nSummary:")
    print(f"  IID restart tests: {iid_count}")
    print(f"  Non-IID restart tests: {non_iid_count}")
    print(f"  Total restart files to process: {sum(t['num_restart_files'] for t in tasks)}")

def load_config():
    import json
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
    # Load config
    config = load_config()
    
    # Get EA binaries path from config
    ea_binaries_path = config.get('ea_binaries', {}).get('path')
    if not ea_binaries_path:
        raise ValueError("ea_binaries.path not found in config.json")

    # Initialize pipeline
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path=ea_binaries_path
    )
    
    print("STEP 2 TEST: Creating restart test execution plan")
    print("=" * 80)
    
    # Test 1: Create plan for all devices
    print("\nTest 1: All devices and processes")
    all_tasks = pipeline.create_restart_test_plan()
    print(f"Found {len(all_tasks)} total restart tests across all devices")
    
    # Test 2: Create plan for specific device
    print("\n\nTest 2: Specific device (011_08)")
    device_tasks = pipeline.create_restart_test_plan(device_id="011_08")
    print_restart_test_plan(device_tasks)
    
    # Test 3: Specific process
    print("\n\nTest 3: Specific device and process (011_08/raw)")
    process_tasks = pipeline.create_restart_test_plan(
        device_id="011_08",
        process_names="raw"
    )
    print(f"Found {len(process_tasks)} restart tests for 011_08/raw")
    
    # Test 4: Specific hash (if we have at least one task)
    if device_tasks:
        first_hash = device_tasks[0]['file_hash']
        print(f"\n\nTest 4: Specific hash ({first_hash[:12]}...)")
        hash_tasks = pipeline.create_restart_test_plan(
            device_id="011_08",
            file_hash=first_hash
        )
        print_restart_test_plan(hash_tasks)
    
    # Test 5: Verify task structure
    if device_tasks:
        print("\n\nTest 5: Sample task structure")
        sample_task = device_tasks[0]
        print("Sample task keys:", list(sample_task.keys()))
        print("Sample task:")
        for key, value in sample_task.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()