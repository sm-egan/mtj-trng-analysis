# Test script for Step 3
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    
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
    
    print("STEP 3 TEST: Running restart tests")
    print("=" * 80)
    
    # Test 1: Run restart tests for a specific hash
    # First, create a plan to get a hash
    tasks = pipeline.create_restart_test_plan(device_id="011_08", process_names=["xor2", "xor4", "xor8"])
    
    if tasks:
        # Take just the first task for testing
        first_task = tasks[0]
        print(f"\nTest 1: Running restart test for single hash")
        print(f"  Device: {first_task['device_id']}")
        print(f"  Process: {first_task['process_name']}")
        print(f"  Hash: {first_task['file_hash'][:12]}...")
        print(f"  Number of restart files: {first_task['num_restart_files']}")
        
        # Run just this one task
        summary = pipeline.run_restart_tests(tasks=[first_task], skip_existing=False)
        
        print(f"\nResults:")
        print(f"  Completed: {summary['completed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Skipped: {summary['skipped']}")
    
    # Test 2: Run all restart tests for a device/process combination
    print(f"\n\nTest 2: Running all restart tests for 011_08/raw")
    summary = pipeline.run_restart_tests(
        device_id="011_08",
        process_names=["xor2", "xor4", "xor8"],
        skip_existing=True  # Skip ones we just ran
    )
    
    print(f"\nFinal summary:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Skipped: {summary['skipped']}")