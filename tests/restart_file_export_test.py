from pathlib import Path

if __name__ == "__main__":
    # Add project root to Python path
    project_root = Path.cwd().parent  # Assumes notebook is in notebooks/
    sys.path.insert(0, str(project_root))
    
    from src.parquet_data_manager import ParquetExperimentalDataManager
    
    # Test the create_restart_samples method
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path="/Users/shannonegan/Documents/DSV/StochastIQ/umn/SP800-90B_EntropyAssessment/cpp"
    )
    
    # Test on one device and one process type
    print("Testing create_restart_samples method...")
    
    # Test 1: Process all files for a device/process
    print("\n1. Testing with device and process:")
    summary = pipeline.create_restart_samples(
        device_id="07_08",
        process_names=["raw"],
        skip_existing=False  # Force recreation for testing
    )
    
    # Print summary
    print("\nSummary of created restart files:")
    for device, processes in summary.items():
        print(f"\nDevice: {device}")
        for process, files in processes.items():
            print(f"  Process: {process}")
            for file_hash, num_created in files.items():
                print(f"    {file_hash[:12]}...: {num_created} restart files")
    
    # Test 2: Process a specific hash with specific process
    print("\n2. Testing with specific hash and process:")
    if summary.get("07_08", {}).get("raw", {}):
        first_hash = list(summary["07_08"]["raw"].keys())[0]
        print(f"Processing hash: {first_hash[:12]}...")
        
        summary2 = pipeline.create_restart_samples(
            device_id="07_08",
            process_names="xor2",  # Process for xor2 version
            file_hash=first_hash,
            skip_existing=False
        )
        if summary2.get("07_08", {}).get("xor2", {}).get(first_hash):
            print(f"Created {summary2['07_08']['xor2'][first_hash]} restart files for xor2")
    
    # Test 3: Process a specific hash across ALL processes
    print("\n3. Testing with specific hash, searching all processes:")
    if summary.get("07_08", {}).get("raw", {}):
        first_hash = list(summary["07_08"]["raw"].keys())[0]
        print(f"Processing hash: {first_hash[:12]}... across all processes")
        
        summary3 = pipeline.create_restart_samples(
            device_id="07_08",
            file_hash=first_hash,  # No process_names specified - will find all
            skip_existing=False
        )
        
        print("Found and processed in:")
        for process, files in summary3.get("07_08", {}).items():
            if first_hash in files:
                print(f"  - {process}: {files[first_hash]} restart files")
    
    # Verify the files were created correctly
    print("\nVerifying created files...")
    import os
    
    restart_dir = Path("bin/07_08/raw/restart")
    if restart_dir.exists():
        for hash_dir in restart_dir.iterdir():
            if hash_dir.is_dir():
                print(f"\n{hash_dir.name}:")
                for restart_file in sorted(hash_dir.glob("*.bin")):
                    size = os.path.getsize(restart_file)
                    print(f"  {restart_file.name}: {size:,} bytes (should be 1,000,000)")