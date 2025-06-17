# entropy_evaluation_pipeline.py
import subprocess
import hashlib
import numpy as np
import sqlite3
from pathlib import Path
import polars as pl
from datetime import datetime
import logging
from typing import List, Dict, Optional, Union
from src.analysis.processing import xorn_whitening

class EntropyEvaluationPipeline:
    """Pipeline for running NIST SP 800-90B entropy evaluation tests"""
    
    def __init__(self, data_manager, ea_binaries_path, 
                 output_base_dir='bin', results_base_dir='esv'):
        """
        Initialize the pipeline
        
        Args:
            data_manager: ParquetExperimentalDataManager instance
            ea_binaries_path: Path to directory containing ea_iid and ea_non_iid
            output_base_dir: Base directory for binary files
            results_base_dir: Base directory for test results
        """
        self.dm = data_manager
        self.ea_path = Path(ea_binaries_path)
        self.ea_iid = self.ea_path / 'ea_iid'
        self.ea_non_iid = self.ea_path / 'ea_non_iid'
        
        # Verify binaries exist
        if not self.ea_iid.exists():
            raise FileNotFoundError(f"ea_iid not found at {self.ea_iid}")
        if not self.ea_non_iid.exists():
            raise FileNotFoundError(f"ea_non_iid not found at {self.ea_non_iid}")
            
        self.output_base = Path(output_base_dir)
        self.results_base = Path(results_base_dir)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_runs(self, 
                     polars_expr: Optional[pl.Expr] = None,
                     run_ids: Optional[List[str]] = None, 
                     criteria: Optional[Dict] = None,
                     xor_levels: List[int] = [1, 2, 4, 8],
                     skip_existing: bool = True,
                     run_tests: List[str] = ['iid', 'non_iid']):
        """
        Main pipeline: fetch data → export binaries → run tests → save results
        
        Args:
            polars_expr: Polars expression for filtering runs (e.g., pl.col("device_id") == "07_08")
            run_ids: List of specific run IDs to process
            criteria: Dict of criteria for get_runs_by_criteria
            xor_levels: List of XOR whitening levels (1 means raw)
            skip_existing: Skip if output files already exist
            run_tests: Which tests to run ('iid', 'non_iid')
            
        Examples:
            # Process specific device with certain parameters
            pipeline.process_runs(
                polars_expr=(pl.col("device_id") == "07_08") & (pl.col("field") == 10.0)
            )
            
            # Process multiple devices
            pipeline.process_runs(
                polars_expr=pl.col("device_id").is_in(["07_08", "23_07"])
            )
        """
        # Fetch and group data by device
        device_data = self._fetch_and_group_data(polars_expr, run_ids, criteria)
        
        if not device_data:
            self.logger.warning("No data found matching criteria")
            return
        
        # Process each device
        total_devices = len(device_data)
        for idx, (device_id, (binary_data, raw_hash)) in enumerate(device_data.items(), 1):
            self.logger.info(f"Processing device {device_id} ({idx}/{total_devices}) with {len(binary_data):,} samples")
            self._process_device(device_id, binary_data, raw_hash, xor_levels, 
                               skip_existing, run_tests)
    
    def _fetch_and_group_data(self, polars_expr, run_ids, criteria):
        """
        Fetch data and convert to binary format, grouped by device
        
        Returns:
            Dict[device_id, (binary_array, raw_hash)]
        """
        device_data = {}
        
        # Determine which runs to fetch
        if run_ids:
            # Specific run IDs provided
            runs_to_fetch = run_ids
            self.logger.info(f"Using {len(runs_to_fetch)} specified run IDs")
            
        elif polars_expr is not None:
            # Use Polars expression to filter runs
            self.logger.info("Applying Polars expression to filter runs")
            
            # Get all runs as a Polars DataFrame
            with sqlite3.connect(self.dm.db_path) as conn:
                runs_df = pl.read_database("SELECT * FROM runs", conn)
            
            # Apply the Polars expression to filter
            filtered_df = runs_df.filter(polars_expr)
            
            # Extract run_ids from filtered DataFrame
            runs_to_fetch = filtered_df['run_id'].to_list()
            
            self.logger.info(f"Polars expression filtered {len(runs_df)} runs to {len(runs_to_fetch)} runs")
            
            # Log some info about what was selected
            if len(runs_to_fetch) > 0:
                devices_selected = filtered_df['device_id'].unique().to_list()
                self.logger.info(f"Selected devices: {devices_selected}")
                
        elif criteria:
            # Use criteria dict
            runs_to_fetch = self.dm.get_runs_by_criteria(criteria)
            self.logger.info(f"Criteria matched {len(runs_to_fetch)} runs")
            
        else:
            self.logger.error("Must provide run_ids, criteria, or polars_expr")
            return {}
        
        if not runs_to_fetch:
            self.logger.warning("No runs found matching criteria")
            return {}
        
        # Group runs by device using the database
        with sqlite3.connect(self.dm.db_path) as conn:
            placeholders = ', '.join(['?'] * len(runs_to_fetch))
            query = f"""
            SELECT run_id, device_id 
            FROM runs 
            WHERE run_id IN ({placeholders})
            """
            
            cursor = conn.execute(query, runs_to_fetch)
            
            runs_by_device = {}
            for run_id, device_id in cursor:
                if device_id not in runs_by_device:
                    runs_by_device[device_id] = []
                runs_by_device[device_id].append(run_id)
        
        self.logger.info(f"Processing {len(runs_by_device)} devices")
        
        # Process each device's runs
        for device_id, device_runs in runs_by_device.items():
            self.logger.info(f"Fetching {len(device_runs)} runs for device {device_id}")
            
            # Get measurements
            measurements_df = self.dm.get_measurements(device_runs)
            if measurements_df.shape[0] == 0:
                self.logger.warning(f"No measurements found for device {device_id}")
                continue
            
            # Stack measurements
            stacked_df = self.dm.stack_measurements(measurements_df)
            
            # Get threshold
            device_info = self.dm.get_device_info(device_id)
            if device_info:
                threshold = device_info[2]  # threshold is 3rd column
                self.logger.info(f"Using stored threshold {threshold:.6f} for device {device_id}")
            else:
                self.logger.warning(f"No threshold found for device {device_id}, calculating...")
                from src.analysis.processing import find_p_ap_states
                threshold, _, _ = find_p_ap_states(stacked_df)
                self.logger.info(f"Calculated threshold: {threshold:.6f}")
            
            # Convert to binary. Note this assumes A_ -> P switching
            binary_data = (stacked_df['device_voltage'] < threshold).to_numpy().astype(np.uint8)
            
            # Calculate hash of raw binary data
            raw_hash = hashlib.sha256(binary_data.tobytes()).hexdigest()

            # Calculate switching probability of raw binary data
            sp = np.average(binary_data)
            
            device_data[device_id] = (binary_data, raw_hash, sp)
            self.logger.info(f"Device {device_id}: {len(binary_data):,} binary values, hash: {raw_hash[:16]}..., p = {sp:.5f}")
        
        return device_data
    
    def _process_device(self, device_id, binary_data, raw_hash, xor_levels, 
                       skip_existing, run_tests):
        """Process all XOR levels for a single device"""
        
        for xor_n in xor_levels:
            # Apply XOR whitening
            if xor_n == 1:
                processed_data = binary_data
                process_name = "raw"
            else:
                self.logger.info(f"Applying XOR-{xor_n} whitening")
                processed_data = xorn_whitening(binary_data, n=xor_n)
                process_name = f"xor{xor_n}"
            
            self.logger.info(f"Processing {process_name} for device {device_id} ({len(processed_data):,} samples)")
            
            # Create directory structure
            bin_dir = self.output_base / device_id / process_name
            bin_dir.mkdir(parents=True, exist_ok=True)
            
            # Export binary file
            bin_file = bin_dir / f"{raw_hash}.bin"
            
            if bin_file.exists() and skip_existing:
                self.logger.info(f"Binary file already exists: {bin_file}")
            else:
                with open(bin_file, 'wb') as f:
                    f.write(processed_data.tobytes())
                self.logger.info(f"Exported {len(processed_data)} bytes to {bin_file}")
            
            # Run tests
            for test_type in run_tests:
                self._run_and_save_test(device_id, process_name, bin_file, 
                                      raw_hash, test_type, skip_existing)
    
    def _run_and_save_test(self, device_id, process_name, bin_file, raw_hash, 
                          test_type, skip_existing):
        """Run a single entropy test and save results"""
        
        # Create results directory
        results_dir = self.results_base / device_id / process_name / test_type
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{raw_hash}.txt"
        
        if results_file.exists() and skip_existing:
            self.logger.info(f"Test results already exist: {results_file}")
            return
        
        # Determine which binary to use
        if test_type == 'iid':
            binary = self.ea_iid
        elif test_type == 'non_iid':
            binary = self.ea_non_iid
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Run the test
        self.logger.info(f"Running {test_type} test on {bin_file.name}")
        
        try:
            # Build command
            cmd = [str(binary), '-v', str(bin_file), '1']
            
            # Run with timeout
            start_time = datetime.now()
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=1800)  # 30 minute timeout
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if result.returncode != 0:
                self.logger.error(f"Test failed with return code {result.returncode}")
                self.logger.error(f"stderr: {result.stderr}")
            
            # Save output
            with open(results_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("="*80 + "\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDERR:\n")
                    f.write(result.stderr)
            
            self.logger.info(f"Test completed in {duration:.2f}s, saved to {results_file}")
            
            # Parse and log key results
            self._log_test_results(result.stdout, test_type)
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Test timed out after 30 minutes")
            with open(results_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("ERROR: Test timed out after 30 minutes\n")
        except Exception as e:
            self.logger.error(f"Error running test: {e}")
            raise
    
    def _log_test_results(self, stdout, test_type):
        """Parse and log key results from test output"""
        try:
            if test_type == 'iid':
                # Look for H_original
                if 'H_original:' in stdout:
                    h_original = stdout.split('H_original: ')[1].split('\n')[0]
                    self.logger.info(f"  H_original: {h_original}")
                
                # Check if passed
                if '** Passed IID permutation tests' in stdout:
                    self.logger.info("  ✓ Passed IID permutation tests")
                else:
                    self.logger.warning("  ✗ Failed IID permutation tests")
                    
            elif test_type == 'non_iid':
                # Look for min entropy
                if 'min(H_r, H_c):' in stdout:
                    min_entropy = stdout.split('min(H_r, H_c): ')[1].split('\n')[0]
                    self.logger.info(f"  Min entropy: {min_entropy}")
                elif 'min(H_r, H_c, H_I):' in stdout:
                    min_entropy = stdout.split('min(H_r, H_c, H_I): ')[1].split('\n')[0]
                    self.logger.info(f"  Min entropy: {min_entropy}")
        except Exception as e:
            self.logger.debug(f"Could not parse test results: {e}")


# Example usage
if __name__ == "__main__":
    from src.parquet_data_manager import ParquetExperimentalDataManager
    
    # Initialize
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path="/Users/shannonegan/Documents/DSV/StochastIQ/umn/SP800-90B_EntropyAssessment/cpp"
    )
    
    # Example 1: Process specific device
    pipeline.process_runs(
        polars_expr=(pl.col("device_id") == "07_08"),
        xor_levels=[1, 2, 4, 8],
        skip_existing=True
    )
    
    # Example 2: Process multiple devices with specific parameters
    pipeline.process_runs(
        polars_expr=(
            pl.col("device_id").is_in(["07_08", "23_07"]) & 
            (pl.col("field") == 10.0) &
            (pl.col("s_amplitude") > 0.1)
        ),
        xor_levels=[1, 8]
    )
    
    # Example 3: Complex filtering like in your notebook
    pipeline.process_runs(
        polars_expr=(
            (pl.col("s_amplitude") == 4.73) & 
            (pl.col("s_width") == 4e-9)
        ),
        xor_levels=[1, 2, 4, 8]
    )
    
    # Example 4: Process all devices with certain criteria
    pipeline.process_runs(
        polars_expr=(pl.col("repeats") > 1000),
        xor_levels=[1],
        run_tests=['iid']  # Only run IID test
    )