# entropy_evaluation_pipeline.py
import subprocess
import hashlib
import numpy as np
import sqlite3
from pathlib import Path
import polars as pl
from datetime import datetime
import logging
from typing import List, Optional, Tuple
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
    
    def evaluate_entropy(self, 
                        device_id: str,
                        polars_expr: Optional[pl.Expr] = None,
                        run_ids: Optional[List[str]] = None,
                        xor_levels: List[int] = [1, 2, 4, 8],
                        skip_existing: bool = True,
                        tests: List[str] = ['iid', 'non_iid']):
        """
        Evaluate entropy for a specific device
        
        Args:
            device_id: Device ID to process (e.g., "07_08")
            polars_expr: Polars expression for filtering runs
            run_ids: List of specific run IDs (must be pre-sorted by row_number!)
            xor_levels: List of XOR whitening levels (1 means raw)
            skip_existing: Skip if output files already exist
            tests: Which tests to run ('iid', 'non_iid')
            
        Example:
            pipeline.evaluate_entropy(
                device_id="07_08",
                polars_expr=(pl.col("field") == 10.0),
                xor_levels=[1, 2, 4, 8]
            )
        """
        # Get binary data
        binary_data, raw_hash, sp = self.get_binary_data(device_id, polars_expr, run_ids)
        if binary_data is None:
            self.logger.warning("No data found matching criteria")
            return
        
        self.logger.info(f"Device {device_id}: {len(binary_data):,} samples, p = {sp:.5f}")
        
        # Process each XOR level
        for xor_n in xor_levels:
            if xor_n == 1:
                processed_data = binary_data
                process_name = "raw"
            else:
                self.logger.info(f"Applying XOR-{xor_n} whitening")
                processed_data = xorn_whitening(binary_data, n=xor_n)
                process_name = f"xor{xor_n}"
            
            # Export binary file
            bin_file = self.export_binary(device_id, process_name, processed_data, 
                                         raw_hash, skip_existing)
            
            # Run tests
            for test in tests:
                self.run_test(device_id, process_name, bin_file, raw_hash, 
                             test, skip_existing)
    
    def get_binary_data(self, device_id: str, polars_expr: Optional[pl.Expr], 
                        run_ids: Optional[List[str]]) -> Tuple[Optional[np.ndarray], str, float]:
        """
        Get binary data from runs for a specific device
        
        Returns:
            Tuple of (binary_array, hash, switching_probability) or (None, "", 0.0) if no data
        """
        # Get run IDs to process
        if run_ids:
            # Direct path - skip database query for device_id
            runs_to_fetch = run_ids
            self.logger.info(f"Using {len(runs_to_fetch)} specified run IDs for device {device_id}")
                
        elif polars_expr is not None:
            self.logger.info("Applying Polars expression to filter runs")
            
            # Add device filter to expression
            device_filter = (pl.col("device_id") == device_id) & polars_expr
            
            with sqlite3.connect(self.dm.db_path) as conn:
                runs_df = pl.read_database("SELECT * FROM runs", conn)
            
            df = runs_df.filter(device_filter).sort("row_number")
            
            if len(df) == 0:
                self.logger.warning(f"No runs found for device {device_id} matching expression")
                return None, "", 0.0
            
            runs_to_fetch = df['run_id'].to_list()
            
        else:
            self.logger.error("Must provide run_ids or polars_expr")
            return None, "", 0.0
        
        # Get measurements
        measurements_df = self.dm.get_measurements(runs_to_fetch)
        if measurements_df.shape[0] == 0:
            self.logger.warning(f"No measurements found")
            return None, "", 0.0
        
        # Stack and convert to binary
        stacked_df = self.dm.stack_measurements(measurements_df)
        
        # Get threshold
        device_info = self.dm.get_device_info(device_id)
        if device_info:
            threshold = device_info[1]
            self.logger.info(f"Using stored threshold {threshold:.6f}")
        else:
            from src.analysis.processing import find_p_ap_states
            threshold, _, _ = find_p_ap_states(stacked_df)
            self.logger.info(f"Calculated threshold: {threshold:.6f}")
        
        # Convert to binary. Assumes AP -> P switching
        binary_data = (stacked_df['device_voltage'] < threshold).to_numpy().astype(np.uint8)
        raw_hash = hashlib.sha256(binary_data.tobytes()).hexdigest()
        switching_probability = np.mean(binary_data)
        
        return binary_data, raw_hash, switching_probability
    
    def export_binary(self, device_id, process_name, data, raw_hash, skip_existing):
        """Export binary data to file"""
        bin_dir = self.output_base / device_id / process_name
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        bin_file = bin_dir / f"{raw_hash}.bin"
        
        if not (bin_file.exists() and skip_existing):
            with open(bin_file, 'wb') as f:
                f.write(data.tobytes())
            self.logger.info(f"Exported {len(data)} bytes to {bin_file}")
        
        return bin_file
    
    def run_test(self, device_id, process_name, bin_file, raw_hash, test_type, skip_existing):
        """Run entropy test and save results"""
        # Setup
        results_dir = self.results_base / device_id / process_name / test_type
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{raw_hash}.txt"
        
        if results_file.exists() and skip_existing:
            self.logger.info(f"Test results already exist: {results_file}")
            return
        
        # Run test
        binary = self.ea_iid if test_type == 'iid' else self.ea_non_iid
        cmd = [str(binary), '-v', str(bin_file), '1']
        
        self.logger.info(f"Running {test_type} test on {bin_file.name}")
        
        try:
            start_time = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Save results
            # with open(results_file, 'w') as f:
            #     f.write(f"Command: {' '.join(cmd)}\n")
            #     f.write(f"Duration: {duration:.2f} seconds\n")
            #     f.write(f"Return code: {result.returncode}\n")
            #     f.write("="*80 + "\n")
            #     f.write(result.stdout)
            #     if result.stderr:
            #         f.write("\n" + "="*80 + "\n")
            #         f.write("STDERR:\n")
            #         f.write(result.stderr)

            # Filter out redundant progress lines
            lines = result.stdout.split('\n')
            filtered_lines = []
            last_progress_line = None
            
            for line in lines:
                # Check if this is a progress line
                if '% of Permutation test rounds' in line and '% of Permutation tests' in line:
                    last_progress_line = line
                else:
                    # If we had progress lines before this, add the last one
                    if last_progress_line is not None:
                        filtered_lines.append(last_progress_line)
                        last_progress_line = None
                    filtered_lines.append(line)
            
            # Don't forget the last progress line if it was the final line
            if last_progress_line is not None:
                filtered_lines.append(last_progress_line)
            
            filtered_output = '\n'.join(filtered_lines)
            
            # Save results
            with open(results_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("="*80 + "\n")
                f.write(filtered_output)  # <-- Changed from result.stdout
                if result.stderr:
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDERR:\n")
                    f.write(result.stderr)
            
            self.logger.info(f"Test completed in {duration:.2f}s")
            
            # Log key results
            if test_type == 'iid' and 'H_original:' in result.stdout:
                h = result.stdout.split('H_original: ')[1].split('\n')[0]
                self.logger.info(f"  H_original: {h}")
            elif test_type == 'non_iid':
                for pattern in ['min(H_r, H_c):', 'min(H_r, H_c, H_I):']:
                    if pattern in result.stdout:
                        h = result.stdout.split(pattern)[1].split('\n')[0]
                        self.logger.info(f"  Min entropy: {h}")
                        break
                        
        except subprocess.TimeoutExpired:
            self.logger.error(f"Test timed out after 30 minutes")
            with open(results_file, 'w') as f:
                f.write("ERROR: Test timed out after 30 minutes\n")


# Example usage
if __name__ == "__main__":
    from src.parquet_data_manager import ParquetExperimentalDataManager
    
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path="/Users/shannonegan/Documents/DSV/StochastIQ/umn/SP800-90B_EntropyAssessment/cpp"
    )
    
    # Process a device
    pipeline.evaluate_entropy(
        device_id="07_08",
        polars_expr=(pl.col("s_amplitude") == 4.73) & (pl.col("s_width") == 4e-9) & (pl.col("row_number").is_between(252,268)),
        xor_levels=[4, 8]
    )
    
    # Process with specific run IDs (fast path)
    # pipeline.evaluate_entropy(
    #     device_id="07_08",
    #     run_ids=["D07_08_SW100.0_SA0.15_F10.0_abc123", "D07_08_SW100.0_SA0.15_F10.0_def456"],
    #     xor_levels=[1, 8]
    # )