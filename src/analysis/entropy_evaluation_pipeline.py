# src/analysis/entropy_evaluation.py
import subprocess
import hashlib
import numpy as np
from pathlib import Path
import polars as pl
from datetime import datetime
import logging
from typing import List, Optional, Tuple
from src.analysis.processing import xorn_whitening
import sqlite3


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
                        tests: List[str] = ['non_iid', 'iid']):
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
        binary_data, raw_hash, sp, metadata_str = self.get_binary_data(device_id, polars_expr, run_ids)
        if binary_data is None:
            self.logger.warning("No data found matching criteria")
            return

        
        self.logger.info(f"Device {device_id}: {len(binary_data):,} samples, p = {sp:.5f}")
        

        # Set test as outer loop, since non-iid tests runs much faster you don't have to wait for results
        for test in tests:
            for xor_n in xor_levels:                
                if xor_n == 1:
                    process_name = "raw"
                else:
                    process_name = f"xor{xor_n}"

                # Kind of hacky/repetitive with export_binary, fix later. Construct bin_file the same way as export_binary in order to check if the target file already exists
                ## Added this when I reversed order of test and xor loop
                bin_dir = self.output_base / device_id / process_name            
                bin_file = bin_dir / f"{raw_hash}.bin"

                if not (bin_file.exists() and skip_existing):
                    if process_name == "raw":
                        processed_data = binary_data
                    else:
                        self.logger.info(f"Applying XOR-{xor_n} whitening")
                        processed_data = xorn_whitening(binary_data, n=xor_n)

                    self.logger.info(f"    P(X=1) = {np.average(processed_data):.7f}")
                    # Export binary file
                    bin_file = self.export_binary(device_id, process_name, processed_data, 
                                                 raw_hash, skip_existing)

                self.run_test(device_id, process_name, bin_file, raw_hash, 
                                 test, skip_existing, metadata_str)


    
    def get_binary_data(self, device_id: str, polars_expr: Optional[pl.Expr], 
                        run_ids: Optional[List[str]], 
                        group_by: Optional[list] = ["s_amplitude", "s_width", "r_amplitude", "r_width", "field"]) -> Tuple[Optional[np.ndarray], str, float, str]:
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
            
            # Sorting by row_number extremely important for preserving sequential order of bits
            ## Otherwise entropy may be over-estimated
            df = runs_df.filter(device_filter).sort("row_number")

            df_group_by = df.group_by(group_by, maintain_order=True)
            if df_group_by.len().height > 1:
                self.logger.warning("Combining runs with different pulse/field parameters in binary data")
            
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

        # Add switching probability to the tuple
        metadata_str = str(tuple(group_by) + ('switching_probability',))
        for name, data in df_group_by:
            metadata_str += f"\n{name + (float(f"{switching_probability:.6f}"),)}"

        
        return binary_data, raw_hash, switching_probability, metadata_str
    
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


    def export_restart_binaries(self, device_id=None, process_names=None, 
                          samples_per_file=1_000_000, skip_existing=True):
        """
        Create restart test sample files from existing binary files.
        Each restart file contains exactly samples_per_file samples (default 1M).
        
        Args:
            device_id: Specific device to process (e.g., "07_08"). If None, process all.
            process_names: List of process names (e.g., ["raw", "xor2"]). If None, process all.
            file_hash: Specific file hash to process. If provided with device_id but no process_names,
                       will search for this hash in all process directories of the device.
            samples_per_file: Number of samples per restart file (default 1,000,000)
            skip_existing: Skip if restart files already exist
            
        Returns:
            dict: Summary of created files {device_id: {process: {hash: num_files_created}}}
        """
        import numpy as np
        from pathlib import Path
        
        summary = {}
        
        # Convert process_names to list if it's a string
        if process_names is None:
            process_names = ["raw","xor2","xor4","xor8"]
        elif process_names and isinstance(process_names, str):
            process_names = [process_names]
        else:
            self.logger.info(f"User passed list of process names: {process_names}")    
        
        # Determine which devices to process
        if device_id:
            device_dirs = [self.output_base / device_id]
        else:
            device_dirs = [d for d in self.output_base.iterdir() if d.is_dir()]
        
        for device_dir in device_dirs:
            device_id = device_dir.name
            summary[device_id] = {}
            
            process_dirs = [device_dir / p for p in process_names if (device_dir / p).exists()]

            for process_dir in process_dirs:
                process_name = process_dir.name
                summary[device_id][process_name] = {}
                
                # Get all .bin files
                bin_files = list(process_dir.glob('*.bin'))
                
                for bin_file in bin_files:
                    # Extract hash from filename
                    file_hash = bin_file.stem
                    
                    # Create restart directory structure
                    restart_dir = process_dir / 'restart' / file_hash
                    
                    # Check if we should skip
                    if skip_existing and restart_dir.exists() and any(restart_dir.glob('*.bin')):
                        self.logger.info(f"Skipping {device_id}/{process_name}/{file_hash} - restart samples already exist")
                        summary[device_id][process_name][file_hash] = 0
                        continue
                    
                    # Create directory
                    restart_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Read the original binary file
                    self.logger.info(f"Reading {bin_file.name}")
                    with open(bin_file, 'rb') as f:
                        data = np.frombuffer(f.read(), dtype=np.uint8)
                    
                    total_samples = len(data)
                    num_files = total_samples // samples_per_file
                    
                    if num_files == 0:
                        self.logger.warning(f"File {bin_file.name} has only {total_samples} samples, "
                                          f"need at least {samples_per_file} for restart test")
                        summary[device_id][process_name][file_hash] = 0
                        continue
                    
                    # Create restart sample files
                    self.logger.info(f"Creating {num_files} restart files from {total_samples} total samples")
                    
                    for i in range(num_files):
                        start_idx = i * samples_per_file
                        end_idx = start_idx + samples_per_file
                        
                        # Extract the sample
                        sample_data = data[start_idx:end_idx]
                        
                        # Write to file
                        output_file = restart_dir / f"{i}.bin"
                        with open(output_file, 'wb') as f:
                            f.write(sample_data.tobytes())
                        
                        self.logger.debug(f"Created {output_file} with {len(sample_data)} samples")
                    
                    summary[device_id][process_name][file_hash] = num_files
                    
                    # Log info about unused samples
                    unused_samples = total_samples - (num_files * samples_per_file)
                    if unused_samples > 0:
                        self.logger.info(f"Discarded last {unused_samples} samples from {bin_file.name}")
        
        return summary

    def parse_test_result_file(self, result_path):
        """
        Parse a single test result file for H_original and failure status.
        
        Args:
            result_path: Path to the result file
            
        Returns:
            tuple: (h_original, failed) or (None, None) if file doesn't exist
        """
        if not result_path.exists():
            return None, None
        
        try:
            with open(result_path, 'r') as f:
                content = f.read()
            
            # Extract H_original
            h_original = None
            if 'H_original:' in content:
                h_line = content.split('H_original:')[1].split('\n')[0].strip()
                try:
                    h_original = float(h_line)
                except ValueError:
                    self.logger.warning(f"Could not parse H_original value: {h_line}")
            
            # Check for failure
            failed = 'Failed' in content
            
            return h_original, failed
            
        except Exception as e:
            self.logger.error(f"Error parsing {result_path}: {e}")
            return None, None

    def create_restart_test_plan(self, device_id=None, process_names=None, file_hash=None):
        """
        Create an execution plan for restart tests based on existing test results.
        
        Args:
            device_id: Specific device (if None, all devices)
            process_names: Specific processes (if None, all processes)
            file_hash: Specific file hash (if None, all files)
            
        Returns:
            list: List of restart test tasks, each containing:
                {
                    'device_id': str,
                    'process_name': str,
                    'file_hash': str,
                    'bin_file': Path,
                    'restart_dir': Path,
                    'test_flag': '-i' or '-n',
                    'h_original': float,
                    'num_restart_files': int
                }
        """
        tasks = []
        
        # Determine scope
        if device_id:
            device_dirs = [self.output_base / device_id]
        else:
            device_dirs = [d for d in self.output_base.iterdir() if d.is_dir()]
        
        for device_dir in device_dirs:
            current_device_id = device_dir.name
            
            # Filter processes
            if process_names:
                if isinstance(process_names, str):
                    process_names = [process_names]
                process_dirs = [device_dir / p for p in process_names if (device_dir / p).exists()]
            else:
                process_dirs = [d for d in device_dir.iterdir() 
                               if d.is_dir() and d.name != 'restart']
            
            for process_dir in process_dirs:
                current_process = process_dir.name
                
                # Filter files
                if file_hash:
                    bin_files = [process_dir / f"{file_hash}.bin"]
                    bin_files = [f for f in bin_files if f.exists()]
                else:
                    bin_files = list(process_dir.glob('*.bin'))
                
                for bin_file in bin_files:
                    current_hash = bin_file.stem
                    
                    # Check restart directory
                    restart_dir = process_dir / 'restart' / current_hash
                    if not restart_dir.exists() or not any(restart_dir.glob('*.bin')):
                        self.logger.warning(f"No restart files found for {current_device_id}/{current_process}/{current_hash[:12]}...")
                        continue
                    
                    # Count restart files
                    restart_files = list(restart_dir.glob('*.bin'))
                    num_restart_files = len(restart_files)
                    
                    # Parse IID results
                    iid_path = self.results_base / current_device_id / current_process / 'iid' / f"{current_hash}.txt"
                    iid_h, iid_failed = self.parse_test_result_file(iid_path)
                    
                    if iid_h is None:
                        self.logger.warning(f"No IID results for {current_device_id}/{current_process}/{current_hash[:12]}...")
                        continue
                    
                    # Determine test type based on IID results
                    if not iid_failed:
                        # IID passed - use IID restart
                        task = {
                            'device_id': current_device_id,
                            'process_name': current_process,
                            'file_hash': current_hash,
                            'bin_file': bin_file,
                            'restart_dir': restart_dir,
                            'test_flag': '-i',
                            'h_original': iid_h,
                            'num_restart_files': num_restart_files,
                            'source_test': 'iid'
                        }
                        tasks.append(task)
                    else:
                        # IID failed - use non-IID restart
                        non_iid_path = self.results_base / current_device_id / current_process / 'non_iid' / f"{current_hash}.txt"
                        non_iid_h, _ = self.parse_test_result_file(non_iid_path)
                        
                        if non_iid_h is None:
                            self.logger.warning(f"IID failed but no non-IID results for {current_device_id}/{current_process}/{current_hash[:12]}...")
                            continue
                        
                        task = {
                            'device_id': current_device_id,
                            'process_name': current_process,
                            'file_hash': current_hash,
                            'bin_file': bin_file,
                            'restart_dir': restart_dir,
                            'test_flag': '-n',
                            'h_original': non_iid_h,
                            'num_restart_files': num_restart_files,
                            'source_test': 'non_iid'
                        }
                        tasks.append(task)
        
        return tasks


    def run_test(self, device_id, process_name, bin_file, raw_hash, test_type, skip_existing, metadata_str):
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
        
        self.logger.info(f"Running {test_type} test on {str(bin_file)}")
        
        try:
            start_time = datetime.now()
            
            if test_type == 'iid':
                # For IID tests: monitor for failures and filter progress lines
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         text=True, bufsize=1)
                
                output_lines = []
                last_progress_line = None
                failed = False
                
                # Read output line by line
                for line in process.stdout:
                    line = line.rstrip()
                    
                    # Check for failure
                    if 'Failed' in line:
                        self.logger.warning(f"  IID test failed: {line}")
                        output_lines.append(line)
                        failed = True
                        process.terminate()
                        break
                    
                    # Filter progress lines
                    if '% of Permutation test rounds' in line and '% of Permutation tests' in line:
                        last_progress_line = line
                    else:
                        # Add the last progress line before non-progress content
                        if last_progress_line is not None:
                            output_lines.append(last_progress_line)
                            last_progress_line = None
                        output_lines.append(line)

                    if 'Passed IID permutation tests' in line:
                        self.logger.info(f"    IID tests passed: {line}")
                
                # Add final progress line if it exists
                if last_progress_line is not None:
                    output_lines.append(last_progress_line)
                
                # Wait for process to finish
                process.wait()
                stderr = process.stderr.read() if process.stderr else ""
                
                # Prepare output
                filtered_output = '\n'.join(output_lines)
                if failed:
                    filtered_output += "\n\n[Test terminated early due to failure]"
                
                result_returncode = process.returncode
                result_stderr = stderr
                
            else:
                # For non-IID tests: run normally without filtering
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                filtered_output = result.stdout
                result_returncode = result.returncode
                result_stderr = result.stderr
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Save results
            with open(results_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Return code: {result_returncode}\n")
                f.write(f"{metadata_str}\n")
                f.write("="*80 + "\n")
                f.write(filtered_output)
                if result_stderr:
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDERR:\n")
                    f.write(result_stderr)
            
            self.logger.info(f"Test completed in {duration:.2f}s")
            
            # Log key results for IID tests
            if 'H_original:' in filtered_output:
                h = filtered_output.split('H_original: ')[1].split('\n')[0]
                self.logger.info(f"  H_original: {h}")
                        
        except subprocess.TimeoutExpired:
            self.logger.error(f"Test timed out after 120 minutes")
            with open(results_file, 'w') as f:
                f.write("ERROR: Test timed out after 120 minutes\n")
