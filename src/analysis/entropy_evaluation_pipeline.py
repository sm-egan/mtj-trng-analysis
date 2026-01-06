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
from src.analysis.stats import compute_binary_statistics, compute_correlation_coefficients
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
                        xor_method: str = "nonoverlapping",
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
            tests: Which tests to run ('iid', 'non_iid', 'restart')
                Note: 'restart' requires 'iid' to run first, and 'non_iid' if IID fails
            
        Example:
            # Run all tests including restart
            pipeline.evaluate_entropy(
                device_id="07_08",
                polars_expr=(pl.col("field") == 10.0),
                xor_levels=[1, 2, 4, 8],
                tests=['iid', 'non_iid', 'restart']
            )
            
            # Run only restart tests (assumes IID/non-IID already completed)
            pipeline.evaluate_entropy(
                device_id="07_08",
                polars_expr=(pl.col("field") == 10.0),
                tests=['restart']
            )
        """
        # Validate test types
        valid_tests = {'iid', 'non_iid', 'restart'}
        invalid_tests = set(tests) - valid_tests
        if invalid_tests:
            raise ValueError(f"Invalid test types: {invalid_tests}. Valid options: {valid_tests}")
        
        # Check if restart is requested without prerequisite tests
        if 'restart' in tests:
            if 'iid' not in tests:
                self.logger.warning("Restart tests require IID results. Will check for existing IID results.")
        
        # Separate restart from other tests
        basic_tests = [t for t in tests if t != 'restart']
        run_restart = 'restart' in tests
        
        # Store hashes for each process level for restart tests
        process_hashes = {}
        
        # Run basic tests if requested
        if basic_tests:
            # Get binary data
            binary_data, raw_hash, sp, metadata_str = self.get_binary_data(device_id, polars_expr, run_ids)
            if binary_data is None:
                self.logger.warning("No data found matching criteria")
                return
            
            self.logger.info(f"Device {device_id}: {len(binary_data):,} samples, p = {sp:.5f}")
            
            # Run IID and non-IID tests
            for test in basic_tests:
                for xor_n in xor_levels:                
                    if xor_n == 1:
                        process_name = "raw"
                    else:
                        process_name = f"xor{xor_n}"
                        if xor_method == "rolling":
                            process_name = process_name + "_r"

                    # Store hash for each process level
                    process_hashes[process_name] = raw_hash

                    bin_dir = self.output_base / device_id / process_name            
                    bin_file = bin_dir / f"{raw_hash}.bin"

                    if not (bin_file.exists() and skip_existing):
                        if process_name == "raw":
                            processed_data = binary_data
                        else:
                            self.logger.info(f"Applying XOR-{xor_n} whitening")
                            processed_data = xorn_whitening(binary_data, n=xor_n, method=xor_method)

                        self.logger.info(f"    P(X=1) = {np.average(processed_data):.7f}")
                        # Export binary file
                        bin_file = self.export_binary(device_id, process_name, processed_data, 
                                                    raw_hash, skip_existing)

                    self.run_test(device_id, process_name, bin_file, raw_hash, 
                                test, skip_existing, metadata_str)
        
        # Run restart tests if requested
        if run_restart:
            self.logger.info("\n" + "="*80)
            self.logger.info("Starting restart tests")
            self.logger.info("="*80)
            
            # If we just ran basic tests, we know which files to process
            if process_hashes:
                # Create restart binaries for the files we just processed
                for process_name, file_hash in process_hashes.items():
                    self.logger.info(f"\nCreating restart samples for {device_id}/{process_name}/{file_hash[:12]}...")
                    self.export_restart_binaries(
                        device_id=device_id,
                        process_names=process_name,
                        file_hash=file_hash,
                        skip_existing=skip_existing
                    )
                
                # Run restart tests for these specific files
                self.logger.info("\nRunning restart tests...")
                for process_name, file_hash in process_hashes.items():
                    tasks = self.create_restart_test_plan(
                        device_id=device_id,
                        process_names=process_name,
                        file_hash=file_hash
                    )
                    if tasks:
                        self.run_restart_tests(tasks=tasks, skip_existing=skip_existing)
                    else:
                        self.logger.warning(f"No restart tests available for {device_id}/{process_name}/{file_hash[:12]}...")
            
            else:
                # No basic tests were run, so process all existing files
                self.logger.info("Running restart tests for all existing test results...")
                
                # Determine which processes to check based on xor_levels
                process_names_to_check = []
                for xor_n in xor_levels:
                    if xor_n == 1:
                        process_names_to_check.append("raw")
                    else:
                        process_names_to_check.append(f"xor{xor_n}")
                
                # Create restart binaries for all existing files
                self.logger.info(f"Creating restart samples for device {device_id}...")
                self.export_restart_binaries(
                    device_id=device_id,
                    process_names=process_names_to_check,
                    skip_existing=skip_existing
                )
                
                # Run restart tests
                self.run_restart_tests(
                    device_id=device_id,
                    process_names=process_names_to_check,
                    skip_existing=skip_existing
                )
        
        self.logger.info("\nEntropy evaluation complete.")
    
    def get_binary_data(self, device_id: str, polars_expr: Optional[pl.Expr], 
                        run_ids: Optional[List[str]], 
                        group_by: Optional[list] = ["s_amplitude", "s_width", "r_amplitude", "r_width", "field"]) -> Tuple[Optional[np.ndarray], str, float, str]:
        """
        Get binary data from runs for a specific device
        TO DO: ADD FUNCTIONALITY TO THIS METHOD TO CHECK IF # MEASUREMENTS IS DOUBLE THE # OF REPEATS.
        IF SO, USE SLICING METHOD FROM DEVICE 027_12 NOTEBOOK TO SEPARATE RESET AND PERTURB MEASUREMENTS
        
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


    def export_restart_binaries(self, device_id=None, process_names=None, file_hash=None,
                          samples_per_file=1_000_000, skip_existing=True):
        """
        Create restart test sample files from existing binary files.
        Each restart file contains exactly samples_per_file samples (default 1M).
        
        Args:
            device_id: Specific device to process (e.g., "07_08"). If None, process all.
            process_names: List of process names (e.g., ["raw", "xor2"]). If None, process all.
            file_hash: Specific file hashes to process. If None, process all .bin files.
            samples_per_file: Number of samples per restart file (default 1,000,000)
            skip_existing: Skip if restart files already exist
            
        Returns:
            dict: Summary of created files {device_id: {process: {hash: num_files_created}}}
        """
        import numpy as np
        from pathlib import Path
        
        summary = {}
        
        # Convert process_names to list if it's a string
        # if process_names is None:
        #     process_names = ["raw","xor2","xor4","xor8"]
        # elif process_names and isinstance(process_names, str):
        #     process_names = [process_names]
        # else:
        #     self.logger.info(f"User passed list of process names: {process_names}")
        
        # Determine which devices to process
        if device_id:
            device_dirs = [self.output_base / device_id]
        else:
            device_dirs = [d for d in self.output_base.iterdir() if d.is_dir() and '_' in d.name]
        
        for device_dir in device_dirs:
            device_id = device_dir.name
            summary[device_id] = {}
            
            if process_names:
                if isinstance(process_names, str):
                    process_names = [process_names]
                process_dirs = [device_dir / p for p in process_names if (device_dir / p).exists()]
            else:
                process_dirs = [d for d in device_dir.iterdir() 
                               if d.is_dir() and d.name != 'restart']

            for process_dir in process_dirs:
                process_name = process_dir.name
                summary[device_id][process_name] = {}
                
                # Get all .bin files
                if file_hash:
                    if isinstance(file_hash, str):
                        bin_files = [process_dir / f"{file_hash}.bin"]
                    else:  # assume it's a list/iterable
                        bin_files = [process_dir / f"{hash_val}.bin" for hash_val in file_hash]
                    bin_files = [f for f in bin_files if f.exists()]
                else:
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
        You can either specify a specific device, process names, and/or file hash as lists of strings,
        or leave them as None to process all devices, processes, and files.
        
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
            # Use presence of underscore to filter out "unstructured" directory
            device_dirs = [d for d in self.output_base.iterdir() if d.is_dir() and '_' in d.name]
        
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
                    if isinstance(file_hash, str):
                        bin_files = [process_dir / f"{file_hash}.bin"]
                    else:  # assume it's a list/iterable
                        bin_files = [process_dir / f"{hash_val}.bin" for hash_val in file_hash]
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

    def run_restart_tests(self, tasks=None, device_id=None, process_names=None, 
                        file_hash=None, skip_existing=True):
        """
        Run restart tests based on a task list or criteria.
        
        Args:
            tasks: List of restart test tasks from create_restart_test_plan()
                If None, will create task list using other parameters
            device_id: Specific device (used if tasks is None)
            process_names: Specific processes (used if tasks is None)
            file_hash: Specific file hash (used if tasks is None)
            skip_existing: Skip if restart test results already exist
            
        Returns:
            dict: Summary of test results
        """
        # If no tasks provided, create them
        if tasks is None:
            self.logger.info("Creating restart test plan...")
            tasks = self.create_restart_test_plan(device_id, process_names, file_hash)
            
        if not tasks:
            self.logger.warning("No restart tests to run")
            return {}
        
        self.logger.info(f"Running {len(tasks)} restart test sets")
        
        # Check if ea_restart binary exists
        ea_restart = self.ea_path / 'ea_restart'
        if not ea_restart.exists():
            raise FileNotFoundError(f"ea_restart not found at {ea_restart}")
        
        summary = {
            'total_tasks': len(tasks),
            'completed': 0,
            'skipped': 0,
            'failed': 0,
            'results': []
        }
        
        for task_idx, task in enumerate(tasks):
            self.logger.info(f"\nProcessing task {task_idx + 1}/{len(tasks)}: "
                            f"{task['device_id']}/{task['process_name']}/{task['file_hash'][:12]}...")
            
            # Get metadata from the source test file
            metadata_str = self._extract_metadata_from_test_results(task)
            
            # Process each restart file
            task_results = {
                'task': task,
                'restart_results': []
            }
            
            for i in range(task['num_restart_files']):
                restart_file = task['restart_dir'] / f"{i}.bin"
                
                # Determine output path
                results_dir = self.results_base / task['device_id'] / task['process_name'] / 'restart' / task['file_hash']
                results_dir.mkdir(parents=True, exist_ok=True)
                results_file = results_dir / f"{i}.txt"
                
                # Check if should skip
                if results_file.exists() and skip_existing:
                    self.logger.info(f"  Skipping restart file {i} - results already exist")
                    summary['skipped'] += 1
                    continue
                
                # Run restart test
                result = self._run_single_restart_test(
                    ea_restart, restart_file, task['test_flag'], 
                    task['h_original'], results_file, metadata_str
                )
                
                if result['success']:
                    summary['completed'] += 1
                    task_results['restart_results'].append(result)
                    self.logger.info(f"  Restart file {i}: H_min = {result.get('h_final', 'N/A')}")
                else:
                    summary['failed'] += 1
                    self.logger.error(f"  Restart file {i}: Error running restart test")
            
            summary['results'].append(task_results)
        
        # Log summary
        self.logger.debug(f"\nRestart test summary:")
        self.logger.debug(f"  Total tasks: {summary['total_tasks']}")
        self.logger.debug(f"  Completed tests: {summary['completed']}")
        self.logger.debug(f"  Skipped tests: {summary['skipped']}")
        self.logger.debug(f"  Failed tests: {summary['failed']}")
        
        return summary
    
    def _run_single_restart_test(self, ea_restart, restart_file, test_flag, 
                                h_original, results_file, metadata_str):
        """
        Run a single restart test and save results.
        
        Args:
            ea_restart: Path to ea_restart binary
            restart_file: Path to restart sample file
            test_flag: '-i' or '-n'
            h_original: Original entropy value
            results_file: Path to save results
            metadata_str: Metadata string to include in results
            
        Returns:
            dict: Result information including success status
        """
        # Construct command
        cmd = [str(ea_restart), test_flag, '-v', str(restart_file), '1', f"{h_original:.6f}"]
        
        self.logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            # start_time = datetime.now()
            # result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            # duration = (datetime.now() - start_time).total_seconds()

            start_time = datetime.now()
            
            if test_flag == '-i':
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
                    if '** Failed' in line:
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
            
            # Save results with header similar to iid/non_iid tests
            with open(results_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Return code: {result_returncode}\n")
                f.write(metadata_str)
                f.write("\n" + "="*80 + "\n")
                f.write(filtered_output)
                if result_stderr:
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDERR:\n")
                    f.write(result_stderr)
            
            # Check that validation test passed
            validation_passed = False
            if 'Validation Test Passed' in filtered_output:
                validation_passed = True
                self.logger.debug("Validation test passed")

            # Extract H_original from output for logging
            h_value = None
            if 'min(H_r, H_c, H_I):' in filtered_output:
                h_line = filtered_output.split('min(H_r, H_c, H_I): ')[1].split('\n')[0].strip()
                try:
                    h_value = float(h_line)
                except ValueError:
                    pass
            
            return {
                'success': result_returncode == 0,
                'validation_passed': validation_passed,
                'duration': duration,
                'h_final': h_value,
                'file': str(restart_file)
        
            }
        except Exception as e:
            self.logger.error(f"Error running restart test: {e}")
            return {'success': False, 'error': str(e)}
        

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

    def _extract_metadata_from_test_results(self, task):
        """
        Extract metadata string from the source test results file.
        
        Args:
            task: Task dictionary containing test information
            
        Returns:
            str: Metadata string to include in restart test results
        """
        # Determine which test file to read from
        test_type = task['source_test']  # 'iid' or 'non_iid'
        source_results_file = (self.results_base / task['device_id'] / 
                            task['process_name'] / test_type / 
                            f"{task['file_hash']}.txt")
        
        if not source_results_file.exists():
            self.logger.warning(f"Could not find source test file: {source_results_file}")
            return "Metadata not available"
        
        try:
            with open(source_results_file, 'r') as f:
                content = f.read()
            
            # Extract the metadata section (between Return code and ===)
            lines = content.split('\n')
            metadata_lines = []
            in_metadata = False
            
            for line in lines:
                if line.startswith('Return code:'):
                    in_metadata = True
                    continue
                # WARNING: This assumes the metadata section ends with a line of 80 equals signs.
                elif line.startswith('=' * 80):
                    break
                elif in_metadata and line.strip():
                    metadata_lines.append(line)
            
            return '\n'.join(metadata_lines)
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return "Error extracting metadata"    
    
    def collect_entropy_results(self, device_id=None, process_names=None, output_csv=None,
                                include_binary_stats=True, correlation_lags=None):
        """
        Collect entropy test results from all test files and compile into a CSV.
        
        Args:
            device_id: Specific device to process (if None, all devices)
            process_names: Specific processes to include (if None, all processes)
            output_csv: Path to output CSV file (if None, returns DataFrame)
            include_binary_stats: Whether to include binary file statistics (default: True)
            correlation_lags: List of correlation lags to compute (default: [1, 2, 4, ..., 1024])

        Returns:
            pd.DataFrame: DataFrame with entropy results
        """
        import pandas as pd
        from pathlib import Path
        
        results_list = []
        
        # Determine scope
        if device_id:
            device_dirs = [self.results_base / device_id]
        else:
            device_dirs = [d for d in self.results_base.iterdir() if d.is_dir() and d.name.split('_')[0].isdigit()]

        if process_names:
            if isinstance(process_names, str):
                process_names = [process_names]
        
        # Through device directories
        for device_dir in device_dirs:
            current_device = device_dir.name
            
            # Filter processes
            if process_names:
                process_dirs = [device_dir / p for p in process_names if (device_dir / p).exists()]
            else:
                process_dirs = [d for d in device_dir.iterdir() if d.is_dir()]

            file_hashes = self._get_unique_hashes(device_dir)
            self.logger.info(f"Found {len(file_hashes)} unique file hashes for Device {current_device}: {file_hashes}")

            for process_dir in process_dirs:
                current_process = process_dir.name

                # Get all unique file hashes for this device and process
                ## device, process, and file_hash uniquely specify a row in the DataFrame/csv
                self.logger.info(f"Processing {process_dir}")

                for file_hash in file_hashes:
                    # Initialize result dictionary. Results for all test types will be aggregated here
                    result = {
                        'device': current_device,
                        'processing': current_process,
                        'raw_hash': file_hash,
                        'switching_probability': None,
                        's_amplitude': None,
                        's_width': None,
                        'r_amplitude': None,
                        'r_width': None,
                        'field': None,
                        'iid_pass': None,
                        'restart_type':None,
                        'restart_pass': None,
                        'h_original_iid': None,
                        'h_original_non_iid': None,
                        'h_final': None
                    }
                    
                    get_metadata = True
                    for test_type in ['iid', 'non_iid', 'restart']:
                        test_dir = process_dir / test_type
                        if not test_dir.exists():
                            self.logger.warning(f"Test type {test_type} directory does not exist for {test_dir}. Skipping...")
                            continue

                        # Parse IID and non-IID results
                        if test_type in ['iid', 'non_iid']:
                            test_file = test_dir / f"{file_hash}.txt"

                            if not test_file.exists():
                                self.logger.warning(f"Missing {test_type} results for {current_device}/{current_process}/{file_hash}...")
                                continue

                            # Only get metadata if it's the first time parsing this file
                            parsed_result, metadata_found = self._parse_iid_non_iid_result_file(test_file, test_type=test_type, find_metadata=get_metadata)
                            # if _parse_iid_non_iid_result_file returned metadata_found, we can skip it next time
                            get_metadata = not metadata_found
                            
                            if parsed_result:
                                result.update(parsed_result)
                                self.logger.info(f"{test_type.capitalize()} results for {current_device}/{current_process}/{file_hash}: {parsed_result}")
                        
                        elif test_type == 'restart':
                            restart_dir = test_dir / file_hash
                            if not restart_dir.exists():
                                self.logger.warning(f"Missing restart results for {current_device}/{current_process}/{file_hash}...")
                                continue

                            parsed_result = self._parse_restart_results(test_dir / file_hash)
                            self.logger.debug(f"Parsed restart results for {current_device}/{current_process}/{file_hash}: {parsed_result}")
                            if parsed_result:
                                result.update(parsed_result)
                                self.logger.info(f"Restart results for {current_device}/{current_process}/{file_hash}: {parsed_result}")
                    
                    # Add binary statistics to the result
                    if include_binary_stats:
                        binary_stats = self._get_binary_statistics(current_device, current_process, file_hash, correlation_lags)
                        result.update(binary_stats)

                    # Add result to the list
                    results_list.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results_list).sort_values(by=['device', 'r_amplitude', 'r_width', 'processing', 'switching_probability'])
        
        # Reorder columns to match the expected format
        column_order = [
            'device', 'processing', 'raw_hash',
            'switching_probability', 's_amplitude', 's_width', 
            'r_amplitude', 'r_width', 'field', 'iid_pass', 'restart_type', 'restart_pass', 
            'h_original_iid', 'h_original_non_iid', 'h_final'
        ]

        # Add binary statistics columns if requested
        if include_binary_stats:
            column_order.extend(self._get_binary_statistics_column_names(correlation_lags))
        
        # Add missing columns with NaN
        for col in column_order:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Reorder and select columns
        df = df[column_order]
        
        # Save to CSV if path provided
        if output_csv:
            df.to_csv(output_csv, index=False)
            self.logger.info(f"Saved {len(df)} results to {output_csv}")
        
        return df

    def _get_unique_hashes(self, device_dir):
        device_path = Path(device_dir)
        processing_levels = ['raw', 'xor2', 'xor4', 'xor8']
        test_types = ['iid', 'non_iid']
        
        all_hashes = set()
        
        for proc_level in processing_levels:
            for test_type in test_types:
                dir_path = device_path / proc_level / test_type
                
                if not dir_path.exists():
                    continue
                
                # Convert generator to set of stems, then union
                dir_hashes = {txt_file.stem for txt_file in dir_path.glob('*.txt')}
                all_hashes |= dir_hashes  # Union operator
        
        return sorted(list(all_hashes))

    def _parse_iid_non_iid_result_file(self, result_file, test_type=None, find_metadata=False):
        """
        Parse a single IID or non-IID test result file.
        
        Args:
            txtfile: Path to the result file
            test_type: 'iid' or 'non_iid' (optional, used for logging)
            find_metadata: If True, extract metadata from the file
        Returns:
            dict: Parsed results or None if parsing fails
        """
        try:
            with open(result_file, 'r') as f:
                content = f.read()

            result = {}

            if not test_type:
                test_type = 'iid' if 'ea_iid' in content else 'non_iid'
            
            h_original = None
            if 'H_original:' in content:
                h_line = content.split('H_original:')[1].split('\n')[0].strip()
                try:
                    h_original = float(h_line)
                except ValueError:
                    self.logger.warning(f"Could not parse H_original value: {h_line}")
                    return None       

            if test_type == 'iid':
                result['iid_pass'] = 'Passed' in content and 'Failed' not in content
                result['h_original_iid'] = h_original
            elif test_type == 'non_iid':
                result['h_original_non_iid'] = h_original
            else:
                self.logger.warning(f"Unknown test type: {test_type}")
                return None
            
            if find_metadata:
                metadata = self._parse_entropy_result_for_metadata(content)
                if metadata:
                    result.update(metadata)
                    return result, True
                else:
                    self.logger.warning(f"Could not extract metadata from {result_file}")
                    return result, False
            else:
                return result, False
        except Exception as e:
            self.logger.error(f"Error parsing {result_file}: {e}")
            return None
    
    def _parse_restart_results(self, restart_result_dir):
        """
        Parse all restart test results in a directory.
        
        Args:
            result_dir: Path to the directory containing restart test results
            get_metadata: If True, extract metadata from the results
            
        Returns:
            list: List of parsed results
        """
        result = {}
        if not restart_result_dir.exists():
            self.logger.warning(f"Restart result directory does not exist: {restart_result_dir}")
            return None
        
        # Get all restart files for this hash
        restart_files = sorted(restart_result_dir.glob('*.txt'))
        self.logger.info(f"Found {len(restart_files)} restart files in {restart_result_dir}")
        if restart_files:
            # Process all restart files to find minimum entropy
            min_h = float('inf')
            all_passed = True
            
            for restart_file in restart_files:
                try:
                    with open(restart_file, 'r') as f:
                        content = f.read()
                    
                    # Extract restart test type
                    if 'ea_restart -i' in content:
                        restart_type = 'iid'
                    else:
                        restart_type = 'non_iid'

                    # Check pass/fail
                    if '** Failed' in content:
                        all_passed = False #TO DO: IF THIS HAPPENS AND WE'RE DOING IID TESTS WE SHOULD SWITCH TO NON-IID

                    # Extract min entropy
                    if 'min(H_r, H_c, H_I): ' in content:
                        h_line = content.split('min(H_r, H_c, H_I): ')[1].split('\n')[0].strip()
                        try:
                            h_value = float(h_line)
                            min_h = min(min_h, h_value)
                        except ValueError:
                            pass
                except Exception as e:
                    self.logger.error(f"Error reading {restart_file}: {e}")

        else:
            self.logger.warning(f"No restart files found in {restart_result_dir}")
            return None

        if min_h != float('inf'):
            result['h_final'] = min_h
            result['restart_pass'] = all_passed
            result['restart_type'] = restart_type
        else:
            self.logger.warning(f"No valid H values found in restart files for {restart_result_dir}")
            return None
        
        return result
        
    def _parse_entropy_result_for_metadata(self, file_content):
        """        Extract metadata from the content of an entropy test result file."""
        # Extract metadata (switching probability and parameters)
        metadata_match = False
        result = {}
        for line in file_content.split('\n'):
            if "('s_amplitude'," in line and 'switching_probability' in line:
                metadata_match = True
                continue
            if metadata_match and line.startswith('(') and line.endswith(')'):
                # Parse the tuple
                try:
                    values = eval(line)  # Safe here as we control the format
                    result['s_amplitude'] = values[0]
                    result['s_width'] = values[1]
                    result['r_amplitude'] = values[2]
                    result['r_width'] = values[3]
                    result['field'] = values[4]
                    result['switching_probability'] = values[5]
                except:
                    self.logger.warning(f"Could not parse metadata")
                break
        if len(result) > 0:
            self.logger.debug(f"Extracted metadata: {result}")
            return result
        else:
            return None

    def _get_binary_statistics(self, device_id, process_name, file_hash, 
                            correlation_lags=None):
        """
        Compute statistics for a binary file.
        
        Args:
            device_id: Device ID
            process_name: Process name (e.g., 'raw', 'xor2')
            file_hash: File hash
            correlation_lags: List of lags for autocorrelation (default: powers of 2 up to 2^20)
            
        Returns:
            dict: Flat statistics dictionary with keys: mean, variance, corr_lag_1, corr_lag_2, etc.
                Returns dict with None values if file is missing or empty.
        """
        if correlation_lags is None:
            correlation_lags = (2 ** np.arange(0, 21)).tolist()
        
        # Initialize result with None values
        stats_result = {'mean': None, 'variance': None}
        for lag in correlation_lags:
            stats_result[f'corr_lag_{lag}'] = None
        
        # Construct path to binary file
        bin_file = self.output_base / device_id / process_name / f"{file_hash}.bin"
        
        if not bin_file.exists():
            self.logger.debug(f"Binary file not found: {bin_file}")
            return stats_result
        
        # Read and compute
        try:
            with open(bin_file, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            
            if len(data) == 0:
                self.logger.warning(f"Empty binary file: {bin_file}")
                return stats_result
            
            # Delegate to stats module
            stats = compute_binary_statistics(data, correlation_lags=correlation_lags)
            
            # Log summary
            self.logger.info(f"Binary statistics for {bin_file.name}:")
            self.logger.info(f"  Samples: {stats['n_samples']:,}")
            self.logger.info(f"  Mean: {stats['mean']:.6f}")
            self.logger.info(f"  Variance: {stats['variance']:.6f}")
            if 1 in stats['correlations']:
                self.logger.info(f"  Correlation (lag=1): {stats['correlations'][1]:.6f}")
            
            # Flatten for CSV export
            stats_result['mean'] = stats['mean']
            stats_result['variance'] = stats['variance']
            for lag in correlation_lags:
                if lag in stats['correlations']:
                    stats_result[f'corr_lag_{lag}'] = stats['correlations'][lag]
            
            return stats_result
                
        except Exception as e:
            self.logger.error(f"Error computing binary statistics for {bin_file}: {e}")
            return stats_result

    def _get_binary_statistics_column_names(self, correlation_lags=None):
        """
        Get list of column names for binary statistics.
        
        Args:
            correlation_lags: List of correlation lags 
            
        Returns:
            list: Column names for binary statistics
        """
        if correlation_lags is None:
            correlation_lags = np.power(2, np.arange(0, 21)) # Powers of 2 from 0 to 1 MiB
        
        columns = ['mean', 'variance']
        for lag in correlation_lags:
            columns.append(f'corr_lag_{lag}')
        
        return columns