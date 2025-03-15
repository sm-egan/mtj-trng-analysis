# src/parquet_data_manager.py
import pandas as pd
import numpy as np
import sqlite3
import re
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from src.data_manager import ExperimentalDataManager

class ParquetExperimentalDataManager(ExperimentalDataManager):
    """
    Extends ExperimentalDataManager to store measurement data in Parquet files
    instead of the SQLite database. Uses a columnar approach where each run's
    measurements are stored as a separate column in the Parquet file.
    """
    def __init__(self, db_path='database/mtj_switching_data.db', 
                 parquet_dir='database/parquet_data', 
                 data_dir='sample_data', logging_level='INFO'):
        """
        Initialize the Parquet data manager.
        
        Parameters:
        -----------
        db_path : str or Path
            Path to the SQLite database
        parquet_dir : str or Path
            Directory to store Parquet files
        data_dir : str or Path
            Path to directory containing data files
        logging_level : str
            Logging level ('INFO', 'DEBUG', etc.)
        """
        # Initialize parent class
        super().__init__(db_path=db_path, data_dir=data_dir, logging_level=logging_level)
        
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure our schema includes fields for Parquet storage
        self.initialize_parquet_schema()
    
    def initialize_parquet_schema(self):
        """Modify the database schema to include fields for Parquet storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if columns already exist
            cursor.execute("PRAGMA table_info(runs)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Add parquet_file column if it doesn't exist
            if 'parquet_file' not in columns:
                self.logger.info("Adding parquet_file column to runs table")
                cursor.execute("ALTER TABLE runs ADD COLUMN parquet_file TEXT")
            
            # Add parquet_column column if it doesn't exist (for columnar storage)
            if 'parquet_column' not in columns:
                self.logger.info("Adding parquet_column column to runs table")
                cursor.execute("ALTER TABLE runs ADD COLUMN parquet_column TEXT")
            
            conn.commit()
    
    def initialize_database(self):
        """Override to include our Parquet fields"""
        super().initialize_database()
        self.initialize_parquet_schema()
    
    def import_data_file(self, filepath):
        """
        Import measurement data from a file, storing measurements in Parquet.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the data file to import
            
        Returns:
        --------
        str or None
            The run_id of the imported data, or None if import failed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Get folder name to use for Parquet grouping
        folder_name = filepath.parent.name
        # Sanitize folder name for use as a filename
        parquet_group = re.sub(r'[^\w\-_]', '_', folder_name)
        if not parquet_group:
            parquet_group = "ungrouped"
        
        # Get device ID from parent folder name if it exists
        device_id = folder_name.split("Device_")[1] if folder_name.startswith('Device_') else None

        # Validate filename format
        if not self.validator.is_valid_filename(filepath.name):
            self.logger.warning(f"Skipping file with invalid format: {filepath.name}")
            return None
        
        # Parse parameters from filename
        params = self.parse_filename(filepath.name)
        if device_id:
            params['device_id'] = device_id
        
        # Load the data file
        try:
            self.logger.debug(f"Loading data from {filepath.name}")
            data = np.loadtxt(filepath, delimiter='\t')
        except ValueError as e:
            self.logger.debug(f"numpy.loadtxt failed with error: {str(e)}")
            try:
                self.logger.debug(f"Attempting pandas read for {filepath.name}")
                df = pd.read_csv(filepath, delimiter='\t', header=None)
                max_cols = df.shape[1]
                valid_data = df[df.notna().sum(axis=1) == max_cols]
                if len(valid_data) > 0:
                    data = valid_data.iloc[0].values
                else:
                    raise ValueError("No valid data rows found")
            except Exception as e2:
                self.logger.error(f"Both reading methods failed for {filepath.name}")
                raise

        # Generate run ID and hash
        try:
            run_id, run_hash = self.generate_run_id(params, data)
        except ValueError as e:
            self.logger.error(f"Run ID generation failed for {filepath.name}")
            return None
        
        # Check if run already exists
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT run_id FROM runs WHERE run_hash = ?", 
                (run_hash,)
            ).fetchone()
            
            if existing:
                self.logger.info(f"Skipping duplicate run (existing run_id: {existing[0]})")
                return existing[0]
        
        # Prepare Parquet file path
        parquet_file = f"{parquet_group}.parquet"
        parquet_path = self.parquet_dir / parquet_file
        
        # If data is 2D, flatten it
        if len(data.shape) > 1:
            data = data.flatten()
            
        # Generate a clean column name from run_id that's safe for parquet
        column_name = f"run_{re.sub(r'[^\w]', '_', run_id)}"
        
        # Create or update the Parquet file
        if parquet_path.exists():
            try:
                # Read existing parquet file
                existing_df = pd.read_parquet(parquet_path)
                
                # Check if the column already exists
                if column_name in existing_df.columns:
                    self.logger.info(f"Column {column_name} already exists in {parquet_file}")
                else:
                    # Add the new column
                    # First ensure all data arrays are the same length by padding
                    max_length = max([len(data)] + [len(existing_df[col].dropna()) for col in existing_df.columns])
                    
                    # Pad data if needed
                    padded_data = np.pad(data, (0, max(0, max_length - len(data))), 
                                         'constant', constant_values=np.nan)
                    
                    # Add the new column
                    existing_df[column_name] = pd.Series(padded_data)
                    
                    # Save back to parquet
                    existing_df.to_parquet(parquet_path, engine='pyarrow', index=False)
                    self.logger.info(f"Added column {column_name} to {parquet_file}")
            except Exception as e:
                self.logger.error(f"Error updating Parquet file: {str(e)}")
                
                # If we can't update, create a new file with just this column
                df = pd.DataFrame({column_name: pd.Series(data)})
                df.to_parquet(parquet_path, engine='pyarrow', index=False)
                self.logger.info(f"Created new Parquet file {parquet_file} with column {column_name}")
        else:
            # Create new Parquet file
            df = pd.DataFrame({column_name: pd.Series(data)})
            df.to_parquet(parquet_path, engine='pyarrow', index=False)
            self.logger.info(f"Created new Parquet file {parquet_file} with column {column_name}")
        
        # Insert run metadata into SQLite (with Parquet reference)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert run metadata
            cursor.execute('''
                INSERT INTO runs 
                (run_id, run_hash, device_id, filename, hz, row_number, r_amplitude, r_width,
                 s_amplitude, s_width, field, source, repeats, parquet_file, parquet_column)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, run_hash, 
                params.get('device_id'),
                filepath.name,
                params['hz'], params['row'],
                params['r_amplitude'], params['r_width'],
                params['s_amplitude'], params['s_width'],
                params['field'], params['source'], params['repeats'],
                parquet_file, column_name
            ))
            
            conn.commit()
        
        self.logger.info(f"Successfully imported run {run_id} to Parquet file {parquet_file}")
        return run_id
    
    def get_measurements(self, run_id=None):
        """
        Get measurements from Parquet storage if available.
        
        Parameters:
        -----------
        run_id : str, optional
            If provided, get measurements for a specific run
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing measurements
        """
        if run_id is None:
            # Getting all measurements could be very inefficient
            # Let's just return an empty DataFrame with the right columns
            self.logger.warning("Requesting all measurements is not supported with Parquet storage")
            return pd.DataFrame(columns=['run_id', 'device_voltage', 'row_index', 'col_index'])
        
        # Get Parquet location information
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT parquet_file, parquet_column FROM runs WHERE run_id = ?"
            result = conn.execute(query, (run_id,)).fetchone()
            
            if not result or not result[0] or not result[1]:
                self.logger.warning(f"No Parquet storage info found for run {run_id}")
                # Fall back to original method? (there should be no measurements in SQLite)
                return pd.DataFrame(columns=['run_id', 'device_voltage', 'row_index', 'col_index'])
            
            parquet_file, parquet_column = result
        
        # Load from Parquet
        parquet_path = self.parquet_dir / parquet_file
        if not parquet_path.exists():
            self.logger.error(f"Parquet file not found: {parquet_path}")
            return pd.DataFrame(columns=['run_id', 'device_voltage', 'row_index', 'col_index'])
        
        try:
            # Read the specific column from the Parquet file
            # Use pyarrow to read just the column we need
            table = pq.read_table(parquet_path, columns=[parquet_column])
            df = table.to_pandas()
            
            # Extract the data from the column
            voltage_array = df[parquet_column].dropna().values
            
            # Create DataFrame in the format expected by the application
            measurements_df = pd.DataFrame({
                'run_id': [run_id] * len(voltage_array),
                'device_voltage': voltage_array,
                'row_index': [0] * len(voltage_array),
                'col_index': range(len(voltage_array))
            })
            
            return measurements_df
            
        except Exception as e:
            self.logger.error(f"Error loading measurements from Parquet: {str(e)}")
            return pd.DataFrame(columns=['run_id', 'device_voltage', 'row_index', 'col_index'])
            
    def get_run_with_measurements(self, run_id):
        """
        Get a single run with all its measurements.
        Overrides parent method to handle Parquet storage.
        """
        # Get run metadata
        with sqlite3.connect(self.db_path) as conn:
            run_query = "SELECT * FROM runs WHERE run_id = ?"
            run_df = pd.read_sql_query(run_query, conn, params=[run_id])
            
        if run_df.empty:
            return None
            
        # Get measurements
        measurements_df = self.get_measurements(run_id)
        
        if measurements_df.empty:
            return run_df  # Just return the metadata
            
        # Convert to wide format expected by application
        measurements_wide = pd.DataFrame(
            [measurements_df['device_voltage'].values],
            columns=[f'measurement_{i}' for i in range(len(measurements_df))]
        )
        
        # Combine run metadata with measurements
        result = pd.concat([run_df, measurements_wide], axis=1)
        return result
        
    def clean_up_parquet_files(self):
        """
        Optimize Parquet files by removing unused columns and combining small files.
        This is a maintenance operation that can be run periodically.
        """
        # This would be an additional feature to optimize storage
        # For now, we'll just implement a placeholder
        self.logger.info("Parquet cleanup not implemented yet")
