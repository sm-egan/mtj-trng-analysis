# src/parquet_data_manager.py
import pandas as pd
import polars as pl
import numpy as np
import sqlite3
import re
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List, Union
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
    def __init__(self, db_path='database/mtj_switching_data_parquet.db', 
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

            if 'binary_parquet_column' not in columns:
                self.logger.info("Adding binary_parquet_column column to runs table")
                cursor.execute("ALTER TABLE runs ADD COLUMN binary_parquet_column TEXT")
            
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
                existing_filename = conn.execute(
                    "SELECT filename FROM runs WHERE run_hash = ?",
                    (run_hash,)
                ).fetchone()

                if existing_filename[0] == filepath.name:
                    self.logger.info(f"Skipping duplicate run (existing run_id: {existing[0]})")
                    return existing[0]

                else:
                    self.logger.warning(f"Found identical run hash with non-identical filenames. A distinct run will be added to database but double-check files {existing_filename[0]} and {filepath.name} in the unlikely event of miscopied filename or identical data.")
        
        # Prepare Parquet file path
        parquet_file = f"{parquet_group}.parquet"
        parquet_path = self.parquet_dir / parquet_file
        
        # If data is 2D, flatten it
        if len(data.shape) > 1:
            data = data.flatten()

        data = data.astype(np.float32)
            
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
        Get measurements from Parquet storage. Optimized to read multiple columns
        from the same Parquet file in a single operation.
        
        Parameters:
        -----------
        run_id : str or list/array of str, optional
            If provided, get measurements for specified run(s)
            
        Returns:
        --------
        polars.DataFrame
            DataFrame containing measurements
        """
        empty_df = pl.DataFrame()

        # Handle case where run_id is None
        if run_id is None:
            self.logger.warning("Requesting all measurements is not supported with Parquet storage")
            return empty_df
        
        # Handle single run_id
        if isinstance(run_id, str):
            runs_info = self._get_parquet_info([run_id])
            if not runs_info:
                return empty_df
            return self._read_multiple_runs_from_parquet(runs_info)
        
        # Handle list/array of run_ids
        if isinstance(run_id, (list, np.ndarray, pd.Series)):
            run_ids = list(run_id)
            if not run_ids:
                return empty_df
            
            runs_info = self._get_parquet_info(run_ids)
            if not runs_info:
                return empty_df
            
            return self._read_multiple_runs_from_parquet(runs_info)
        
        # Invalid run_id type
        self.logger.error(f"Invalid run_id type: {type(run_id)}")
        return empty_df
    
    def _get_parquet_info(self, run_ids):
        """
        Get Parquet file and column information for multiple run IDs.
        
        Parameters:
        -----------
        run_ids : list
            List of run IDs to get information for
            
        Returns:
        --------
        dict
            Dictionary mapping Parquet file paths to lists of (run_id, column_name) tuples
        """
        from collections import defaultdict
        if not run_ids:
            return {}
            
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ', '.join(['?'] * len(run_ids))
            query = f"""
            SELECT run_id, parquet_file, parquet_column 
            FROM runs 
            WHERE run_id IN ({placeholders})
            AND parquet_file IS NOT NULL
            AND parquet_column IS NOT NULL
            """
            
            results = conn.execute(query, run_ids).fetchall()
            self.logger.debug(f"Result of query: {str(results)}")
            
        if not results:
            self.logger.warning(f"No Parquet storage info found for any of the requested run IDs")
            return {}

        # Group by Parquet file to optimize reading
        parquet_info = defaultdict(dict)
        for run_id, parquet_file, parquet_column in results:
            parquet_path = self.parquet_dir / parquet_file
            if parquet_path.exists():
                self.logger.debug(f"Appending to parquet_info {run_id} : {parquet_column}")

                parquet_info[str(parquet_path)].update({parquet_column: run_id})

            else:
                self.logger.warning(f"Parquet file not found: {parquet_path}")
                
        return parquet_info

    def _read_multiple_runs_from_parquet(self, parquet_info):
        """
        Read measurements for multiple runs, optimized to read columns from the
        same Parquet file in a single operation.
        
        Parameters:
        -----------
        parquet_info : dict
            Dictionary mapping Parquet file paths to lists of (run_id, column_name) tuples
            
        Returns:
        --------
        polars.DataFrame
            DataFrame with all measurements
        """
        # Initialize empty DataFrame with the right schema
        empty_df = pl.DataFrame()

        if not parquet_info:
            empty_df

        # Loop over each each Parquet file and the list of (run_id, column) tuples for each one
        for parquet_path, run_info in parquet_info.items():
            try:
                # Get all column names to read from this file as a list
                columns = list(run_info.keys())
                
                self.logger.debug(f"Reading {len(columns)} columns from {parquet_path}")
                
                # Read all columns at once for this file and convert pyarrow.Table object to Polars DataFrame
                # Also rename parquet columns to run_ids that match SQLite metadata
                df = pl.from_arrow(pq.read_table(parquet_path, columns=columns)).rename(run_info)

                return df
                
            except Exception as e:
                self.logger.error(f"Error reading from Parquet file {parquet_path}: {str(e)}")
        
        return empty_df

    def stack_measurements(self, measurements_df):
        """
        Parameters:
        -----------
        measurements_df : polars.DataFrame
            Dataframe in the format output by get_measurements where each column label is a run ID, rows are device voltages
        Returns:
        --------
        polars.DataFrame
            DataFrame with vertically stacked measurements and run_id info in each row
        """
        # List to collect individual DataFrames before final concatenation
        dfs_to_stack = []
        
        # Create empty result DataFrame for cases where no data is processed
        empty_df = pl.DataFrame(schema={
                'run_id': pl.Utf8,
                'row_index': pl.UInt32,
                'device_voltage': pl.Float32
        })
        
        for run_id in measurements_df.columns:
            try:
                # Get the column and drop nulls
                col = measurements_df[run_id].drop_nulls()
                
                if len(col) == 0:
                    continue
                
                # Create row indices using a simple range
                row_indices = pl.Series("row_index", range(len(col)), dtype=pl.UInt32)
                
                # Create a DataFrame for this run
                run_df = pl.DataFrame({
                    "run_id": pl.Series([run_id] * len(col), dtype=pl.Utf8),
                    "row_index": row_indices,
                    "device_voltage": col.cast(pl.Float32)
                })
                
                # Add to our collection
                dfs_to_stack.append(run_df)
                
            except Exception as e:
                self.logger.error(f"Error stacking column {run_id} from measurements DataFrame: {str(e)}", exc_info=True)
        
        # Concatenate all the DataFrames if we have any
        if dfs_to_stack:
            return pl.concat(dfs_to_stack)
        else:
            return empty_df

    def get_runs_by_criteria(self, criteria) -> list:
        """
        Get run IDs that match specific criteria.
        
        Parameters:
        -----------
        criteria : dict
            Dictionary of column name to value mappings for filtering
            
        Returns:
        --------
        list
            List of run_ids that match the criteria
        """
        query_parts = []
        params = []
        
        for column, value in criteria.items():
            if isinstance(value, (list, tuple)):
                # Handle IN clause for lists
                placeholders = ', '.join(['?'] * len(value))
                query_parts.append(f"{column} IN ({placeholders})")
                params.extend(value)
            else:
                # Handle equality for single values
                query_parts.append(f"{column} = ?")
                params.append(value)
        
        query = "SELECT run_id FROM runs"
        if query_parts:
            query += " WHERE " + " AND ".join(query_parts)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            run_ids = [row[0] for row in cursor.fetchall()]
        
        return run_ids
    
    def get_runs_with_measurements(self, criteria):
        """
        Get runs matching criteria along with their measurements.
        
        Parameters:
        -----------
        criteria : dict
            Dictionary of column name to value mappings for filtering
            
        Returns:
        --------
        tuple
            (runs_df, measurements_df) - Polars DataFrames with run metadata and measurements
        """
        # First get matching run_ids
        run_ids = self.get_runs_by_criteria(criteria)
        
        if not run_ids:
            self.logger.warning(f"No runs match the criteria: {criteria}")
            return (
                pl.DataFrame(), 
                pl.DataFrame(schema={
                    'run_id': pl.Utf8,
                    'device_voltage': pl.Float32
                })
            )
        
        # Get run metadata
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ', '.join(['?'] * len(run_ids))
            query = f"SELECT * FROM runs WHERE run_id IN ({placeholders})"
            runs_df = pd.read_sql_query(query, conn, params=run_ids)
        
        # Get measurements for all matching runs
        measurements_df = self.get_measurements(run_ids)
        
        # Convert runs_df to polars
        runs_pl_df = pl.from_pandas(runs_df)
        
        return runs_pl_df, measurements_df
    
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
        
        if len(measurements_df) == 0:
            return run_df  # Just return the metadata
            
        # Convert polars DataFrame to pandas if needed
        if isinstance(measurements_df, pl.DataFrame):
            measurements_pd_df = measurements_df.to_pandas()
        else:
            measurements_pd_df = measurements_df
            
        # Convert to wide format expected by application
        measurements_wide = pd.DataFrame(
            [measurements_pd_df['device_voltage'].values],
            columns=[f'measurement_{i}' for i in range(len(measurements_pd_df))]
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
