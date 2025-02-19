import pandas as pd
import numpy as np
import sqlite3
import re
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import hashlib
from src.data_validation import DataValidator

class ExperimentalDataManager:
    def __init__(self, db_path='database/experiment_data.db', data_dir='sample_data'):
        """
        Initialize the data manager with a SQLite database connection.
        
        Parameters:
        -----------
        db_path : str or Path
            Path to the SQLite database file
        data_dir : str or Path
            Path to directory containing data files
        """
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.validator = DataValidator()
        
        # Initialize database
        self.initialize_database()
    
    def initialize_database(self):
        """Create the database schema if it doesn't exist."""
        self.logger.info("Initializing database...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Create runs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    hz REAL,
                    row_number INTEGER,
                    r_amplitude REAL,
                    r_width REAL,
                    s_amplitude REAL,
                    s_width REAL,
                    field REAL,
                    source REAL,
                    repeats INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create measurements table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS measurements (
                    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    data_point REAL,
                    point_index INTEGER,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            ''')
    
    def parse_filename(self, filename):
        """
        Parse experimental parameters from filename.
        
        Parameters:
        -----------
        filename : str or Path
            Name of the file to parse
            
        Returns:
        --------
        dict
            Dictionary of parsed parameters
        """
        # Convert to Path object and get stem (filename without extension)
        filepath = Path(filename)
        name_to_parse = filepath.stem
        
        self.logger.debug(f"Parsing parameters from: {name_to_parse}")
        
        patterns = {
            'hz': r'Hz_(\d+)',
            'row': r'Row\s*(\d+)',
            'r_amplitude': r'RAmp-([\d.E+-]+)',
            'r_width': r'RWid([\d.E+-]+)',
            's_amplitude': r'SAmp([\d.E+-]+)',
            's_width': r'SWid([\d.E+-]+)',
            'field': r'Field([\d.E+-]+)',
            'source': r'Source([\d.E+-]+)',
            'repeats': r'Repeats([\d.E+-]+)'
        }
        
        params = {}
        for key, pattern in patterns.items():
            try:
                match = re.search(pattern, name_to_parse)
                if match:
                    value_str = match.group(1)
                    self.logger.debug(f"Found {key} match: '{value_str}'")
                    # Strip any trailing periods that might have been captured
                    value_str = value_str.rstrip('.')
                    try:
                        value = float(value_str)
                        params[key] = value
                        self.logger.debug(f"Successfully converted {key}: {value}")
                    except ValueError as e:
                        self.logger.error(f"Could not convert {key} value '{value_str}' to float: {e}")
                        raise
                else:
                    self.logger.warning(f"Could not find {key} in filename: {name_to_parse}")
            except Exception as e:
                self.logger.error(f"Error parsing {key} from filename: {str(e)}")
                raise
                
        # Verify we got all required parameters
        required_params = patterns.keys()
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
            
        return params
        
        params = {}
        for key, pattern in patterns.items():
            try:
                match = re.search(pattern, filename)
                if match:
                    value_str = match.group(1)
                    try:
                        value = float(value_str)
                        params[key] = value
                        self.logger.debug(f"Parsed {key}: {value}")
                    except ValueError:
                        self.logger.error(f"Could not convert {key} value '{value_str}' to float")
                        raise
                else:
                    self.logger.warning(f"Could not find {key} in filename: {filename}")
            except Exception as e:
                self.logger.error(f"Error parsing {key} from filename: {str(e)}")
                raise
                
        # Verify we got all required parameters
        required_params = patterns.keys()
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
            
        return params

    def is_run_in_database(self, metadata: Dict, measurements: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Check if a run with identical metadata and measurements exists in database.
        
        Parameters:
        -----------
        metadata : dict
            Run metadata
        measurements : numpy.ndarray
            Run measurements
            
        Returns:
        --------
        tuple
            (bool: whether run exists, Optional[int]: existing run_id if found)
        """
        # Get all existing runs
        existing_runs = self.get_runs()
        
        for _, run in existing_runs.iterrows():
            run_measurements = self.get_measurements(run['run_id'])['data_point'].values
            
            identical, _ = self.validator.are_runs_identical(
                metadata,
                measurements,
                run.to_dict(),
                run_measurements
            )
            
            if identical:
                return True, run['run_id']
                
        return False, None

    def import_data_file(self, filepath):
        """Enhanced version of import_data_file with validation"""
        filepath = Path(filepath)
        
        # Validate filename format
        if not self.validator.is_valid_filename(filepath.name):
            self.logger.warning(f"Skipping file with invalid format: {filepath.name}")
            return None
            
        # Parse parameters and read data
        params = self.parse_filename(filepath.name)
        data = pd.read_csv(filepath, header=None, sep='\t').iloc[0].values
        
        # Check if run already exists
        exists, existing_run_id = self.is_run_in_database(params, data)
        if exists:
            self.logger.info(f"Skipping duplicate run (existing run_id: {existing_run_id})")
            return existing_run_id
        
    def import_data_file(self, filepath):
        """
        Import a single data file into the database.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the data file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.logger.info(f"Importing data from {filepath}")
        
        # Parse parameters from filename
        params = self.parse_filename(filepath.name)
        
        # Read the data points
        data = pd.read_csv(filepath, header=None, sep='\t').iloc[0]
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert run metadata
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO runs 
                (filename, hz, row_number, r_amplitude, r_width, s_amplitude, 
                 s_width, field, source, repeats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filepath.name, params['hz'], params['row'], 
                params['r_amplitude'], params['r_width'],
                params['s_amplitude'], params['s_width'],
                params['field'], params['source'], params['repeats']
            ))
            
            run_id = cursor.lastrowid
            
            # Insert measurements
            measurements = [(run_id, float(value), idx) 
                          for idx, value in enumerate(data)]
            
            conn.executemany('''
                INSERT INTO measurements (run_id, data_point, point_index)
                VALUES (?, ?, ?)
            ''', measurements)
            
        self.logger.info(f"Successfully imported run {run_id}")
        return run_id
        
    def import_directory(self, directory=None):
        """
        Import all .txt files from a directory.
        
        Parameters:
        -----------
        directory : str or Path, optional
            Directory to import from. If None, uses self.data_dir
        """
        directory = Path(directory) if directory else self.data_dir
        
        self.logger.info(f"Importing all data files from {directory}")
        imported_runs = []
        
        for filepath in directory.glob('*.txt'):
            try:
                run_id = self.import_data_file(filepath)
                imported_runs.append(run_id)
            except Exception as e:
                self.logger.error(f"Error importing {filepath}: {str(e)}")
        
        return imported_runs
    
    def get_runs(self):
        """Get all experimental runs as a pandas DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM runs", conn)
    
    def get_measurements(self, run_id=None):
        """
        Get measurements as a pandas DataFrame.
        If run_id is specified, only get measurements for that run.
        """
        with sqlite3.connect(self.db_path) as conn:
            if run_id is not None:
                query = "SELECT * FROM measurements WHERE run_id = ?"
                return pd.read_sql_query(query, conn, params=[run_id])
            else:
                return pd.read_sql_query("SELECT * FROM measurements", conn)
    
    def get_run_with_measurements(self, run_id):
        """Get a single run with all its measurements as a wide-format DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT r.*, m.data_point, m.point_index
            FROM runs r
            JOIN measurements m ON r.run_id = m.run_id
            WHERE r.run_id = ?
            """
            # Get the data
            df = pd.read_sql_query(query, conn, params=[run_id])
            
            # Pivot to wide format
            df_wide = df.pivot(index='run_id', 
                             columns='point_index', 
                             values='data_point')
            
            # Add back the run metadata
            metadata = df[['run_id', 'filename', 'hz', 'row_number', 
                         'r_amplitude', 'r_width', 's_amplitude', 
                         's_width', 'field', 'source', 'repeats']].iloc[0]
            
            return pd.concat([metadata.to_frame().T, df_wide], axis=1)

# Example usage
if __name__ == "__main__":
    # Initialize data manager
    dm = ExperimentalDataManager()
    
    # Import all files from sample_data directory
    dm.import_directory()
    
    # Get all runs
    runs = dm.get_runs()
    print("\nAvailable runs:")
    print(runs)
    
    if not runs.empty:
        # Get first run with measurements
        first_run_id = runs['run_id'].iloc[0]
        run_data = dm.get_run_with_measurements(first_run_id)
        print("\nFirst run data:")
        print(run_data)
