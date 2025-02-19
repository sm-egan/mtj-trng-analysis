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

def format_scientific(value):
    """Format scientific notation values in a consistent, readable way"""
    if abs(value) < 1e-6 or abs(value) > 1e4:
        return f"{value:.2e}".replace('e-0', 'e-').replace('e+0', 'e')
    else:
        return f"{value:.3f}"

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
            # Create runs table with new identifier fields
            conn.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,  -- Changed to TEXT for compound identifier
                    run_hash TEXT UNIQUE,     -- Added for data integrity
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
                    import_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS measurements (
                    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,              -- Changed to TEXT to match runs table
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

    def generate_run_id(self, params: Dict, measurements: np.ndarray) -> Tuple[str, str]:
        """
        Generate a compound run identifier and hash.
        
        Parameters:
        -----------
        params : dict
            Dictionary of experimental parameters
        measurements : np.ndarray
            Array of measurement data
            
        Returns:
        --------
        tuple
            (run_id, run_hash)
        """
        # Create a deterministic but human-readable run ID
        run_components = [
            f"SW{format_scientific(params['s_width'])}",
            f"SA{format_scientific(params['s_amplitude'])}",
            f"F{format_scientific(params['field'])}"
            #datetime.now().strftime("%Y%m%d")
        ]
        base_run_id = "_".join(run_components)
        
        # Get run identity from validator
        run_identity = self.validator.compute_run_identity(params, measurements)
        if not run_identity.is_valid:
            raise ValueError(f"Invalid run data: {run_identity.validation_message}")
        
        # Keep trying different hash lengths if we get a collision
        hash_length = 6
        while True:
            proposed_run_id = f"{base_run_id}_{run_identity.full_hash[:hash_length]}"
            # Check if this ID already exists
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", 
                    (proposed_run_id,)
                ).fetchone()
                if not existing:
                    break
                hash_length += 1
                if hash_length > 12:  # Safeguard against infinite loop
                    raise ValueError("Unable to generate unique run_id")
        
        return proposed_run_id, run_identity.full_hash

    def import_data_file(self, filepath):
        """Enhanced version of import_data_file with hash-based IDs"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Validate filename format
        if not self.validator.is_valid_filename(filepath.name):
            self.logger.warning(f"Skipping file with invalid format: {filepath.name}")
            return None
        
        # Parse parameters from filename
        params = self.parse_filename(filepath.name)
        
        # Read the data points
        data = pd.read_csv(filepath, header=None, sep='\t').iloc[0].values
        
        # Generate run ID and hash
        try:
            run_id, run_hash = self.generate_run_id(params, data)
        except ValueError as e:
            self.logger.error(f"Error generating run ID: {e}")
            return None
        
        # Check if run hash already exists
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT run_id FROM runs WHERE run_hash = ?", 
                (run_hash,)
            ).fetchone()
            
            if existing:
                self.logger.info(f"Skipping duplicate run (existing run_id: {existing[0]})")
                return existing[0]
        
        # Insert new run
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert run metadata
            cursor.execute('''
                INSERT INTO runs 
                (run_id, run_hash, filename, hz, row_number, r_amplitude, r_width,
                 s_amplitude, s_width, field, source, repeats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, run_hash, filepath.name,
                params['hz'], params['row'],
                params['r_amplitude'], params['r_width'],
                params['s_amplitude'], params['s_width'],
                params['field'], params['source'], params['repeats']
            ))
            
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

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a specific run and its associated measurements from the database.
        
        Parameters:
        -----------
        run_id : str
            The run_id to delete
            
        Returns:
        --------
        bool
            True if run was found and deleted, False otherwise
        """
        self.logger.info(f"Attempting to delete run {run_id}")
        
        with sqlite3.connect(self.db_path) as conn:
            # First verify the run exists
            exists = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", 
                (run_id,)
            ).fetchone()
            
            if not exists:
                self.logger.warning(f"Run {run_id} not found in database")
                return False
            
            # Delete associated measurements first (due to foreign key constraint)
            conn.execute("DELETE FROM measurements WHERE run_id = ?", (run_id,))
            
            # Then delete the run itself
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            
            self.logger.info(f"Successfully deleted run {run_id} and its measurements")
            return True

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
