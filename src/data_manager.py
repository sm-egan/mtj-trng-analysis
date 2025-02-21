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
    try:
        if abs(value) < 1e-6 or abs(value) > 1e4:
            return f"{value:.2e}".replace('e-0', 'e-').replace('e+0', 'e')
        else:
            return f"{value:.3f}"
    except Exception as e:
        # If formatting fails, return simple string conversion
        return str(value)

class ExperimentalDataManager:
    def __init__(self, db_path='database/mtj_switching_data.db', data_dir='sample_data'):
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
        logging.basicConfig(level=logging.DEBUG,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.validator = DataValidator()
        
        # Initialize database
        self.initialize_database()

    def initialize_database(self):
        """Create or update the database schema."""
        self.logger.info("Initializing database...")
        
        with sqlite3.connect(self.db_path) as conn:
            # First, check if the runs table exists and its current structure
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check if run_hash column exists
                '''
                Don't think this block is necessary anymore. Double check and maybe remove
                '''
                cursor.execute("PRAGMA table_info(runs)")
                columns = [info[1] for info in cursor.fetchall()]
                
                if 'run_hash' not in columns:
                    self.logger.info("Updating database schema to include run_hash...")
                    # Add the run_hash column
                    conn.execute("ALTER TABLE runs ADD COLUMN run_hash TEXT UNIQUE")
            else:
                # Create new table with current schema
                conn.execute('''
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY,
                        run_hash TEXT UNIQUE,
                        device_id TEXT,
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
                
            # Always ensure measurements table exists
            conn.execute('''
                CREATE TABLE IF NOT EXISTS measurements (
                    run_id TEXT,
                    device_voltage REAL,
                    #measurement_index INTEGER,
                    row_index INTEGER,
                    col_index INTEGER,
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

        patterns = {
            'hz': r'Hz_(\d+)',
            'row': r'Row\s*(\d+)',
            'r_amplitude': r'RAmp([+-]?[\d.E+-]+)',  # Modified to handle +/- in any position
            'r_width': r'RWid([\d.E+-]+)',
            's_amplitude': r'SAmp([+-]?[\d.E+-]+)',  # Also fix this for consistency
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
        run_components = [
            f"D{params.get('device_id')}" if params.get('device_id') else "",
            f"SW{format_scientific(params['s_width'])}",
            f"SA{format_scientific(params['s_amplitude'])}",
            f"F{params['field']:.1f}"
        ]
        base_run_id = "_".join(x for x in run_components if x)
        self.logger.debug(f"Generated base_run_id: {base_run_id}")
        
        # Get run identity hash
        run_identity = self.validator.compute_run_identity(params, measurements)
        if not run_identity.is_valid:
            raise ValueError(f"Invalid run data: {run_identity.validation_message}")
        
        self.logger.debug(f"Generated initial hash: {run_identity.full_hash}")
        
        # Keep trying different hash lengths if we get a collision
        hash_length = 6
        while True:
            proposed_run_id = f"{base_run_id}_{run_identity.full_hash[:hash_length]}"
            self.logger.debug(f"Trying proposed_run_id: {proposed_run_id}")
            
            # Check if this ID exists with different data
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT run_hash FROM runs WHERE run_id = ?", 
                    (proposed_run_id,)
                ).fetchone()
                
                self.logger.debug(f"Database query result: {existing}")
                
                if not existing:
                    self.logger.debug("No existing run_id found")
                    break
                elif existing[0] == run_identity.full_hash:
                    self.logger.debug("Found exact match")
                    break
                else:
                    self.logger.debug(f"Found collision, extending hash length to {hash_length + 1}")
                    hash_length += 1
                    if hash_length > 12:  # Safeguard against infinite loop
                        raise ValueError("Unable to generate unique run_id")
        
        self.logger.debug(f"Final run_id: {proposed_run_id}")
        return proposed_run_id, run_identity.full_hash

    def import_data_file(self, filepath):
        """Enhanced version of import_data_file with hash-based IDs"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Get device ID from parent folder name if it exists
        device_id = filepath.parent.name.split("Device_")[1] if filepath.parent.name.startswith('Device_') else None

        # Validate filename format
        if not self.validator.is_valid_filename(filepath.name):
            self.logger.warning(f"Skipping file with invalid format: {filepath.name}")
            return None
        
        # Parse parameters from filename
        params = self.parse_filename(filepath.name)
        if device_id:
            params['device_id'] = device_id
        
        # Read the data points
        #data = np.loadtxt(filepath, delimiter='\t')
        # try:
        # # First try reading with numpy
        #     data = np.loadtxt(filepath, delimiter='\t')
        # except ValueError as e:
        #     # If that fails, try pandas which is more forgiving
        #     try:
        #         df = pd.read_csv(filepath, delimiter='\t', header=None)
        #         # Get first row that has the expected number of columns
        #         max_cols = df.shape[1]
        #         valid_data = df[df.notna().sum(axis=1) == max_cols]
        #         if len(valid_data) > 0:
        #             data = valid_data.iloc[0].values
        #         else:
        #             raise ValueError("No valid data rows found")
        #     except Exception as e2:
        #         self.logger.error(f"Error reading data from {filepath}: {str(e2)}")
        #         raise
        # Read the data points with debug logging
        try:
            self.logger.debug(f"Attempting numpy.loadtxt for {filepath.name}")
            data = np.loadtxt(filepath, delimiter='\t')
            self.logger.debug(f"Successfully used numpy.loadtxt. Data type: {data.dtype}, shape: {data.shape}")
        except ValueError as e:
            self.logger.debug(f"numpy.loadtxt failed with error: {str(e)}")
            try:
                self.logger.debug(f"Attempting pandas read for {filepath.name}")
                df = pd.read_csv(filepath, delimiter='\t', header=None)
                max_cols = df.shape[1]
                valid_data = df[df.notna().sum(axis=1) == max_cols]
                if len(valid_data) > 0:
                    data = valid_data.iloc[0].values
                    self.logger.debug(f"Successfully used pandas. Data type: {data.dtype}, shape: {data.shape}")
                else:
                    raise ValueError("No valid data rows found")
            except Exception as e2:
                self.logger.error(f"Both reading methods failed for {filepath.name}")
                raise

        # Add debug logging before run ID generation
        self.logger.debug(f"Data sample for {filepath.name}: {data[:5]}")
        
        # Generate run ID and hash
        try:
            run_id, run_hash = self.generate_run_id(params, data)
        except ValueError as e:
            self.logger.error(f"Run ID generation failed for {filepath.name}")
            self.logger.error(f"Data type at failure: {type(data)}, dtype: {data.dtype}")
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
        
        # Insert new run with optimized database operations
        with sqlite3.connect(self.db_path) as conn:
            # Enable faster inserts
            conn.execute('PRAGMA synchronous = OFF')
            conn.execute('PRAGMA journal_mode = WAL')
            
            cursor = conn.cursor()
            
            # Begin transaction explicitly
            cursor.execute('BEGIN TRANSACTION')
            try:
                # Insert run metadata
                cursor.execute('''
                    INSERT INTO runs 
                    (run_id, run_hash, device_id, filename, hz, row_number, r_amplitude, r_width,
                     s_amplitude, s_width, field, source, repeats)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id, run_hash, 
                    params.get('device_id'),
                    filepath.name,
                    params['hz'], params['row'],
                    params['r_amplitude'], params['r_width'],
                    params['s_amplitude'], params['s_width'],
                    params['field'], params['source'], params['repeats']
                ))
                
                # Insert measurements in chunks
                CHUNK_SIZE = 100000  # Process 100k rows at a time
                for i in range(0, len(data), CHUNK_SIZE):
                    chunk = data[i:i + CHUNK_SIZE]
                    measurements = [(run_id, float(value), idx + i) 
                                  for idx, value in enumerate(chunk)]
                    
                    cursor.executemany('''
                        INSERT INTO measurements (run_id, device_voltage, measurement_index)
                        VALUES (?, ?, ?)
                    ''', measurements)
                
                # Commit the transaction
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                raise e
        
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
            # First get the run metadata
            run_query = "SELECT * FROM runs WHERE run_id = ?"
            run_df = pd.read_sql_query(run_query, conn, params=[run_id])
            
            if run_df.empty:
                return None
                
            # Then get the measurements
            measurements_query = """
            SELECT measurement_index, device_voltage
            FROM measurements 
            WHERE run_id = ?
            ORDER BY measurement_index
            """
            measurements_df = pd.read_sql_query(measurements_query, conn, params=[run_id])
            
            # Convert measurements to wide format
            if not measurements_df.empty:
                # Create measurement columns with prefix 'point_'
                measurements_wide = pd.DataFrame(
                    [measurements_df['device_voltage'].values],
                    columns=[f'measurement_{i}' for i in measurements_df['measurement_index']]
                )
                
                # Combine run metadata with measurements
                result = pd.concat([run_df, measurements_wide], axis=1)
                return result
            
            return run_df

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
# if __name__ == "__main__":
#     # Initialize data manager
#     dm = ExperimentalDataManager()
    
#     # Import all files from sample_data directory
#     dm.import_directory()
    
#     # Get all runs
#     runs = dm.get_runs()
#     print("\nAvailable runs:")
#     print(runs)
    
#     if not runs.empty:
#         # Get first run with measurements
#         first_run_id = runs['run_id'].iloc[0]
#         run_data = dm.get_run_with_measurements(first_run_id)
#         print("\nFirst run data:")
#         print(run_data)
if __name__ == "__main__":
    # Initialize data manager
    dm = ExperimentalDataManager()

    # Import all files from sample_data directory
    dm.import_directory()
    
    # Get all runs
    runs = dm.get_runs()
    print("\nAvailable runs:")
    # Display key columns in a readable format
    if not runs.empty:
        display_cols = ['run_id', 'filename', 's_width', 's_amplitude', 'field', 'repeats']
        print(runs[display_cols].to_string())
        print(f"\nTotal runs: {len(runs)}")
        
        # Get first run with measurements
        first_run_id = runs['run_id'].iloc[0]
        run_data = dm.get_run_with_measurements(first_run_id)
        
        if run_data is not None:
            print("\nFirst run metadata:")
            # Display metadata columns
            metadata_cols = [col for col in run_data.columns if not col.startswith('measurement_')]
            print(run_data[metadata_cols].to_string())
            
            print("\nMeasurement summary:")
            measurement_cols = [col for col in run_data.columns if col.startswith('measurement_')]
            measurements = run_data[measurement_cols].iloc[0]
            print(f"Number of measurements: {len(measurements)}")