# setup_test.py
from pathlib import Path
import pandas as pd
import sqlite3
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist"""
    # Get the project root directory (assuming this script is in src/)
    project_root = Path(__file__).parent.parent
    
    # Create directories
    dirs = {
        'sample_data': project_root / 'sample_data',
        'database': project_root / 'database',
    }
    
    for name, path in dirs.items():
        path.mkdir(exist_ok=True)
        logger.info(f"Created/verified {name} directory at: {path}")
    
    return dirs

def test_database_connection(db_path):
    """Test SQLite database connection and create a simple table"""
    try:
        with sqlite3.connect(db_path) as conn:
            # Create a simple test table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            ''')
            # Insert a test value
            conn.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
            
            # Query the test value
            result = conn.execute("SELECT value FROM test").fetchone()
            
            logger.info(f"Successfully connected to database and retrieved: {result[0]}")
            
            # Clean up test table
            conn.execute("DROP TABLE test")
            
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def test_pandas():
    """Test pandas installation and basic functionality"""
    try:
        # Create a simple DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        logger.info("Successfully created pandas DataFrame")
        return True
    except Exception as e:
        logger.error(f"Pandas test failed: {e}")
        return False

def main():
    """Run all setup tests"""
    logger.info("Starting setup tests...")
    
    # Create directories
    dirs = setup_directories()
    
    # Test pandas
    if not test_pandas():
        logger.error("Pandas test failed!")
        sys.exit(1)
    
    # Test database
    db_path = dirs['database'] / 'experiment_data.db'
    if not test_database_connection(db_path):
        logger.error("Database test failed!")
        sys.exit(1)
    
    logger.info("All setup tests completed successfully!")
    return dirs

if __name__ == "__main__":
    main()