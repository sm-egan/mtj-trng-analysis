import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parents[1]  # Go up one level from tests/
sys.path.insert(0, str(project_root))

from src.data_manager import ExperimentalDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_import():
    """Test the enhanced data manager with hash-based run IDs"""
    # Set up paths relative to project root
    project_root = Path(__file__).parents[1]
    sample_data_dir = project_root / 'sample_data'
    db_path = project_root / 'database' / 'test.db'

    # Initialize data manager
    logger.info(f"Initializing data manager with database at {db_path}")
    dm = ExperimentalDataManager(db_path=db_path, data_dir=sample_data_dir)
    
    # Import sample data file
    sample_files = list(sample_data_dir.glob('*.txt'))
    logger.info(f"Found {len(sample_files)} files in sample_data directory")
    
    # Import each file
    run_ids = []
    for file in sample_files:
        logger.info(f"\nImporting file: {file.name}")
        run_id = dm.import_data_file(file)
        if run_id:
            run_ids.append(run_id)
            logger.info(f"Successfully imported as run_id: {run_id}")
    
    # Verify imports
    logger.info("\nVerifying imports...")
    runs_df = dm.get_runs()
    logger.info(f"\nFound {len(runs_df)} runs in database:")
    for _, run in runs_df.iterrows():
        logger.info(f"Run ID: {run['run_id']}")
        logger.info(f"Hash: {run['run_hash']}")
        logger.info(f"Filename: {run['filename']}")
        logger.info(f"Parameters: SW={run['s_width']}, SA={run['s_amplitude']}, F={run['field']}")
        logger.info("---")
    
    # Verify measurements
    logger.info("\nVerifying measurements...")
    for run_id in run_ids:
        measurements = dm.get_measurements(run_id)
        logger.info(f"Run {run_id} has {len(measurements)} measurements")

if __name__ == "__main__":
    test_data_import()