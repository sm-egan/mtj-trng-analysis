import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.parquet_data_manager import ParquetExperimentalDataManager

def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('compute_binary_and_switching.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Compute binary and switching probabilities")
    parser.add_argument("--db_path", default="database/mtj_switching_data_parquet.db", help="Path to SQLite database")
    parser.add_argument("--parquet_dir", default="database/parquet_data", help="Path to Parquet files directory")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if data exists")
    parser.add_argument("--device", help="Process only the specified device ID")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip_binary", action="store_true", help="Skip binary data computation")
    parser.add_argument("--skip_switching", action="store_true", help="Skip switching probability computation")
    
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Starting data computation pipeline")
    
    try:
        # Initialize data manager
        data_manager = ParquetExperimentalDataManager(
            db_path=args.db_path, 
            parquet_dir=args.parquet_dir,
            logging_level="DEBUG" if args.verbose else "INFO"
        )
        
        # Compute binary data
        if not args.skip_binary:
            logger.info("Step 1: Computing binary data")
            data_manager.write_binary_data(
                force_recalculate=args.force,
                device_id=args.device
            )
            logger.info("Binary data computation completed successfully")
        else:
            logger.info("Skipping binary data computation")
        
        # Compute switching probabilities
        if not args.skip_switching:
            logger.info("Step 2: Computing switching probabilities")
            data_manager.write_switching_probabilities(
                force_recalculate=args.force,
                device_id=args.device
            )
            logger.info("Switching probability computation completed successfully")
        else:
            logger.info("Skipping switching probability computation")
        
        logger.info("All operations completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during computation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())