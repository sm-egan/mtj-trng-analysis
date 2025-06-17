# scripts/migrations/03_delete_bad_d07_08_runs.py

import argparse
import logging
import sqlite3
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('delete_bad_runs.log')
        ]
    )
    return logging.getLogger(__name__)

def identify_bad_runs(db_path):
    """
    Identify runs from device D07_08 with row_number > 1167 or r_amplitude = -1.1
    
    Returns:
        list: List of dictionaries with run information
    """
    logger = logging.getLogger(__name__)
    
    with sqlite3.connect(db_path) as conn:
        query = """
        SELECT run_id, device_id, row_number, r_amplitude, parquet_file, parquet_column, filename
        FROM runs 
        WHERE device_id = '07_08' 
        AND (row_number > 1167 OR r_amplitude = -1.1)
        ORDER BY row_number, r_amplitude
        """
        
        cursor = conn.execute(query)
        runs = []
        
        for row in cursor.fetchall():
            runs.append({
                'run_id': row[0],
                'device_id': row[1], 
                'row_number': row[2],
                'r_amplitude': row[3],
                'parquet_file': row[4],
                'parquet_column': row[5],
                'filename': row[6]
            })
    
    logger.info(f"Found {len(runs)} bad runs to delete")
    return runs

def delete_runs_from_database(db_path, run_ids):
    """Delete run metadata from SQLite database"""
    logger = logging.getLogger(__name__)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Delete from runs table
        for run_id in run_ids:
            cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            logger.info(f"Deleted run {run_id} from database")
        
        conn.commit()
        logger.info(f"Successfully deleted {len(run_ids)} runs from database")

def delete_columns_from_parquet(parquet_dir, runs_info):
    """Delete columns from Parquet files"""
    logger = logging.getLogger(__name__)
    
    # Group runs by parquet file for efficient processing
    files_to_process = {}
    for run in runs_info:
        parquet_file = run['parquet_file']
        parquet_column = run['parquet_column']
        
        if parquet_file not in files_to_process:
            files_to_process[parquet_file] = []
        files_to_process[parquet_file].append({
            'column': parquet_column,
            'run_id': run['run_id']
        })
    
    # Process each parquet file
    for parquet_file, columns_info in files_to_process.items():
        parquet_path = Path(parquet_dir) / parquet_file
        
        if not parquet_path.exists():
            logger.warning(f"Parquet file not found: {parquet_path}")
            continue
            
        logger.info(f"Processing parquet file: {parquet_file}")
        
        try:
            # Read the parquet file
            df = pd.read_parquet(parquet_path)
            
            # Remove the bad columns
            columns_to_remove = [col_info['column'] for col_info in columns_info]
            columns_found = [col for col in columns_to_remove if col in df.columns]
            columns_missing = [col for col in columns_to_remove if col not in df.columns]
            
            if columns_missing:
                logger.warning(f"Columns not found in {parquet_file}: {columns_missing}")
            
            if columns_found:
                # Drop the columns
                df_clean = df.drop(columns=columns_found)
                
                # Write back to parquet
                df_clean.to_parquet(parquet_path, engine='pyarrow', index=False)
                
                for col_info in columns_info:
                    if col_info['column'] in columns_found:
                        logger.info(f"Deleted column {col_info['column']} (run {col_info['run_id']}) from {parquet_file}")
                
                logger.info(f"Updated parquet file {parquet_file} - removed {len(columns_found)} columns")
            else:
                logger.warning(f"No columns to remove from {parquet_file}")
                
        except Exception as e:
            logger.error(f"Error processing parquet file {parquet_file}: {e}")
            raise

def delete_bad_runs(db_path, parquet_dir, dry_run=True, force=False):
    """Main function to delete bad runs"""
    logger = logging.getLogger(__name__)
    
    # Step 1: Identify bad runs
    logger.info("STEP 1: Identifying bad runs")
    bad_runs = identify_bad_runs(db_path)
    
    if not bad_runs:
        logger.info("No bad runs found. Nothing to delete.")
        return
    
    # Display what will be deleted
    logger.info(f"\nFound {len(bad_runs)} runs to delete:")
    logger.info("-" * 80)
    for run in bad_runs:
        logger.info(f"Run ID: {run['run_id']}")
        logger.info(f"  Device: {run['device_id']}")
        logger.info(f"  Row: {run['row_number']}, R_amp: {run['r_amplitude']}")
        logger.info(f"  File: {run['filename']}")
        logger.info(f"  Parquet: {run['parquet_file']} -> {run['parquet_column']}")
        logger.info("")
    
    if dry_run:
        logger.info("DRY RUN MODE: No changes made. Use --execute to perform deletion.")
        return
    
    # Confirmation check
    if not force:
        response = input(f"\nAre you sure you want to delete {len(bad_runs)} runs? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Deletion cancelled by user.")
            return
    
    # Step 2: Delete from database
    logger.info("\nSTEP 2: Deleting from database")
    run_ids = [run['run_id'] for run in bad_runs]
    delete_runs_from_database(db_path, run_ids)
    
    # Step 3: Delete from parquet files
    logger.info("\nSTEP 3: Deleting from parquet files")
    delete_columns_from_parquet(parquet_dir, bad_runs)
    
    logger.info(f"\nMigration completed successfully!")
    logger.info(f"Deleted {len(bad_runs)} runs from device D07_08")
    logger.info("You can now run populate_device_data.py --force --device D07_08 to recalculate thresholds")

def main():
    parser = argparse.ArgumentParser(description="Delete bad runs from device D07_08")
    parser.add_argument("--db_path", default="database/mtj_switching_data_parquet.db", 
                      help="Path to SQLite database")
    parser.add_argument("--parquet_dir", default="database/parquet_data", 
                      help="Directory containing parquet files")
    parser.add_argument("--execute", action="store_true", 
                      help="Actually perform the deletion (default is dry-run)")
    parser.add_argument("--force", action="store_true", 
                      help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Perform the deletion
    delete_bad_runs(
        db_path=args.db_path,
        parquet_dir=args.parquet_dir, 
        dry_run=not args.execute,
        force=args.force
    )

if __name__ == "__main__":
    main()