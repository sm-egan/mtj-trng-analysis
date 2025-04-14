# scripts/migrations/remove_binary_columns.py
import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import sqlite3
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
            logging.FileHandler('remove_binary_columns.log')
        ]
    )
    return logging.getLogger(__name__)

def check_column_exists(conn, table, column):
    """Check if a column exists in a table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [info[1] for info in cursor.fetchall()]
    return column in columns

def remove_binary_columns(db_path, parquet_dir, dry_run=True):
    """Remove binary columns from parquet files and update database schema"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting binary column removal migration (dry run: {dry_run})")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Step 1: Process parquet files to remove binary columns
    logger.info("STEP 1: Removing binary columns from parquet files")
    
    # Get all parquet files
    parquet_files = set()
    cursor.execute("SELECT DISTINCT parquet_file FROM runs WHERE parquet_file IS NOT NULL")
    for (parquet_file,) in cursor.fetchall():
        parquet_files.add(parquet_file)
    
    logger.info(f"Found {len(parquet_files)} parquet files to process")
    
    for parquet_file in parquet_files:
        parquet_path = Path(parquet_dir) / parquet_file
        if not parquet_path.exists():
            logger.warning(f"Parquet file not found: {parquet_path}")
            continue
        
        logger.info(f"Processing {parquet_file}")
        
        # Read the parquet file
        try:
            df = pd.read_parquet(parquet_path)
            
            # Identify binary columns (those starting with "binary_")
            binary_columns = [col for col in df.columns if col.startswith("binary_")]
            
            if not binary_columns:
                logger.info(f"No binary columns found in {parquet_file}")
                continue
            
            logger.info(f"Found {len(binary_columns)} binary columns in {parquet_file}")
            
            # Remove binary columns
            if not dry_run:
                df_clean = df.drop(columns=binary_columns)
                
                # Write back to parquet
                df_clean.to_parquet(parquet_path, engine='pyarrow', index=False)
                logger.info(f"Removed binary columns from {parquet_file}")
            else:
                logger.info(f"Would remove: {', '.join(binary_columns)} (dry run)")
            
        except Exception as e:
            logger.error(f"Error processing {parquet_file}: {e}")
    
    # Step 2: Update database schema
    logger.info("\nSTEP 2: Updating database schema")
    
    # Check if parquet_binary_column exists
    has_binary_column = check_column_exists(conn, "runs", "parquet_binary_column")
    has_analog_column = check_column_exists(conn, "runs", "parquet_analog_column")
    has_parquet_column = check_column_exists(conn, "runs", "parquet_column")
    
    # Verify we have correct columns to start with
    if not has_binary_column:
        logger.warning("parquet_binary_column doesn't exist in runs table, skipping its removal")
    
    if not has_analog_column:
        logger.warning("parquet_analog_column doesn't exist in runs table, can't rename it")
    
    if has_parquet_column:
        logger.warning("parquet_column already exists in runs table, can't rename to it")
    
    if not dry_run:
        try:
            # Start a transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # 1. Copy data from parquet_analog_column to parquet_column if needed
            if has_analog_column and not has_parquet_column:
                logger.info("Creating parquet_column and copying data from parquet_analog_column")
                cursor.execute("ALTER TABLE runs ADD COLUMN parquet_column TEXT")
                cursor.execute("UPDATE runs SET parquet_column = parquet_analog_column")
            
            # 2. Drop parquet_binary_column
            if has_binary_column:
                logger.info("Dropping parquet_binary_column")
                # SQLite doesn't directly support dropping columns before version 3.35.0
                # So we need to create a new table without that column
                
                # Get current table schema without the binary column
                cursor.execute("PRAGMA table_info(runs)")
                columns = [info[1] for info in cursor.fetchall() if info[1] != "parquet_binary_column"]
                columns_sql = ", ".join(columns)
                
                # Create a new table without binary column
                cursor.execute(f"""
                    CREATE TABLE runs_new (
                        {columns_sql},
                        PRIMARY KEY (run_id)
                    )
                """)
                
                # Copy data
                cursor.execute(f"INSERT INTO runs_new SELECT {columns_sql} FROM runs")
                
                # Replace old table
                cursor.execute("DROP TABLE runs")
                cursor.execute("ALTER TABLE runs_new RENAME TO runs")
            
            # 3. Drop parquet_analog_column if we created parquet_column
            if has_analog_column and not has_parquet_column:
                logger.info("Dropping parquet_analog_column")
                # Get current table schema without the analog column
                cursor.execute("PRAGMA table_info(runs)")
                columns = [info[1] for info in cursor.fetchall() if info[1] != "parquet_analog_column"]
                columns_sql = ", ".join(columns)
                
                # Create a new table without analog column
                cursor.execute(f"""
                    CREATE TABLE runs_new (
                        {columns_sql},
                        PRIMARY KEY (run_id)
                    )
                """)
                
                # Copy data
                cursor.execute(f"INSERT INTO runs_new SELECT {columns_sql} FROM runs")
                
                # Replace old table
                cursor.execute("DROP TABLE runs")
                cursor.execute("ALTER TABLE runs_new RENAME TO runs")
            
            # Commit the transaction
            conn.commit()
            logger.info("Database schema updated successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating database schema: {e}")
    else:
        logger.info("Would update database schema (dry run)")
        if has_binary_column:
            logger.info("Would drop parquet_binary_column")
        if has_analog_column and not has_parquet_column:
            logger.info("Would rename parquet_analog_column to parquet_column")
    
    # Close database connection
    conn.close()
    
    logger.info("Migration completed")

def main():
    parser = argparse.ArgumentParser(description="Remove binary columns from parquet files and update schema")
    parser.add_argument("--db_path", default="database/mtj_switching_data_parquet.db", 
                      help="Path to SQLite database")
    parser.add_argument("--parquet_dir", default="database/parquet_data", 
                      help="Directory containing parquet files")
    parser.add_argument("--execute", action="store_true", 
                      help="Execute the migration (without this flag, it's a dry run)")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if not args.execute:
        logger.info("Running in DRY RUN mode. No changes will be made.")
        logger.info("Use --execute to actually perform the migration.")
    
    remove_binary_columns(args.db_path, args.parquet_dir, dry_run=not args.execute)

if __name__ == "__main__":
    main()