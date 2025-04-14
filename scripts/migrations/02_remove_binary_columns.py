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

def remove_binary_columns(db_path, parquet_dir):
    """Remove binary columns from parquet files and update database schema"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting binary column removal migration")
    
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
            df_clean = df.drop(columns=binary_columns)
            
            # Write back to parquet
            df_clean.to_parquet(parquet_path, engine='pyarrow', index=False)
            logger.info(f"Removed binary columns from {parquet_file}")
            
        except Exception as e:
            logger.error(f"Error processing {parquet_file}: {e}")
    
    # Step 2: Update database schema
    logger.info("\nSTEP 2: Updating database schema")
    
    # Check which columns exist
    has_binary_column = check_column_exists(conn, "runs", "parquet_binary_column")
    has_analog_column = check_column_exists(conn, "runs", "parquet_analog_column")
    has_parquet_column = check_column_exists(conn, "runs", "parquet_column")
    
    # Handle column renaming/dropping logic
    columns_to_remove = []
    
    # Always remove binary column
    if has_binary_column:
        logger.info("Setting all parquet_binary_column values to NULL")
        cursor.execute("UPDATE runs SET parquet_binary_column = NULL")
        conn.commit()
        columns_to_remove.append("parquet_binary_column")
    
    # Check if both parquet_column and parquet_analog_column exist
    if has_parquet_column and has_analog_column:
        logger.info("Both parquet_column and parquet_analog_column exist. Checking for data to merge...")
        
        # First check if there are NULL values in parquet_column that have data in analog_column
        cursor.execute("""
            SELECT COUNT(*) FROM runs 
            WHERE parquet_column IS NULL AND parquet_analog_column IS NOT NULL
        """)
        null_column_count = cursor.fetchone()[0]
        
        if null_column_count > 0:
            logger.info(f"Found {null_column_count} rows where parquet_column is NULL but analog_column has data")
            logger.info("Copying these values from analog column to parquet_column")
            
            # Copy missing values from analog to parquet_column
            cursor.execute("""
                UPDATE runs 
                SET parquet_column = parquet_analog_column 
                WHERE parquet_column IS NULL AND parquet_analog_column IS NOT NULL
            """)
            conn.commit()
        
        # Now check if there are any remaining mismatches
        cursor.execute("""
            SELECT COUNT(*) FROM runs 
            WHERE parquet_column IS NOT NULL AND parquet_analog_column IS NOT NULL
            AND parquet_column != parquet_analog_column
        """)
        mismatch_count = cursor.fetchone()[0]
        
        if mismatch_count > 0:
            logger.warning(f"Found {mismatch_count} rows where values don't match even after merging!")
            logger.warning("This suggests a change in run_id or naming method.")
            logger.warning("Keeping both columns for manual review.")
        else:
            # Check if there are any remaining cases where analog has data but parquet doesn't
            cursor.execute("""
                SELECT COUNT(*) FROM runs 
                WHERE parquet_column IS NULL AND parquet_analog_column IS NOT NULL
            """)
            remaining_null_count = cursor.fetchone()[0]
            
            if remaining_null_count > 0:
                logger.warning(f"Still found {remaining_null_count} problematic rows after attempting merge")
                logger.warning("Keeping both columns for manual review.")
            else:
                logger.info("All values match or have been merged. Will remove parquet_analog_column.")
                columns_to_remove.append("parquet_analog_column")
    
    # If parquet_column doesn't exist but analog_column does, create and copy
    elif not has_parquet_column and has_analog_column:
        logger.info("Creating parquet_column and copying data from parquet_analog_column")
        cursor.execute("ALTER TABLE runs ADD COLUMN parquet_column TEXT")
        cursor.execute("UPDATE runs SET parquet_column = parquet_analog_column")
        conn.commit()
        
        # After copying, we can remove the analog column
        columns_to_remove.append("parquet_analog_column")
    
    # Now remove columns if needed
    if columns_to_remove:
        try:
            # Start a transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Get current table schema without the columns to remove
            cursor.execute("PRAGMA table_info(runs)")
            columns = [info[1] for info in cursor.fetchall() if info[1] not in columns_to_remove]
            columns_sql = ", ".join(columns)
            
            logger.info(f"Creating new table without columns: {', '.join(columns_to_remove)}")
            
            # Create a new table without unwanted columns
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
        logger.info("No columns need to be removed from the database schema")
    
    # Close database connection
    conn.close()
    
    logger.info("Migration completed")

def main():
    parser = argparse.ArgumentParser(description="Remove binary columns from parquet files and update schema")
    parser.add_argument("--db_path", default="database/mtj_switching_data_parquet.db", 
                      help="Path to SQLite database")
    parser.add_argument("--parquet_dir", default="database/parquet_data", 
                      help="Directory containing parquet files")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    remove_binary_columns(args.db_path, args.parquet_dir)

if __name__ == "__main__":
    main()