# scripts/migrations/01_add_devices_and_binary_data.py

import sqlite3
import argparse
from pathlib import Path
import logging

def check_if_column_exists(conn, table, column):
    """Check if a column exists in a table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [info[1] for info in cursor.fetchall()]
    return column in columns

def migrate_database(db_path):
    """
    Add devices table, switching_probability, and binary data column tracking to runs
    
    Args:
        db_path: Path to SQLite database
        dry_run: If True, print SQL but don't execute
    """
        
    # Connect to database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if devices table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='devices'")
        if not cursor.fetchone():
            print("Creating devices table...")
            conn.execute("""
            CREATE TABLE devices (
                device_id TEXT PRIMARY KEY,
                threshold REAL,
                peak_p REAL,
                peak_ap REAL,
                tmr REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
        else:
            print("devices table already exists")
        
        # Check and add switching_probability
        if not check_if_column_exists(conn, "runs", "switching_probability"):
            print("Adding switching_probability column...")
            conn.execute("ALTER TABLE runs ADD COLUMN switching_probability REAL")
        else:
            print("switching_probability column already exists")
        
        # Check if we need to rename parquet_column
        if check_if_column_exists(conn, "runs", "parquet_column"):
            print("Renaming parquet_column to parquet_analog_column...")
            conn.execute("ALTER TABLE runs RENAME COLUMN parquet_column TO parquet_analog_column")
        elif not check_if_column_exists(conn, "runs", "parquet_analog_column"):
            print("Neither parquet_column nor parquet_analog_column exist, skipping rename")
        else:
            print("Column already renamed to parquet_analog_column")
        
        # Check and add parquet_binary_column
        if not check_if_column_exists(conn, "runs", "parquet_binary_column"):
            print("Adding parquet_binary_column...")
            conn.execute("ALTER TABLE runs ADD COLUMN parquet_binary_column TEXT")
        else:
            print("parquet_binary_column already exists")
        
        # Check and add device_id_fk
        if not check_if_column_exists(conn, "runs", "device_id_fk"):
            print("Adding device_id_fk column with reference...")
            conn.execute("""
            ALTER TABLE runs ADD COLUMN device_id_fk TEXT 
            REFERENCES devices(device_id) ON DELETE SET NULL
            """)
            
            # Update existing runs with device_id_fk from device_id
            conn.execute("""
            UPDATE runs SET device_id_fk = device_id
            WHERE device_id IS NOT NULL
            """)
        else:
            print("device_id_fk column already exists")
        
        print(f"Successfully migrated database at {db_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate database schema to add devices table and binary data support")
    parser.add_argument("--db_path", default="database/mtj_switching_data_parquet.db", help="Path to SQLite database")
    args = parser.parse_args()
    
    migrate_database(args.db_path)