# scripts/import_drive_data.py
import sys
import logging
from tqdm import tqdm
from pathlib import Path
import argparse
import shutil

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from src.drive_manager import GoogleDriveManager
from src.data_manager import ExperimentalDataManager

def main(folder_id, file_pattern=r"Hz_.*\.txt$"):
    """Import experimental data from Google Drive to local database"""

    # Use relative paths from project root
    credentials_path = project_root / 'config' / 'service_account.json'
    temp_dir = project_root / 'temp_downloads'
    db_path = project_root / 'database' / 'mtj_switching_data.db'
    
    drive_manager = GoogleDriveManager(credentials_path)
    data_manager = ExperimentalDataManager(db_path=db_path)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Initialize managers
    drive_manager = GoogleDriveManager(credentials_path)
    data_manager = ExperimentalDataManager()
    
    try:
        # List all files in folder
        #logger.info(f"Listing files in folder {folder_id}")
        
        # Recursively list all matching files
        logger.info(f"Recursively searching for files matching pattern: {file_pattern}")
        files = drive_manager.list_files_recursively(folder_id, file_pattern)
        logger.info(f"Found {len(files)} matching files")
        
        # Create temporary directory for downloads
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        # Process files
        imported_count = 0
        error_count = 0
        
        for file in tqdm(files, desc="Importing files"):
            try:
                # Download file
                file_path = drive_manager.download_file(file['id'], file['folder_path'], temp_dir)
                
                # Import to database
                run_id = data_manager.import_data_file(file_path)
                
                if run_id:
                    imported_count += 1
                    logger.debug(f"Successfully imported {file['name']} as {run_id}")
                
                # Clean up downloaded file
                file_path.unlink()
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {file['name']}: {str(e)}")
                continue
        
        logger.info(f"Import complete. Successfully imported {imported_count} files.")
        if error_count > 0:
            logger.warning(f"Encountered errors with {error_count} files.")
            
    finally:
        # Cleanup using rmtree to handle nested directories
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Successfully cleaned up temporary directory")
            except Exception as e:
                logger.warning(f"Could not remove temporary directory structure: {e}")
                logger.info("You may need to manually delete the temp_downloads directory")

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Import experimental data from Google Drive')
    parser.add_argument('--folder_id', default='10oVxCZ40AVmH75EywvSp-8yKrC6MegD4', help='Google Drive folder ID to import from')
    parser.add_argument('--pattern', default=r"Hz_.*\.txt$", help='Regex pattern to match filenames')
    
    args = parser.parse_args()
    
    main(args.folder_id, args.pattern)