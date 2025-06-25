from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from pathlib import Path
import logging

def extract_device_id(folder_path):
    """Extract device ID from folder path string"""
    # Split path into components
    folders = folder_path.split('/')
    
    # Look for device folder
    for folder in folders:
        if folder.startswith('Device_'):
            # Return just the ID portion (e.g., '017_04')
            return folder
    
    return None

class GoogleDriveManager:
    def __init__(self, credentials_path='config/service_account.json'):
        """Initialize the Drive service with credentials"""
        self.logger = logging.getLogger(__name__)
        
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.service = build('drive', 'v3', credentials=credentials)
    
    def list_folder_contents(self, folder_id):
        """List all files in a folder"""
        try:
            results = self.service.files().list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            self.logger.error(f"Error listing folder contents: {str(e)}")
            raise

    def list_files_recursively(self, folder_id, file_pattern=r"Hz_.*\.txt$", include_do_not_import=True) -> list:
        """
        Recursively list all matching files in a folder and its subfolders
        
        Args:
            folder_id: str - The Google Drive folder ID
            file_pattern: str - Regex pattern to match filenames
        
        Returns:
            list of dicts with file info including parent folder names
        """
        import re
        
        def explore_folder(folder_id, current_path=''):
            matching_files = []
            
            try:
                # Handle pagination to get ALL files
                page_token = None
                while True:
                    # Build request with proper pageSize and pagination
                    request_params = {
                        'q': f"'{folder_id}' in parents",
                        'fields': "nextPageToken, files(id, name, mimeType)",
                        'supportsAllDrives': True,
                        'includeItemsFromAllDrives': True,
                        'pageSize': 1000  # Maximum allowed by Google Drive API
                    }
                    
                    # Add page token for subsequent pages
                    if page_token:
                        request_params['pageToken'] = page_token
                    
                    # Make the API call
                    results = self.service.files().list(**request_params).execute()
                    items = results.get('files', [])
                    
                    # Debug logging (you can remove this later)
                    txt_files = [item for item in items if item['name'].endswith('.txt')]
                    self.logger.debug(f"Found {len(items)} items in folder: {current_path or 'root'} (page)")
                    self.logger.debug(f"{len(txt_files)} .txt files in this page")
                    
                    # Process items in this page
                    for item in items:
                        # If it's a folder, recurse
                        if item['mimeType'] == 'application/vnd.google-apps.folder':
                            folder_name = item['name']
                            if not include_do_not_import and folder_name.lower() == 'do_not_import':
                                self.logger.info(f"Skipping folder marked Do_not_import: {current_path}/{folder_name}")
                                continue
                            new_path = f"{current_path}/{folder_name}" if current_path else folder_name
                            matching_files.extend(explore_folder(item['id'], new_path))
                        
                        # If it's a file and matches pattern, add it
                        elif re.match(file_pattern, item['name']):
                            matching_files.append({
                                'id': item['id'],
                                'name': item['name'],
                                'folder_path': current_path,
                                'mimeType': item['mimeType']
                            })
                    
                    # Check if there are more pages
                    page_token = results.get('nextPageToken')
                    if not page_token:
                        break  # No more pages, we're done with this folder
                        
                return matching_files
            
            except Exception as e:
                self.logger.error(f"Error exploring folder {folder_id} (path: {current_path}): {str(e)}")
                return []

        return explore_folder(folder_id)

    def download_file(self, file_id, folder_path, output_dir):
        """Download a specific file"""
        try:
            # Get file metadata
            file = self.service.files().get(
                fileId=file_id,
                supportsAllDrives=True
            ).execute()
            
            # Create device-specific subdirectory if needed
            device_id = extract_device_id(folder_path)
            if device_id:
                output_dir = Path(output_dir) / device_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file content
            request = self.service.files().get_media(fileId=file_id)
            file_handle = io.BytesIO()
            downloader = MediaIoBaseDownload(file_handle, request)
            
            done = False
            while not done:
                _, done = downloader.next_chunk()
            
            # Save to file with original name
            output_path = output_dir / file['name']
            with open(output_path, 'wb') as f:
                f.write(file_handle.getvalue())
                
            return output_path
                
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            raise