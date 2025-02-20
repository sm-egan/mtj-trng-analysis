from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from pathlib import Path
import logging

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

    def download_file(self, file_id, output_dir):
        """Download a specific file"""
        try:
            # Get file metadata
            file = self.service.files().get(
                fileId=file_id,
                supportsAllDrives=True
            ).execute()
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file content
            request = self.service.files().get_media(fileId=file_id)
            file_handle = io.BytesIO()
            downloader = MediaIoBaseDownload(file_handle, request)
            
            done = False
            while not done:
                _, done = downloader.next_chunk()
            
            # Save to file
            output_path = output_dir / file['name']
            with open(output_path, 'wb') as f:
                f.write(file_handle.getvalue())
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            raise