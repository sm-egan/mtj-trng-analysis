from google.oauth2 import service_account
from googleapiclient.discovery import build
import os

# The required scopes for accessing Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_with_service_account():
    """Authenticate using the service account credentials"""
    credentials = service_account.Credentials.from_service_account_file(
        'service_account.json', 
        scopes=SCOPES
    )
    return credentials

def list_files_in_folder(service, folder_id):
    """List files in the specified folder within a shared drive"""
    try:
        # Modified query to explicitly look in shared drives
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=10,
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora='allDrives',  # This tells the API to look in shared drives
            driveId=folder_id     # Specify which drive to search in
        ).execute()
        
        items = results.get('files', [])
        if items:
            print("\nFound files:")
            for item in items:
                print(f"- {item['name']} ({item['mimeType']})")
        else:
            print("\nNo files found in this folder")
        
        return items
    except Exception as e:
        print(f"\nError listing files: {str(e)}")
        return []

def check_permissions(service, folder_id):
    try:
        permissions = service.permissions().list(
            fileId=folder_id,
            supportsAllDrives=True,
            fields='permissions(emailAddress, role, type)'
        ).execute()
        print("\nCurrent permissions:")
        for perm in permissions.get('permissions', []):
            print(f"- {perm.get('emailAddress', 'N/A')}: {perm.get('role')} ({perm.get('type')})")
    except Exception as e:
        print(f"Error checking permissions: {str(e)}")

def main():
    creds = authenticate_with_service_account()
    service = build('drive', 'v3', credentials=creds)
    
    folder_id = '10oVxCZ40AVmH75EywvSp-8yKrC6MegD4'  # Your Data folder ID
    print(f"Attempting to access folder ID: {folder_id}")
    
    try:
        folder = service.files().get(
            fileId=folder_id,
            supportsAllDrives=True,
            fields='name, owners'
        ).execute()
        print(f"Successfully accessed folder: {folder.get('name')}")
        print(f"Folder owner: {folder.get('owners')[0].get('emailAddress')}")
    except Exception as e:
        print(f"Error accessing folder: {str(e)}")
    
    check_permissions(service, folder_id)
    list_files_in_folder(service, folder_id)
if __name__ == '__main__':
    main()