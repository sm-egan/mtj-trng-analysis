from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

def verify_service_account_credentials():
    """Check if we can properly load and read the service account credentials"""
    try:
        print("Step 1: Verifying service account credentials...")
        with open('service_account.json', 'r') as f:
            creds_data = json.load(f)
        
        print(f"✓ Service account email: {creds_data.get('client_email')}")
        print(f"✓ Project ID: {creds_data.get('project_id')}")
        return True
    except FileNotFoundError:
        print("✗ Error: service_account.json file not found")
        return False
    except json.JSONDecodeError:
        print("✗ Error: service_account.json is not valid JSON")
        return False
    except Exception as e:
        print(f"✗ Unexpected error reading credentials: {str(e)}")
        return False

def verify_drive_service(service):
    """Test if we can make a basic API call to Google Drive"""
    try:
        print("\nStep 2: Testing Drive API connection...")
        # Try to get the about information - a basic API call
        about = service.about().get(fields="user").execute()
        print(f"✓ Successfully connected to Drive API")
        return True
    except Exception as e:
        print(f"✗ Error connecting to Drive API: {str(e)}")
        return False

def verify_folder_access(service, folder_id):
    """Test if we can access the specific folder"""
    try:
        print(f"\nStep 3: Verifying access to folder {folder_id}...")
        folder = service.files().get(
            fileId=folder_id,
            supportsAllDrives=True,
            fields='name, driveId, mimeType, owners, shared'
        ).execute()
        
        print("✓ Folder details:")
        print(f"  - Name: {folder.get('name')}")
        print(f"  - Type: {folder.get('mimeType')}")
        print(f"  - Drive ID: {folder.get('driveId')}")
        print(f"  - Shared: {folder.get('shared')}")
        return folder
    except Exception as e:
        print(f"✗ Error accessing folder: {str(e)}")
        return None

def list_files_in_folder(service, folder_id):
    """List files in a shared folder"""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=10,
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            # Removing corpora and driveId parameters since this is a shared folder, not a shared drive
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
        print(f"\n❌ Error listing files: {str(e)}")
        return []

def list_files_recursively(service, folder_id, indent=""):
    """
    Recursively list all files and folders within a given folder.
    
    Args:
        service: The Google Drive API service instance
        folder_id: ID of the folder to explore
        indent: String used for visual hierarchy in output
    """
    try:
        # Query for all files and folders in the current folder
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=1000,  # Increased to handle more files
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        items = results.get('files', [])
        
        for item in items:
            # Print the current item
            print(f"{indent}{'📁' if item['mimeType'] == 'application/vnd.google-apps.folder' else '📄'} {item['name']}")
            
            # If it's a folder, recursively list its contents
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                list_files_recursively(service, item['id'], indent + "  ")
                
    except Exception as e:
        print(f"{indent}❌ Error listing files: {str(e)}")

def main():
    folder_id = '10oVxCZ40AVmH75EywvSp-8yKrC6MegD4'
    
    print("=== Google Drive Access Diagnostic ===\n")
    
    # [Previous credential and service verification code stays the same]
    credentials = service_account.Credentials.from_service_account_file(
        'service_account.json', 
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=credentials)
    
    print("\nAttempting to list files...")
    list_files_in_folder(service, folder_id)

    print("Exploring folder structure...\n")
    list_files_recursively(service, folder_id)

if __name__ == '__main__':
    main()