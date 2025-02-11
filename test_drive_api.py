from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os

# Use more restricted scope
SCOPES = [
    'https://www.googleapis.com/auth/drive.file'
]

def authenticate_drive():
    """Handle the authentication process with Google Drive."""
    creds = None
    if os.path.exists('token.pickle'):
        print("Found existing token...")
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing token...")
            creds.refresh(Request())
        else:
            print("Getting new token...")
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def list_files_in_folder(service, folder_id):
    """
    List only files within a specific folder.
    
    Args:
        service: Google Drive API service instance
        folder_id: ID of the folder to search in (from folder's Drive URL)
    """
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        pageSize=10,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def list_files_in_folder_recursive(service, folder_id, prefix=''):
    """
    Recursively list files and folders within a Google Drive folder.
    
    Args:
        service: Google Drive API service instance
        folder_id: ID of the folder to search in
        prefix: String to show folder hierarchy in output (used for indentation)
    """
    # Query to get both files and folders in the current folder
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        pageSize=1000,  # Increased to handle more items
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    items = results.get('files', [])
    
    all_items = []
    
    for item in items:
        # Store basic information about the item
        item_info = {
            'name': f"{prefix}{item['name']}",
            'id': item['id'],
            'mimeType': item['mimeType']
        }
        
        # If it's a folder, recursively get its contents
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            print(f"{prefix}📁 {item['name']}/")
            # Recursively get contents of this folder
            subfolder_items = list_files_in_folder_recursive(
                service, 
                item['id'], 
                prefix=prefix + '  '
            )
            all_items.extend(subfolder_items)
        else:
            print(f"{prefix}📄 {item['name']}")
        
        all_items.append(item_info)
    
    return all_items

def list_files_in_shared_drive_recursive(service, folder_id, prefix=''):
    """
    Recursively list files and folders within a shared Google Drive folder.
    
    This function is specifically designed to work with shared drives and folders.
    It includes the necessary parameters to access shared drive content and
    handles the different permission structure of shared resources.
    
    Args:
        service: Google Drive API service instance
        folder_id: ID of the folder to search in
        prefix: String to show folder hierarchy in output (used for indentation)
    """
    # Query to get both files and folders in the current folder
    # Including supportsAllDrives=True to access shared drives
    query = f"'{folder_id}' in parents and trashed=false"
    try:
        results = service.files().list(
            q=query,
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            driveId=folder_id,
            corpora='drive'
        ).execute()
    except Exception as e:
        print(f"Error accessing folder {folder_id}: {str(e)}")
        return []

    items = results.get('files', [])
    all_items = []
    
    for item in items:
        item_info = {
            'name': f"{prefix}{item['name']}",
            'id': item['id'],
            'mimeType': item['mimeType']
        }
        
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            print(f"{prefix}📁 {item['name']}/")
            subfolder_items = list_files_in_shared_drive_recursive(
                service, 
                item['id'], 
                prefix=prefix + '  '
            )
            all_items.extend(subfolder_items)
        else:
            print(f"{prefix}📄 {item['name']}")
        
        all_items.append(item_info)
    
    return all_items

def main():
    """Main function to test Google Drive API access."""
    try:
        print("Starting authentication process...")
        creds = authenticate_drive()
        
        print("Creating Drive service...")
        service = build('drive', 'v3', credentials=creds)
        
        # Replace with your actual folder ID
        folder_id = '10oVxCZ40AVmH75EywvSp-8yKrC6MegD4'

        print("\nListing all files and folders recursively in shared drive:")
        items = list_files_in_shared_drive_recursive(service, folder_id)

        print("\nSummary of found items:")
        files_count = sum(1 for item in items if item['mimeType'] != 'application/vnd.google-apps.folder')
        folders_count = sum(1 for item in items if item['mimeType'] == 'application/vnd.google-apps.folder')
        print(f"Found {files_count} files in {folders_count} folders")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details to help with debugging:")
        import traceback
        traceback.print_exc()
        
    #     print("\nListing all folders and files from the specified folder recursively:")
    #     items = list_files_in_folder_recursive(service, folder_id)

    #     print("\nSummary of found items:")
    #     files_count = sum(1 for item in items if item['mimeType'] != 'application/vnd.google-apps.folder')
    #     folders_count = sum(1 for item in items if item['mimeType'] == 'application/vnd.google-apps.folder')
    #     print(f"Found {files_count} files in {folders_count} folders")
                
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")

    #     if not items:
    #         print('No files found in the specified folder.')
    #     else:
    #         print('Files:')
    #         for item in items:
    #             print(f"- {item['name']} ({item['mimeType']})")
                
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    print("Testing Google Drive API Connection...")
    main()