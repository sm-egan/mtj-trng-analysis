from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os

# Use more restricted scope
SCOPES = ['https://www.googleapis.com/auth/drive.file']

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

def main():
    """Main function to test Google Drive API access."""
    try:
        print("Starting authentication process...")
        creds = authenticate_drive()
        
        print("Creating Drive service...")
        service = build('drive', 'v3', credentials=creds)
        
        # Replace with your actual folder ID
        folder_id = '10oVxCZ40AVmH75EywvSp-8yKrC6MegD4'
        
        print("\nListing files from the specified folder:")
        items = list_files_in_folder(service, folder_id)

        if not items:
            print('No files found in the specified folder.')
        else:
            print('Files:')
            for item in items:
                print(f"- {item['name']} ({item['mimeType']})")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    print("Testing Google Drive API Connection...")
    main()