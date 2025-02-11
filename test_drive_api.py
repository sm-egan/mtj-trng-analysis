# Import necessary Google API libraries
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os

# These are the permissions we're requesting from Google Drive
# .readonly means we'll only read files, not modify them
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """
    Handle the authentication process with Google Drive.
    This function will:
    1. Check for existing tokens
    2. Use existing token if valid
    3. Refresh expired token if possible
    4. Create new token if needed
    """
    creds = None
    
    # First, try to load existing tokens from token.pickle
    if os.path.exists('token.pickle'):
        print("Found existing token...")
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials available, let's get them
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing token...")
            creds.refresh(Request())
        else:
            print("Getting new token...")
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for future runs
        print("Saving token for future use...")
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def main():
    """
    Main function to test Google Drive API access.
    Will list the first 10 files in your Google Drive.
    """
    try:
        # First, authenticate
        print("Starting authentication process...")
        creds = authenticate_drive()
        
        # Create the Drive API service
        print("Creating Drive service...")
        service = build('drive', 'v3', credentials=creds)
        
        # List files
        print("\nListing some files from your Drive:")
        results = service.files().list(
            pageSize=10,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(f"- {item['name']} ({item['mimeType']})")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    # Make sure your virtual environment is activated
    print("Testing Google Drive API Connection...")
    main()