from __future__ import print_function
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES=['https://www.googleapis.com/auth/drive']

creds = None

if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json',SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request()) 
        
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'client_secret_drive.json', SCOPES)
        creds = flow.run_local_server(port=44605)
        # creds = flow.run_console()
    with open('OAuthkey.json', 'w') as token:
        token.wirte(creds.to_json())
        