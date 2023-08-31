from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from httplib2 import Http
from oauth2client import file
import io

store = file.Storage('storage.json')
creds= store.get()
service = build('drive','v3', http=creds.authorize(Http()))

results = service.files().list(pageSize=100,fields="nextPageToken,files(id,name)").execute()
items = results.get('files',[])
print(items)



request = service.files().get_media(fileId=folder_id)
fh=io.FileIO(file_name, 'wb')

downloader = MediaIoBaseDownload(fh, request)