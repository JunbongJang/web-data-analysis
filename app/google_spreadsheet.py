# Author: Junbong Jang
# Date: 12/4/2018
# app/google_spreadsheet.py

from __future__ import print_function
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from pprint import pprint

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'

class Google_Spreadsheet(object):
    def __init__(self):
        self.spreadsheet_id = ''

    def user_authentication(self):
        """Shows basic usage of the Sheets API.
        Prints values from a sample spreadsheet.
        """
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        store = file.Storage('token.json')
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
            creds = tools.run_flow(flow, store)
        service = build('sheets', 'v4', http=creds.authorize(Http()))

        return service


    def create_spreadsheet(self, service, title):
        spreadsheet_body = {
            'properties': {
                'title': title
            }
        }

        request = service.spreadsheets().create(body=spreadsheet_body)
        response = request.execute()

        pprint(response)
        self.spreadsheet_id = response['spreadsheetId']
        return response['spreadsheetUrl']


    def update_spreadsheet(self, service, list_of_values):
        body = {
            'values': list_of_values
        }
        result = service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheet_id, range='Sheet1',
            valueInputOption='USER_ENTERED', body=body).execute()
        print('{0} cells appended.'.format(result
                                           .get('updates')
                                           .get('updatedCells')))
