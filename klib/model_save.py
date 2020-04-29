# -*- coding: utf-8 -*-
# Put credentials.json and setting.yaml to working directory.

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def google_drive_upload(fn, loc='2_kaggle'):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()

    drive = GoogleDrive(gauth)

    # Upload Local File
    # f = drive.CreateFile()
    # f.SetContentFile('google-drive-api.pdf')
    # f.Upload()

    # Get Google Drive Information
    # https://note.nkmk.me/python-pydrive-list-file/

    # file_list = drive.ListFile().GetList()
    # for f in file_list:
    #     print(f['title'], '   \t', f['id'])

    # for f in drive.ListFile(
    #         {'q': 'mimeType = "application/vnd.google-apps.folder"'}).GetList():
    #     print(f['title'], '   \t', f['id'])
    #
    # 2_kaggle    	 0B8of5HHbH-kAUW8tSVF6YTROdW8

    for f in drive.ListFile({'q': 'title = "{}"'.format(loc)}).GetList():
        print(f['title'], '  \t', f['id'])

    # Upload a Local File to the Google Drive Folder
    # https://note.nkmk.me/python-pydrive-folder/
    folder_id = drive.ListFile(
        {'q': 'title = "{}"'.format(loc)}
    ).GetList()[0]['id']

    f = drive.CreateFile({"parents": [{"id": folder_id}]})
    f.SetContentFile(fn)
    f['title'] = os.path.basename(fn)
    f.Upload()
