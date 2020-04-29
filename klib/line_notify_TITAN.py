# -*- coding: utf-8 -*-

import requests


def notifier(message='', image_name=None):
    """
    :param message: text
    :param image_name: image_file name (path)
    :return: nothing. just send message and image to line account.
    """
    url = "https://notify-api.line.me/api/notify"
    token = 'rVmq980uf3xZEGOSVFHRobVRP84nNbvKMjVTKGdOzP9'
    headers = {"Authorization": "Bearer " + token}

    payload = {"message": message}
    if image_name:
        try:
            files = {"imageFile": open(image_name, "rb")}
            requests.post(url, headers=headers, params=payload, files=files)
        except:
            requests.post(url, headers=headers, params=payload)
    else:
        requests.post(url, headers=headers, params=payload)

    # message = 'test'
    # files = {"imageFile": open("./fig/uni1.jpg", "rb")}


if __name__ == '__main__':
    notifier('test from uni on TITAN')
