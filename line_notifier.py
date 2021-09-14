# -*- coding: UTF-8 -*-

import argparse
import datetime
import os
import time

from linebot import LineBotApi
from linebot.models import TextSendMessage


def getModelMessage(doer, jobName, operation):
    hostname = os.uname()[1]
    nowTS = time.time()
    now = datetime.datetime.fromtimestamp(nowTS).strftime('%Y-%m-%d %H:%M %p')
    form = '*******Job Notification*******\nDoer: {}\nJob Name: {}\nJob Item: {}\nhost: {}\nTime: {}'

    return form.format(doer, jobName, operation, hostname, now)


def sendMessage(doer, jobName, operation):
    line_bot_api = LineBotApi(ACCESS_TOKEN)
    line_bot_api.push_message(USER_OR_GROUP_ID,
                              TextSendMessage(text=getModelMessage(doer, jobName, operation)))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--doer')
#     parser.add_argument('--jobName')
#     parser.add_argument('--operation')

#     args = parser.parse_args()

#     line_bot_api = LineBotApi(ACCESS_TOKEN)
#     line_bot_api.push_message(USER_ID,
#                               TextSendMessage(text=getModelMessage(args.doer, args.jobName, args.operation)))
