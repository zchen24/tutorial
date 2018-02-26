#!/usb/bin/env python

"""
schedule IN-PROCESS periodic jobs

setup:
pip install schedule

example use case:
fetch a stock market price and send to you everyday

example output:
Job called at 1519611192.880
Job called at 1519611197.880
Job called at 1519611202.881
"""

from __future__ import print_function
import time
import schedule


def job():
    print('Job called at %.3f' % time.time())


if __name__ == '__main__':
    """call job every 10 seconds"""
    interval = 5
    schedule.every(interval).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
