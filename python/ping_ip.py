#!/usr/bin/env python3

"""
Ping IP in Python

From: https://www.delftstack.com/ Python Ping
"""

import platform
import subprocess

def myping(host):
    parameter = '-n' if platform.system().lower()=='windows' else '-c'

    command = ['ping', parameter, '1', host]
    response = subprocess.call(command)

    if response == 0:
        return True
    else:
        return False

print(myping("www.bing.com"))
