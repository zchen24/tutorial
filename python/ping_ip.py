#!/usr/bin/env python3

"""
Ping IP in Python

From: https://www.delftstack.com/ Python Ping
"""

import platform
import subprocess


def my_ping(host):
    """
    my ping function
    """
    parameter = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', parameter, '1', host]
    response = subprocess.call(command)

    return response == 0


print(my_ping("www.bing.com"))
