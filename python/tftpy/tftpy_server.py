#!/usr/bin/env python

import tftpy

"""
Use python to setup a tftp server
Author: Zihan Chen
Date: 2024-04-21
"""

port = 69
server = tftpy.TftpServer("/path/to/tftp_folder")
server.listen('127.0.0.1', port)
