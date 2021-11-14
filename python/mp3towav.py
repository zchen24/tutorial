#!/usr/bin/env python3

"""
A script that converts an MP3 file to an WAV format

Author: Zihan Chen
Date: 2021-11-14
"""

import os
import glob
import subprocess

files_mp3 = glob.glob("*.mp3")

for file in files_mp3:
    file_out = os.path.splitext(file)[0] + '.wav'
    command = ['ffmpeg', '-i', file, '-f', 'wav', file_out]
    response = subprocess.call(command)
    print('Converting: {}, {}'.format(file, response))