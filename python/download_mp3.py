#!/usr/bin/env python3

"""
Downlaod mp3 file for Cantonese Textbook
Date: 2022-05-29
"""

import os
import subprocess


path_base = 'https://jpchinese.org/DownLoad/cantonese/9789620447990/data/'


for ch in range(31):
    for ls in range(5):
        fname_padded = '{:02}-{}.mp3'.format(ch, ls+1)
        path_full = os.path.join(path_base, '{}-{}.mp3'.format(ch, ls+1))       
        subprocess.call(["wget", path_full])
    
