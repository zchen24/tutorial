#!/usr/bin/env python3


"""
OS module's feature
Date: 2022-09-18
"""

import os
import glob


# get a list of file
fs = glob.glob("*.py")
print('Found {} py files'.format(len(fs)))

# get file size
f = fs[0]
print('File {} size: {} bytes'.format(f, os.path.getsize(f)))

# get file extension
ext = os.path.splitext(f)
print('File ext: {}'.format(ext[-1]))

