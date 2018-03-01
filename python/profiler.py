#!/usr/bin/env python

import cProfile
import re

"""
Show in-code cProfile

In terminal:
python -m cProfile -o output.txt script.py
"""

if __name__ == '__main__':
    profile = cProfile.Profile()
    profile.enable()

    for i in range(100000):
        re.compile("foo|bar")

    profile.disable()
    profile.print_stats(sort='time')
