#!/usr/bin/env python

from __future__ import print_function
from telnetlib import Telnet

"""
Telnet client using telnetlib
Zihan Chen 2018-02-06
"""

# connect to local server port 9876
tn = Telnet('127.0.0.1', 9876)
tn.write('Alibaba\n')
print(tn.read_some())

