#!/usr/bin/env python3

"""
Example struct library:
Pack/unpack bytes, performs conversions between Python
values and C structs as Python bytes.

For a full list of format please see:
https://docs.python.org/3.8/library/struct.html

c: char
b: signed char
h: short
i: int
l: long
"""

import struct


fmt = 'hhi'
out_bytes = struct.pack(fmt, 1, 2, 3)
print(out_bytes)

n1, n2, n3 = struct.unpack('hhi', out_bytes)
print('n1 = {}, n2 = {}, n3 = {}'.format(n1, n2, n3))
