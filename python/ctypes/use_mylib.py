#!/usr/bin/env python

from __future__ import print_function
import os
from ctypes import *
import ctypes


if os.name == 'nt':
    mylib = ctypes.WinDLL('mylib.dll')
else:
    mylib = cdll.LoadLibrary('mylib.so')

val = mylib.my_add(c_int(1), c_int(2))
print('my_add(1,2) = %d' % val)

val = c_int()
mylib.get_a_value(pointer(val))
print('get_a_value(int*) returns %d' % val.value)
