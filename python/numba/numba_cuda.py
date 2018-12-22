#!/usr/bin/env python

"""
Numba cuda example
1) use numba -s to check cuda compability
2) Install if necessary: conda install cudatoolkit
"""


from numba import cuda


dev = cuda.get_current_device()
print('Cuda device name = {}'.format(dev.name))