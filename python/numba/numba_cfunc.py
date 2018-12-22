#!/usr/bin/env python

"""
From numba tutorial
https://numba.pydata.org/numba-doc/dev/user/cfunc.html
"""


from numba import cfunc

@cfunc("float64(float64, float64)")
def add(x, y):
    return x + y


if __name__ == '__main__':
    print('add(4,5) = {}'.format(add.ctypes(4.0, 5.0)))
    # throws exception: CFunc is not callable
    print('add(4,5) = {}'.format(add(4.0, 5.0)))
