#!/usr/bin/env python

from numba.pycc import CC

cc = CC('numba_module')
# cc.verbose = True   # uncomment for verbose compilation


@cc.export('multi4', 'i4(i4, i4)')
@cc.export('mult', 'f8(f8, f8)')
def mult(a, b):
    return a * b


@cc.export('square', 'f8(f8)')
def square(a):
    return a ** 2

if __name__ == '__main__':
    cc.compile()