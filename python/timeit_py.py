#!/usr/bin/env python

"""
timeit module example.

Example output:
cpython:    0.22327 sec
pypy:       0.02673 sec

Reference:
https://docs.python.org/3.7/library/timeit.html
"""

import timeit
import math


def f(x):
    return math.exp(x)


def integrate_f(a, b, steps=1000):
    s = 0
    dx = (b - a) / steps
    for i in range(steps):
        s += f(a + i * dx)
    return s * dx


if __name__ == '__main__':
    print(timeit.timeit('integrate_f(0, 1, 100)',
                        number=10000,
                        setup='from __main__ import integrate_f'))

