#!/usr/bin/env python

"""
SciPy NumPy Beginner 2016
Video: https://www.youtube.com/watch?v=gtejJ3RCddE
"""

from pylab import *


num_pts = 20
x = linspace(0, 2*pi, num_pts)
y = sin(x)
yerr = rand(num_pts)
errorbar(x, y, yerr=yerr)

# reshape
m23 = arange(6).reshape(2,3)


# ===================
# Array Operations
# ===================
a = arange(6)
amean = a.mean()
amin = a.min()
amax = a.max()
a.argmin()
a.argmax()

unravel_index(a.argmin(), a.sha)



# ======================
def np_array():
    a = array([1, 2, 3, 4, 5, 6])
    type(a)
    print('a.dtype = {}'.format(a.dtype))
    print('a.itemsize = {}'.format(a.itemsize))
    print('a.size = {}'.format(a.size))
    print('a.shape = {}'.format(a.shape))
