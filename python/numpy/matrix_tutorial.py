#!/usr/bin/env python

"""
Date: Nov 8, 2018
Video: https://www.youtube.com/watch?v=gtejJ3RCddE
"""

from pylab import *


mat = array([[1,2,3],
             [4,5,6],
             [7,8,9]], dtype=np.float)

print("matrix ndim  = {}".format(mat.ndim))
print("matrix shape = {}".format(mat.shape))


# convert to 1d
mat_flat = mat.flatten()
print('mat_flat = {}'.format(mat_flat))

# 1d to 2d array
print('mat reshape = {}'.format(mat_flat.reshape(3,3)))

# svd
u, s, v = np.linalg.svd(mat)


# bit shifting e.g. left shift 2-bits
mat_left_shift = np.left_shift(mat.astype(np.uint32), 2)
print('mat_left2 = {}'.format(mat_left_shift))
