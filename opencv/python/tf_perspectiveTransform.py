#!/usr/bin/env python3

"""
Shows how to use perspectiveTransform. The function transforms
a sparse set of 2D or 3D vectors.

x'          x
y'  =  M *  y
z'          z
w'          1

Example usage: 3D triangulation Q * [x_l, y_l, disparity, 1]
"""


import cv2
import numpy as np


# the reshape is important
# N x 1 x 3(or 2)
src = np.array([[1, 1, 1],
                [2, 2, 2]], dtype=np.float).reshape(-1, 1, 3)
M = np.array([[1, 0, 0, 1],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float)


dst = cv2.perspectiveTransform(src, M)
print('src = \n{}\n'.format(src))
print('M = \n{}\n'.format(M))
print('dst = \n{}\n'.format(dst))
