#!/usr/bin/env python

"""
How to read/save MATLAB mat file
"""

import numpy as np
import scipy.io as sio


x = np.zeros(10).astype(np.uint8)
y = np.ones(5).astype(np.float)

sio.savemat('tmp.mat', {'x': x,
                        'y': y})

mat = sio.loadmat('tmp.mat')
x_read = mat['x']
y_read = mat['y']
print('x_read = {}\ny_read = {}\n'.format(x_read, y_read))
