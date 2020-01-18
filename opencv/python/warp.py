#!/usr/bin/env python

"""
Shows how to do affine transform (warping)

Date: 2019-07-03
"""

from pylab import *
from scipy import ndimage
import cv2


im = cv2.imread('./data/empire.jpg', cv2.IMREAD_GRAYSCALE)
H = np.array([[1.4,0.05,-100],
              [0.05,1.5,-100],
              [0,0,1]])

# A = H[:2,:2]
# t = H[:2,2]
im2 = ndimage.affine_transform(im, H[:2,:2], H[:2,2])

plt.figure(); plt.gray()
plt.imshow(im)
plt.figure(); plt.gray()
plt.imshow(im2)
plt.show()
