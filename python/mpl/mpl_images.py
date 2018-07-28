#!/usr/bin/env python

"""
Demonstrate matplotlib image operations

Copyright 2018 Zihan Chen
"""


import matplotlib.pyplot as plt
from scipy.misc import face


plt.close('all')
fig, (ax1, ax2) = plt.subplots(2,1)

# read image
img = plt.imread('mpl_basics.jpg')
ax1.imshow(img)

# gray scale images
img_gray = face(gray=True)
ax2.imshow(img_gray, cmap=plt.cm.bone)

# show images
plt.show()
