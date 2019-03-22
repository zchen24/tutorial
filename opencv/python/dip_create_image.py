#!/usr/bin/env python

"""
Shows how to create an image using numpy
"""

import cv2
import numpy as np


width = 800
height = 600
n_channel = 3
data_type = np.uint8     # can also use float types
img = np.zeros((height, width, n_channel), dtype=data_type)
cv2.circle(img, (300, 300), 100, (255, 0, 0), cv2.FILLED)


win_name = 'Window'
cv2.namedWindow(win_name)

cv2.imshow(win_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()