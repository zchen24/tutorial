#!/usr/bin/env python

"""
Shows how to use blue (average)
    1 1 1
K = 1 1 1  *  1/9
    1 1 1
"""

import cv2


img = cv2.imread('../imgs/lena.jpg')

kernel_size = (5, 5)
img_blur = cv2.blur(img, kernel_size)

cv2.imshow('default', img)
cv2.imshow('blur', img_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()



