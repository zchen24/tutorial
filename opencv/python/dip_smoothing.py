#!/usr/bin/env python

"""
Shows how to use blue (average)
"""

import cv2


img = cv2.imread('../imgs/lena.jpg')


# TODO
#  1) dilation
#  2) erosion
#  3) others


kernel_size = (5, 5)
img_blur = cv2.blur(img, kernel_size)

cv2.imshow('default', img)
cv2.imshow('blur', img_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()



