#!/usr/bin/env python

"""
Shows how to use GaussianBlur
"""

import cv2


img = cv2.imread('../imgs/lena.jpg')

kernel_size = (5,5)
sigma_x = 1
img_blur = cv2.GaussianBlur(img, kernel_size, sigma_x)

cv2.imshow('default', img)
cv2.imshow('blur', img_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()



