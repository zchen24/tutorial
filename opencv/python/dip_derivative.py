#!/usr/bin/env python

"""
Shows how to use blue (average)
    1 1 1
K = 1 1 1  *  1/9
    1 1 1
"""

import cv2


img = cv2.imread('../imgs/lena.jpg')

img_x_sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0)
img_y_sobel = cv2.Sobel(img, cv2.CV_8U, 0, 1)

cv2.imshow('default', img)
cv2.imshow('sobel x', img_x_sobel)
cv2.imshow('sobel y', img_y_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()



