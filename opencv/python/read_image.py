#!/usr/bin/env python

"""
How to read & display an image
"""

import cv2


img = cv2.imread('./data/messi5.jpg')
img_gray = cv2.imread('./data/messi5.jpg', cv2.IMREAD_GRAYSCALE)


cv2.imshow('color', img)
cv2.imshow('gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
