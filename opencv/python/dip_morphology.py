#!/usr/bin/env python

"""
Shows how to use basic morphology
1) erosion
2) dilation
3) open
4) close
"""

import cv2
import numpy as np


img = cv2.imread('./data/morphology_j.png')
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(img, kernel)

# dilation
dilation = cv2.dilate(img, kernel)

# opening: erosion followed by dilation
img_opening = cv2.imread('./data/opening.png', cv2.IMREAD_GRAYSCALE)
opening = cv2.morphologyEx(img_opening, cv2.MORPH_OPEN, kernel)

# closing: dilation followed by erosion
img_closing = cv2.imread('./data/closing.png', cv2.IMREAD_GRAYSCALE)
closing = cv2.morphologyEx(img_closing, cv2.MORPH_CLOSE, kernel)


cv2.imshow('default', img)
cv2.imshow('erosion', erosion)
cv2.imshow('dilation', dilation)
cv2.imshow('opening', np.hstack((img_opening, opening)))
cv2.imshow('closing', np.hstack((img_closing, closing)))

cv2.waitKey(0)
cv2.destroyAllWindows()
