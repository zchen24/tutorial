#!/usr/bin/env python

"""
Shows how to resize image

Date: 2018-12-16
"""

import cv2

img_raw = cv2.imread('./data/debayer_raw.png', cv2.IMREAD_GRAYSCALE)

img_raw_half = cv2.resize(img_raw, None, fx=0.5, fy=0.5)

cv2.imshow('img', img_raw)
cv2.imshow('img_half', img_raw_half)
cv2.waitKey(0)
cv2.destroyAllWindows()
