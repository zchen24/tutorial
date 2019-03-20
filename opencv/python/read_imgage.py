#!/usr/bin/env python

"""
How to read & display an image
"""

import cv2


img = cv2.imread('./data/messi5.jpg')
cv2.imshow('default', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
