#!/usr/bin/env python

"""
Shows how to use cv2.integral
"""

import cv2


img = cv2.imread('../imgs/fisherman.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_int = cv2.integral(img_gray)
img_int = 255.0 * img_int / img_int.max()


cv2.imshow('default', img)
cv2.imshow('integral', img_int.astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
