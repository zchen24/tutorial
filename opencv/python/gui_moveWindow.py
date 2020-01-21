#!/usr/bin/env python

"""
How to show named window at certain location
"""

import cv2

img = cv2.imread('../imgs/lena.jpg')
cv2.namedWindow('image')

cv2.imshow("image", img)

for x in range(6):
    cv2.moveWindow('image', x * 100, 100)
    cv2.waitKey(1000)

cv2.destroyAllWindows()
