#!/usr/bin/env python

"""
Put text on image
"""

import cv2

img = cv2.imread('../imgs/lena.jpg')
location = (50, 50)  # x, y
font = cv2.FONT_HERSHEY_PLAIN
fontScale = 2.0
fontColor = (0, 255, 255)  # B,G,R
thickness = 3
cv2.putText(img, 'put text', location, cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, thickness)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
