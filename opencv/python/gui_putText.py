#!/usr/bin/env python

"""
Put text on image
"""

import cv2

img = cv2.imread('../imgs/lena.jpg')
cv2.putText(img, 'put text', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
