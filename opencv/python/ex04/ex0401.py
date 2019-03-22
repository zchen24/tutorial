#!/usr/bin/env python

"""
Learning OpenCV 1e: Ch03 Ex01
Create a program 1) reads from a video, 2) turns to grayscale, 3) canny edge
"""

import cv2
import numpy as np


cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edge = cv2.Canny(img_gray, 100, 200)

    # text labels
    cv2.putText(img, 'original', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
    cv2.putText(img_gray, 'gray', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
    cv2.putText(img_edge, 'edge', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))

    cv2.imshow("image", img)
    cv2.imshow("gray", img_gray)
    cv2.imshow("edge", img_edge)

    # stack images
    img_stack = np.hstack((img_gray, img_edge))
    cv2.imshow("stack", img_stack)

    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()