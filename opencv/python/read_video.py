#!/usr/bin/env python

"""
How to read & display an image
"""

import cv2

cap = cv2.VideoCapture('./data/flame.avi')

while True:
    # read video frame-by-frame
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(50) == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

