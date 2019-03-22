#!/usr/bin/env python

"""
Read an image, display mouse coordinates
"""

import cv2

# event: mouse event type (e.g. CV_EVENT_MOUSEMOVE = 0)
#     x: x position
#     y: y position
# flags: key flags (e.g. CV_EVENT_FLAG_CTRLKEY = 8)
# param: user passed parameter
def callback_mouse(event, x, y, flags, param):
    print('event = {}  mouse x = {}, y = {}, flags = {}'.format(event, x, y, flags))


img = cv2.imread('../../imgs/lena.jpg')
cv2.namedWindow("window")

# windowName   e.g. "myWindow"
# onMouse      callback function
# param        custom data, can be used to pass custom classes
cv2.setMouseCallback("window", callback_mouse, param=None)
cv2.imshow("window", img)
while cv2.waitKey(10) != ord('q'):
    pass
cv2.destroyAllWindows()
