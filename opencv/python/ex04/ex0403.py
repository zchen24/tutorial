#!/usr/bin/env python

"""
Draw highlight region
"""


import cv2
import numpy as np


class ImageHighlighter:

    def __init__(self):
        self.img = cv2.imread('../../imgs/lena.jpg')
        self.win_name = "highlight"
        cv2.namedWindow(self.win_name)
        # windowName   e.g. "myWindow"
        # onMouse      callback function
        # param        custom data, can be used to pass custom classes
        cv2.setMouseCallback(self.win_name, self.callback_mouse, param=None)
        self.point_1 = None
        self.point_2 = None
        cv2.imshow(self.win_name, self.img)
        cv2.waitKey(0)

    def __del__(self):
        cv2.destroyAllWindows()

    # event: mouse event type (e.g. CV_EVENT_MOUSEMOVE = 0)
    #     x: x position
    #     y: y position
    # flags: key flags (e.g. CV_EVENT_FLAG_CTRLKEY = 8)
    # param: user passed parameter
    def callback_mouse(self, event, x, y, flags, param):
        color = (255, 0, 0)
        thickness = cv2.FILLED

        if event == cv2.EVENT_LBUTTONDOWN:
            self.point_1 = (x, y)
            cv2.imshow(self.win_name, self.img)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.point_1 is not None:
                img_draw = np.copy(self.img)
                cv2.rectangle(img_draw, self.point_1, (x, y), color, thickness)
                cv2.imshow(self.win_name, img_draw)
        elif event == cv2.EVENT_LBUTTONUP:
            img_draw = np.copy(self.img)
            cv2.rectangle(img_draw, self.point_1, (x, y), color, thickness)
            self.point_1 = None
            cv2.imshow(self.win_name, img_draw)


highlighter = ImageHighlighter()
