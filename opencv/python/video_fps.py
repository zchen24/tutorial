#!/usr/bin/env python

"""
How to get a video's FPS data

Author: Zihan Chen
Date: 2023-02-02
"""

import cv2

cap = cv2.VideoCapture('./data/flame.avi')
print("Video FPS: {}".format(cap.get(cv2.CAP_PROP_FPS)))
cap.release()
