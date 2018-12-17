#!/usr/bin/env python

"""
Shows how to do white balance using CV xphoto module

Performance:
simpleWB: 400x300 = 2ms

Date: 2018-12-16
"""

import cv2
import cProfile


img = cv2.imread('./data/dog.png')

wb = cv2.xphoto.createSimpleWB()

profile = cProfile.Profile()
profile.enable()

for i in range(100):
    img_wb = wb.balanceWhite(img)

profile.disable()
profile.print_stats(sort='time')

wb = cv2.xphoto.createGrayworldWB()
wb.setSaturationThreshold(0.99)
img_wb_gray = wb.balanceWhite(img)


cv2.imshow('img', img)
cv2.imshow('img_wb', img_wb)
cv2.imshow('img_wb_gray', img_wb_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
