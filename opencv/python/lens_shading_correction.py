#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lsc.png')


cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()