#!/usr/bin/env python

"""
Shows how to use Threshold
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = np.zeros((256, 512), dtype=np.uint8)
for i in range(img.shape[0]):
    img[i, :] = i


# threshold
threshold = 100
max_val = 255

# src > threshold ? max_val : 0
_, img_bin = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY)
cv2.putText(img_bin, 'BINARY', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))

# src > threshold ? 0 : max_val
_, img_bin_inv = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY_INV)
cv2.putText(img_bin_inv, 'BINARY_INV', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))

# src > threshold ? max_vale : src
_, img_trunc = cv2.threshold(img, threshold, max_val, cv2.THRESH_TRUNC)
cv2.putText(img_trunc, 'TRUNC', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))

# src > threshold ? src : 0
_, img_to_zero = cv2.threshold(img, threshold, max_val, cv2.THRESH_TOZERO)
cv2.putText(img_to_zero, 'TOZERO', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))

# src > threshold ? 0 : src
_, img_to_zero_inv = cv2.threshold(img, threshold, max_val, cv2.THRESH_TOZERO_INV)
cv2.putText(img_to_zero_inv, 'TOZERO_INV', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))


# put together
img_r1 = np.hstack((img, img_bin, img_bin_inv))
img_r2 = np.hstack((img_trunc, img_to_zero, img_to_zero_inv))
img_all = np.vstack((img_r1, img_r2))

cv2.imshow("Threshold", img_all)
cv2.waitKey(0)
cv2.destroyAllWindows()


