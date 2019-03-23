#!/usr/bin/env python

"""
Shows how to use smoothing filter
1) Blur (Moving average)
2) Gaussian
3) Median
4) Bilateral
"""

import cv2


img = cv2.imread('../imgs/lena.jpg')

kernel_size = (5,5)
sigma_x = 1
img_gaussian_blur = cv2.GaussianBlur(img, kernel_size, sigma_x)
img_blur = cv2.blur(img, kernel_size)  # avg filter
img_median = cv2.medianBlur(img, 5)
# cv2.bilateralFilter(img)

cv2.putText(img_gaussian_blur, 'Gaussian', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
cv2.putText(img_blur, 'Blur (avg)', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
cv2.putText(img_median, 'Median', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))

cv2.imshow('default', img)
cv2.imshow('Gaussian blur', img_gaussian_blur)
cv2.imshow('Blur', img_blur)
cv2.imshow('Median', img_median)


cv2.waitKey(0)
cv2.destroyAllWindows()



