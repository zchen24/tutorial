#!/usr/bin/env python

import cv2

img_raw = cv2.imread('./data/debayer_raw.png', cv2.IMREAD_GRAYSCALE)

# see cvtColor doc for how to get Bayer code
img_color = cv2.cvtColor(img_raw, cv2.COLOR_BayerRG2BGR)

cv2.imshow('raw', img_raw)
cv2.imshow('img', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()