#!/usr/bin/env python

"""
Shows how to use floodfill
"""

import cv2
import numpy as np

img = cv2.imread('../imgs/fisherman.jpg')
h, w = img.shape[:2]

# flood fill
diff = (6, 6, 6)
mask = np.zeros((h+2, w+2), 'uint8')
seed_point = (10, 10)
cv2.floodFill(img, mask, seed_point, (255, 255, 255), diff, diff)

cv2.imshow('flood fill', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
