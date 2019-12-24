#!/usr/bin/env python3

# Demos how to do a stereo camera calibration using CV
# Images are from
# https://github.com/sourishg/stereo-calibration
#
# Meta inforamtion:
#   - Chessboard: 9 x 6
#   - Chessboard: 24.23 mm
#   - Image size: 640 x 360

# Alternatively use MATLAB camera calibration app


import glob
import os
import numpy as np
import cv2

# Step 1: Data prep
img_list_left = sorted(glob.glob('./data/stereo_calibration/left*.jpg'))
img_list_right = sorted(glob.glob('./data/stereo_calibration/right*.jpg'))

pattern_size = (6,9)
square = 0.02423

img_list_detected = []

for left_file, right_file in zip(img_list_left, img_list_right):
    left = cv2.imread(left_file)
    right = cv2.imread(right_file)


    cv2.findChessboardCorners(left, (6, 9))


    cv2.imshow('image', np.hstack((left, right)))
    cv2.waitKey(100)

cv2.destroyAllWindows()

# Step 2: Mono calibration

# Step 3: Stereo calibration
#   - chessboard detection
#   - calibrate


# Step 4: Re-project & Un-distort


# Step 5: Triangulation
