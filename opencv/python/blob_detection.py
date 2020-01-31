#!/usr/bin/env python3

"""
Shows cv blob detection

https://www.learnopencv.com/blob-detection-using-opencv-python-c/
"""

import cv2


img = cv2.imread('../imgs/BlobTest.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Blob', img); cv2.waitKey(0); cv2.destroyAllWindows()

params = cv2.SimpleBlobDetector_Params()
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)
print("keypoints = {}".format(keypoints))
img_out = cv2.drawKeypoints(img,
                            keypoints,
                            None,
                            (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Blob Out', img_out); cv2.waitKey(0); cv2.destroyAllWindows()


# The behavior of the detector is controlled by the parameters
#
# The filtering step is based on a couple of features e.g.
# color, area size, shape etc. Each category can be turned ON/OFF
# by setting the filterByXXXX flag and tuned by setting relevant
# parameters.
#
# The following setting is tuned for circles

# blob detection parameters
params.minThreshold = 10
params.maxThreshold = 220

# filter by color

# filter by area
params.filterByArea = True
params.minArea = 50
params.maxArea = 50000

# filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# filter by circularity
params.filterByCircularity = True
# params.minCircularity = 0.95

# filter by inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)
print("keypoints = {}".format(keypoints))
img_out = cv2.drawKeypoints(img,
                            keypoints,
                            None,
                            (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Blob Out', img_out); cv2.waitKey(0); cv2.destroyAllWindows()
