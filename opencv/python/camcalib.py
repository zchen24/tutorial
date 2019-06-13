#!/usr/bin/env python

# This code is based on Python OpenCV Camera Calibration Tutorial
# Use ROS Camera Calibration in Real Application
# 
# This example shows
#  - how to detect chessboard 
#  - how to call camera calibration
#  - how to use camera params to undistort image
#  - how to reproject 3d points to image plane (draw circle)

import numpy as np
import cv2
import glob
import pdb

# Step 1: Data prep 
img_list = glob.glob('./data/left*.jpg')
img_list_detected = []

# set opts
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
size = (7,6)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in img_list:
    img = cv2.imread(fname)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(grey, size, None)
    cv2.drawChessboardCorners(img, (7,6), corners,ret)

    # if found, show imgs
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(grey,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        objpoints.append(objp)
        img_list_detected.append(fname)
        print fname

    cv2.imshow('img',img)
    cv2.waitKey(500)

cv2.destroyAllWindows()


# Step 2: Calibration
# shape[::-1]: (480,640) => (640,480)
ret, cmx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, grey.shape[::-1])

# save calibration result
np.savez('calib.npz', cmx=cmx, dist=dist, rvecs=rvecs, tvecs=tvecs)


# Step 3: Validation Undistort Image
img = cv2.imread('./data/left02.jpg')
img_size = grey.shape[::-1]
newcmx, roi=cv2.getOptimalNewCameraMatrix(cmx, dist, img_size, 0, img_size)

dst = cv2.undistort(img, cmx, dist, None, newcmx)
cv2.imshow('original', img)
cv2.imshow('undistort', dst)
cv2.waitKey(500)
cv2.destroyAllWindows()
# pdb.set_trace()

# Step 4: Reproject Points
img = cv2.imread(img_list_detected[0])
imgpts2,_ = cv2.projectPoints(objp, rvecs[0], tvecs[0], cmx, dist)
for pt in imgpts2:
    center = (pt[0,0], pt[0,1])
    cv2.circle(img, center, 5, (0,0,255), 2)
cv2.imshow('reproject', img)
cv2.waitKey()
cv2.destroyAllWindows()





