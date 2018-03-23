#!/usr/bin/env python

import cv2

# load img & grey
fname = './data/left01.jpg'
img = cv2.imread(fname)
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
size = (7,6)
cv2.imshow('img', img)

# find chessboard colors
ret, corners = cv2.findChessboardCorners(imgGrey, size, None)

# get subpix (optional)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cv2.cornerSubPix(imgGrey,corners,(11,11),(-1,-1),criteria)

# draw chessboard
#   - draw corners in differnt pattern based on patternWasFound
#   - True: draw color and lines
#   - False: darw red circles on each color
cv2.drawChessboardCorners(img, size, corners, True)

# show imgs
cv2.imshow('chess', img)
cv2.waitKey()
cv2.destroyAllWindows()




