#!/usr/bin/env python3

# Demos how to do a stereo camera calibration using CV
# Images are from
# https://github.com/sourishg/stereo-calibration
#
# Meta information:
#   - Chessboard: 9 x 6
#   - Chessboard: 24.23 mm
#   - Image size: 640 x 360

# Alternatively use MATLAB camera calibration app


from os.path import basename
import glob
import numpy as np
import cv2
from CamCalibrator import CheckerBoardInfo, CalibratorMono


# Step 1: Data prep
img_list_left = sorted(glob.glob('./data/stereo_calibration/left*.jpg'))
img_list_right = sorted(glob.glob('./data/stereo_calibration/right*.jpg'))


square = 0.02423
b = CheckerBoardInfo(6, 9, square)

checker_boards = []
img_list_detected = []
img_points_left = []
img_points_right = []
board_points = []
img_width = 0
img_height = 0
cv2.namedWindow('Left'); cv2.moveWindow('Left', 100, 50)
cv2.namedWindow('Right'); cv2.moveWindow('Right', 100, 500)

for left_file, right_file in zip(img_list_left, img_list_right):
    left = cv2.imread(left_file)
    right = cv2.imread(right_file)
    img_height, img_width = left.shape[:2]
    if left.shape[:2] != right.shape[:2]:
        print('Left and right image size mismatch')
        continue
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found0, points0 = cv2.findChessboardCorners(left, b.size, flags=flags)
    found1, points1 = cv2.findChessboardCorners(right, b.size, flags=flags)

    # save data when checker board is found in both images
    show_left = np.copy(left)
    show_right = np.copy(right)
    cv2.putText(show_left, basename(left_file), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(show_right, basename(right_file), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    if found0 and found1:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        sub_points0 = cv2.cornerSubPix(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), points0, (11, 11), (-1, -1), criteria)
        sub_points1 = cv2.cornerSubPix(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), points1, (11, 11), (-1, -1), criteria)
        img_points_left.append(sub_points0)
        img_points_right.append(sub_points1)
        board_points.append(b.make_object_points())
        checker_boards.append(b)
        cv2.drawChessboardCorners(show_left, b.size, points0, found0)
        cv2.drawChessboardCorners(show_right, b.size, points1, found1)

    cv2.imshow('Left', show_left)
    cv2.imshow('Right', show_right)
    if cv2.waitKey(100) == ord('q'):
        cv2.destroyAllWindows()
        exit(-1)

# Step 2: Mono calibration

img_size = (img_width, img_height)

# calibrate
K0 = np.eye(3, dtype=np.float64)           # intrinsic camera matrix
D0 = np.zeros((1, 5), dtype=np.float64)    # distortion matrix (plumb bob) model
flags = cv2.CALIB_ZERO_TANGENT_DIST
ret0, K0, D0, Rs0, Ts0 = cv2.calibrateCamera(board_points, img_points_left, (img_width, img_height), K0, D0, flags=flags)
ret1, K1, D1, Rs1, Ts1 = cv2.calibrateCamera(board_points, img_points_left, (img_width, img_height), K0, D0, flags=flags)

print('K0 = \n{}\n'.format(K0))
print('D0 = \n{}\n'.format(D0))
print('K1 = \n{}\n'.format(K1))
print('D1 = \n{}\n'.format(D1))



# Step 3: Stereo camera calibration
R = np.eye(3, dtype=np.float64)
T = np.zeros((3, 1), dtype=np.float64)
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
ret, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(board_points,
                                                      img_points_left,
                                                      img_points_right,
                                                      K0, D0,
                                                      K1, D1,
                                                      img_size,
                                                      R, T,
                                                      flags=flags,
                                                      criteria=criteria)
print('K0 = \n{}\n'.format(K0))
print('D0 = \n{}\n'.format(D0))
print('K1 = \n{}\n'.format(K1))
print('D1 = \n{}\n'.format(D1))
print('R = \n{}\n'.format(R))
print('T = \n{}\n'.format(T))


# Step 4: Re-project & Un-distort
# mono
flags = cv2.CALIB_ZERO_DISPARITY
R0, R1, P0, P1, Q, ROI0, ROI1 = cv2.stereoRectify(K0, D0,
                                                  K1, D1,
                                                  img_size,
                                                  R,
                                                  T,
                                                  flags=flags,
                                                  alpha=-1,
                                                  newImageSize=img_size)

# create x,y map for each camera
m1type = cv2.CV_32FC1
MX0, MY0 = cv2.initUndistortRectifyMap(K0, D0, R0, P0, img_size, m1type)
MX1, MY1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, m1type)

for left_file, right_file in zip(img_list_left, img_list_right):
    left = cv2.imread(left_file)
    right = cv2.imread(right_file)
    left_rect = cv2.remap(left, MX0, MY0, cv2.INTER_LANCZOS4)
    right_rect = cv2.remap(left, MX1, MY1, cv2.INTER_LANCZOS4)
    cv2.imshow('Left', left_rect)
    cv2.imshow('Right', right_rect)
    if cv2.waitKey(100) == ord('q'):
        cv2.destroyAllWindows()
        exit(-1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 5: Triangulation
