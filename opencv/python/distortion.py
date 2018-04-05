#!/usr/bin/env python

"""
Show how to distort an image with OpenCV
"""


import cv2
import numpy as np


def manual_image_undistortion(img, camera_matrix, dist_coefficient):
    """ Manual Image Distortion

    :param img: grayscale image
    :param camera_matrix: 3x3 camera matrix
    :param dist_coefficient: 4x1 or 5x1 distortion coefficient
    :return: undistorted image
    """
    width = img.shape[1]
    height = img.shape[0]

    k1 = dist_coefficient[0]
    k2 = dist_coefficient[1]

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    # create an image with same size
    uimg = np.zeros((height, width, 1), np.uint8)

    for w in range(width):
        for h in range(height):
            rx = (w - cx)/fx
            ry = (h - cy)/fy
            r = np.sqrt(pow(rx, 2) + pow(ry, 2))
            scale = 1 + k1 * pow(r, 2) + k2 * pow(r, 4)
            xu = int((w-cx)*scale + cx)
            yu = int((h-cy)*scale + cy)
            if (0 <= xu < width) & (0 <= yu < height):
                # uimg[yu, xu] = img[h, w]
                uimg[h, w] = img[yu, xu]
    return uimg


# np array
img = cv2.imread('./data/board.png', cv2.IMREAD_GRAYSCALE)
width = img.shape[1]
height = img.shape[0]

# distortion coefficient
# (k1, k2, p1, p2, k3)
dist_coef = np.zeros((5,1))
dist_coef[0, 0] = 0.05
dist_coef[1, 0] = 0.02
dist_coef[2, 0] = 0.0
dist_coef[3, 0] = 0.0
dist_coef[4, 0] = 0.0


# ---- camera matrix -----
# (cx, cy): image center
# fx: x focal length
# fy: y focal length
#
# fx     0    0    cx
#  0    fy    0    cy
#  0     0    1     0
#
cam_matrix = np.eye(3)
cam_matrix[0, 2] = width/2
cam_matrix[1, 2] = height/2
cam_matrix[0, 0] = width/2     # focal length x
cam_matrix[1, 1] = height/2    # focal length y


# cv call to un-distort image
# here it is used to distort a normal image
img_undistorted = cv2.undistort(img, cam_matrix, dist_coef)


# After barrel distortion, there is a boundary
#
# find out boundary
# xu = x * (1 + k1*r^2 + k2*r^4)
x_max_scale = 1 + dist_coef[0, 0] + dist_coef[1, 0]
y_max_scale = 1 + dist_coef[0, 0] + dist_coef[1, 0]
width_new = width / x_max_scale
height_new = height / y_max_scale

print('width_new = ', width_new)
print('height_new = ', height_new)

# crop out the black boundary
x_min = int((width - width_new)/2)
x_max = int((width + width_new)/2)
y_min = int((height - height_new)/2)
y_max = int((height + height_new)/2)
img_undistorted_roi = img_undistorted[y_min:y_max, x_min:x_max]

# resize the cropped image to the original size
img_undistorted_resized = cv2.resize(img_undistorted_roi, (800, 600))



# ----- Manual ------
# Call the manual implementation
img_undistorted_manual = manual_image_undistortion(img, cam_matrix, dist_coef)


# -------------------
# Display images
# -------------------
cv2.imshow('img', img)
cv2.imshow('undistorted', img_undistorted)
cv2.imshow('manual', img_undistorted_manual)

cv2.waitKey(0)

