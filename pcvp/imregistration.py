#!/usr/bin/env python3

"""
Image Registration: PCVP Chapter 3.2
"""

import os
from pylab import *
from scipy import ndimage
from xml.dom import minidom
import cv2

def read_points_from_xml(xmlFileName):
    """Reads control points for face alignment."""
    xmldoc = minidom.parse(xmlFileName)
    facelist = xmldoc.getElementsByTagName('face')
    faces = {}
    for face in facelist:
        fileName = face.attributes['file'].value
        xf = int(face.attributes['xf'].value)
        yf = int(face.attributes['yf'].value)
        xs = int(face.attributes['xs'].value)
        ys = int(face.attributes['ys'].value)
        xm = int(face.attributes['xm'].value)
        ym = int(face.attributes['ym'].value)
        faces[fileName] = np.array([xf, yf, xs, ys, xm, ym])
    return faces


def compute_rigid_transform(refpoints, points):
    """Computes rotation, scale and translation for
    aligning points to reference points (refpoints)."""
    A = np.array([[points[0], -points[1], 1, 0],
                  [points[1], points[0], 0, 1],
                  [points[2], -points[3], 1, 0],
                  [points[3], points[2], 0, 1],
                  [points[4], -points[5], 1, 0],
                  [points[5], points[4], 0, 1]])
    y = np.array([refpoints[0],
                  refpoints[1],
                  refpoints[2],
                  refpoints[3],
                  refpoints[4],
                  refpoints[5]])
    a, b, tx, ty = np.linalg.lstsq(A, y)[0]
    R = np.array([[a, -b],
                  [b, a]])
    return R, tx, ty


def rigid_alignment(faces, path, plotflag=False):
    """Align images rigidly and save as new images"""

    # take the points in the first image as referece points
    refpoints = list(faces.values())[0]
    print('total images = {}'.format(len(faces)))

    # warp each image using affine transform
    for i, face in enumerate(faces):
        points = faces[face]
        R, tx, ty = compute_rigid_transform(refpoints, points)
        T = np.array([[R[1][1], R[1][0]],
                      [R[0][1], R[0][0]]])

        im = cv2.imread(os.path.join(path, face))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im2 = np.zeros(im.shape, 'uint8')

        # warp each color channel
        for c in range(len(im.shape)):
            im2[:,:,c] = ndimage.affine_transform(im[:,:,c], np.linalg.inv(T), offset=[-ty, -tx])

        if plotflag:
            plt.imshow(im2)
            plt.draw()
            plt.axis('off')
            plt.title('img {}: {}'.format(i, face))
            plt.pause(0.1)

        im2= cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)

        # crop away border and save aligned images:
        h, w = im2.shape[:2]
        border = int((w+h)/20)
        # crop away border
        cv2.imwrite(os.path.join(path, 'aligned/'+face), im2[border:h-border, border:w-border,:])


if __name__ == '__main__':
    faces = read_points_from_xml('./pcv_data/jkfaces.xml')
    path = './pcv_data/jkfaces'
    # rigid_alignment(faces, './pcv_data/jkfaces', False)

    # compute mean
    num_faces = len(faces)
    all_faces = np.zeros((400, 300, 3, num_faces))
    all_faces_registered = np.zeros((330, 230, 3, num_faces))
    count = 0
    for i, face in enumerate(faces):
        print('reading image {}: {}'.format(i, face))
        try:
            all_faces[:, :, :, i] = cv2.imread(os.path.join(path, face))
            all_faces_registered[:, :, :, i] = cv2.imread(os.path.join(path, 'aligned', face))
            count += 1
        except:
            continue

    print('{} images added'.format(count))
    mean_face = all_faces[:, :, :, :count].mean(axis=3).astype(np.uint8)
    mean_face_registered = all_faces_registered[:, :, :, :count].mean(axis=3).astype(np.uint8)
    cv2.imshow('mean_face', mean_face)
    cv2.imshow('mean_face_registered', mean_face_registered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
