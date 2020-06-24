#!/usr/bin/env python3

"""
Load an image and show using PIL

color_picker.py file.png 'hsv'
"""

import cv2
from PIL import Image
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color picker')
    parser.add_argument('file', help='Image file name')
    parser.add_argument('format', type=str, default='hsv', help='convert image to')
    args = parser.parse_args()

    # filename
    filename = args.file

    # format
    format = args.format
    format_map = {'hsv': cv2.COLOR_BGR2HSV}
    if not format in format_map.keys():
        raise ValueError('Unsupported format')

    img = cv2.imread(filename)
    if img.size == 0:
        print('Error: failed to read image from {}'.format(filename))

    img_cvt = cv2.cvtColor(img, format_map[format])
    img_cvt_pil = Image.fromarray(img_cvt)
    plt.imshow(img_cvt_pil)
    plt.show()
