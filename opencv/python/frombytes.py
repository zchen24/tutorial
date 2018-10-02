#!/usr/bin/env python

"""
This example shows how to convert bytearray into CV image.

1) shows that CV treats images as numpy arrays
2) construct a raw image using np
3) convert raw into bytes, then convert back to numpy array
4) display raw and img_from_rawbytes

Author: Zihan Chen
Date: 2018-10-01
"""


import numpy as np
import cv2


if __name__ == '__main__':
    # read an image as GRAYSCALE & introspect its type
    img = cv2.imread('../imgs/board.png', cv2.IMREAD_GRAYSCALE)
    print('img type: {} shape: {}  dtype: {}'.format(
        type(img).__name__, img.shape, img.dtype))

    # manually construct a bytearray
    raw = np.zeros((400, 400), dtype=np.uint8)
    for r in range(10):
        for c in range(10):
            if (r + c) % 2 == 1:
                raw[r*40:r*40+40, c*40:c*40+40] = 255

    rawbytes = raw.tobytes()
    img_from_rawbytes = np.frombuffer(rawbytes, dtype=np.uint8).reshape((400, 400))

    # import ipdb; ipdb.set_trace()
    cv2.imshow('raw', raw)
    cv2.imshow('from rawbytes', img_from_rawbytes)
    cv2.waitKey(0)

