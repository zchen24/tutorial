#!/usr/bin/env python

import cv2
from pylab import *


def histeq(img, nbr_bins=256):
    """Histogram equalization of a grayscale image"""
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(img.flatten(), bins[-1], cdf)
    return im2.reshape(img.shape), cdf


img = cv2.imread('./data/messi5.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.gray()
plt.imshow(img)

plt.figure()
plt.hist(img.flatten(), bins=128)
plt.show()

import ipdb; ipdb.set_trace()
