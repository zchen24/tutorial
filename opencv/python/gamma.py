#!/usr/bin/env python

"""
Example shows gamma curves

Reference:
[1] Image Processing Algorithms Part 6: Gamma Correction
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_gamma():
    level = np.arange(256)

    gammas = [0.5, 1.0, 1.5, 2.0]
    legend = []

    for g in gammas:
        level_gamma = 255 * (level/255)**(g)
        plt.plot(level, level_gamma)
        legend.append('gamma = {}'.format(g))

    plt.grid(True)
    plt.legend(legend)
    plt.title('Various Gamma Curves')
    plt.show()


def gamma_correct(img, gamma=1.0):
    img_corrected = 255 * (img / 255) ** (1.0/gamma)
    return img_corrected.astype(np.uint8)


if __name__ == '__main__':
    plot_gamma()

    img = cv2.imread('../imgs/lena.jpg')
    img_0_25 = gamma_correct(img, 0.25)
    img_2_00 = gamma_correct(img, 2.00)

    cv2.imshow('img', img)
    cv2.imshow('gamma025', img_0_25)
    cv2.imshow('gamma200', img_2_00)
    cv2.waitKey(0)
    cv2.destroyAllWindows()