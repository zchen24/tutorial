#!/usr/bin/env python

"""
Window Sinc Filters

1) frequency domain filter, low-pass filter
2) convolution based method


Author: Zihan Chen
Date: Feb 20, 2019
"""

import numpy as np
import matplotlib.pyplot as plt


M = 50
t = np.arange(50)

w_hamming = 0.54 - 0.46 * np.cos(2*np.pi*t/M)
w_blackman = 0.42 - 0.5 * np.cos(2 * np.pi * t / M) + \
             0.08 * np.cos(4 * np.pi * t / M)


plt.plot(t, w_blackman, '.')
plt.plot(t, w_hamming, '.')
plt.xlabel("Sample number")
plt.ylabel("Amplitude")
plt.legend(['Blackman', 'Hamming'])
plt.title('Blackman and Hmaming window')
plt.grid(True)
plt.show()


# =====================================
# Figure 16-4: Example filter kernel
# =====================================

def window_sinc_kernel(fc, M):
    """
    Args:
        fc: cut-off frequency
        M: number of samples

    Returns:
        Windoed-sinc filter kernel
    """
    k = np.ones(50)
    return k





