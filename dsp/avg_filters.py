#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt


fs = np.arange(0, 0.5, step=0.005)
print("freqs = {}".format(fs))

M = 3   # number of samples
H = np.sin(np.pi * fs * M) / (M * np.sin(np.pi * fs))

plt.plot(fs, H, '.')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()