#!/usr/bin/env python

"""
Example: how to use polyfit and polyval to fit linear models
"""

import numpy as np
import matplotlib.pyplot as plt


num_pts = 30
x = np.linspace(-2, 2, num_pts)
y = np.power(x, 2) + 1

y_with_rand = y + np.random.rand(num_pts) - 0.5

# 2 means using order 2 polynomial models
p2 = np.polyfit(x, y_with_rand, 2)
y_fitted = np.polyval(p2, x)

plt.plot(x, y, '-')
plt.plot(x, y_with_rand, '+')
plt.plot(x, y_fitted, 'o-')
plt.legend(['Ground Truth', 'With Noise', 'Fitted'], loc='best')

plt.grid(True)
plt.show()
