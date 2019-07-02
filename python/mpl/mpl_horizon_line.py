#!/usr/bin/env python3

"""
Demo how to add an horizontal line
Date: 2019-07-02
"""


import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 1000)
y_sin = np.sin(x)
plt.plot(x, y_sin)

# Plot a horizontal line
#   - at y = 0.5
#   - color is red
#   - width is 4
plt.axhline(y=0.5, color='r', linewidth=4)
plt.grid(True)
plt.show()
plt.title('Show how to add a horizontal line')