#!/usr/bin/env python3

"""
Demo how to do bar plot
Date: 2019-07-02
"""


import numpy as np
import matplotlib.pyplot as plt

y = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
x_pos = np.arange(len(y))
x_ticks = ['{:02}'.format(i) for i,_ in enumerate(y)]

plt.bar(x_pos, y)
plt.xticks(x_pos, x_ticks)
plt.grid(True, axis='y')
plt.show()
plt.title('Show how to do bar plot')
