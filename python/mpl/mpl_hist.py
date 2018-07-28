#!/usr/bin/env python

import matplotlib.pyplot as plt
from numpy.random import randn


# two subplots
fig, (ax1, ax2) = plt.subplots(2, 1)

# default bin numbers = 10
ax1.hist(randn(1000))
ax1.set_title('Histogram with default 10 bins')

# set bin number
bin = 30
ax2.hist(randn(1000), bins=bin)
ax2.set_title('Histogram with 30 bins')

plt.tight_layout()
plt.show()