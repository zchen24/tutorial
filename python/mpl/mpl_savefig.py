#!/usr/bin/env python

"""
How to save matplotlib figure as png/svg

Copyright 2018 Zihan Chen
"""

import matplotlib.pyplot as plt
from numpy.random import randn

plt.hist(randn(1000))
plt.title('Histogram with default 10 bins')
# save figure
#   bbox_inches='tight' removes whitespaces around figure
plt.savefig('mpl_savefig.svg', bbox_inches='tight')
plt.show()
