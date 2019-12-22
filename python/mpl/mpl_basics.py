#!/usr/bin/env python
"""
Basic Matplotlib examples shows:
- create figure, subplots
- create axes
- x/y label
- x/y limit
- set ticks
- legend
- grid
"""


import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()

rows = 1
cols = 2
ax1 = fig.add_subplot(rows, cols, 1)
ax2 = fig.add_subplot(rows, cols, 2)


x = np.linspace(0, 10, 1000)
y_sin = np.sin(x)
y_sin_2 = np.sin(x) * 0.5
y_cos = np.cos(x)

ax1.plot(x, y_sin)
ax1.plot(x, y_sin_2)
ax2.plot(x, y_cos)

# set title
fig.suptitle('Sup title')
ax1.set_title('Ax1 title sine')
ax2.set_title('Ax2 title cosine')

# set labels
# ax1.set_xlabel('x data')
ax1.set_ylabel('y data sin(x)')
ax2.set_xlabel('x data')
ax2.set_ylabel('y data cos(x)')

# set ticks
ax2.set_xticks(range(0, 11))

# set legend
# see legend documentation for details
# loc: strings
#     - 'best'
#     - 'upper right'
#     - 'center'
ax1.legend(['sin', 'sin * 0.5'], loc='best')

# set grid
ax1.grid(True)
ax2.grid(True)

# save figure
#   bbox_inches='tight' removes whitespaces around figure
plt.savefig('mpl_basics.jpg', bbox_inches='tight')
plt.show()

# close figure
plt.close('all')

