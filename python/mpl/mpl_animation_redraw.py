#!/usr/bin/env python

"""
Demo simple animation with redraw (clears previous plot) 

Date: 2019-07-04
License: BSD 
"""

import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == '__main__':

    x = np.linspace(0, 2*np.pi, 50)

    for t in np.linspace(0, 10, 100):
        y = np.sin(x + t)

        plt.cla()   # clear axis        
        plt.plot(x, y)
        plt.title('y = sin(x + t)')
        plt.grid(True)
        plt.ylim([-1.05, 1.05])
        plt.draw()
        plt.pause(0.05)

    plt.close()
