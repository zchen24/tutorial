#!/usr/bin/env python

import numpy as np
from numba import njit
import cProfile

@njit(parallel=True)
def offset_array(ina, offset, gain):
    return (ina + offset) * gain


if __name__ == '__main__':
    ina = np.zeros((1920, 1080)) + 1
    offset = np.zeros((1920, 1080)) + 2
    out = offset_array(ina, offset, 1.1)

    profile = cProfile.Profile()
    profile.enable()

    for i in range(5000):
        out = offset_array(ina, offset, 1.1)

    profile.disable()
    profile.print_stats(sort='time')