#!/usr/bin/env python

from numba import jit, njit
import numpy as np
import cProfile


@jit(nopython=True, cache=True)
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


if __name__ == '__main__':
    arr = np.array([1,2,3,4,5,6,7,8,9,10]).reshape([2,5])
    sum = sum2d(arr)

    profile = cProfile.Profile()
    profile.enable()

    for i in range(100000):
        sum = sum2d(arr)

    profile.disable()
    profile.print_stats(sort='time')

    print('sum = {}'.format(sum))