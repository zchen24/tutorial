#!/usr/bin/env python

"""
Numba cuda example
1) use numba -s to check cuda compability
2) Install if necessary: conda install cudatoolkit

http://numba.pydata.org/numba-doc/latest/cuda/kernels.html
"""


from numba import cuda
import numpy as np
import cProfile

@cuda.jit
def increment_by_one(an_array):
    tx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    idx = tx + bidx * bdim
    if idx < an_array.size:
        an_array[idx] += 1


dev = cuda.get_current_device()
print('Cuda device name = {}'.format(dev.name))

a = np.ones(1000000)
thread_per_block = 512
block_per_grid = int(np.ceil(a.size / thread_per_block))
increment_by_one[block_per_grid, thread_per_block](a)

profile = cProfile.Profile()
profile.enable()

d_a = cuda.to_device(a)
for i in range(10000):
    increment_by_one[block_per_grid, thread_per_block](d_a)
    # a += 1
d_a.copy_to_host(a)

profile.disable()
profile.print_stats(sort='time')

print(a)