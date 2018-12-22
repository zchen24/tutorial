#!/usr/bin/env python

import numpy as np
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] = val
        return self.array


if __name__ == '__main__':
    b = Bag(5)
    print('Bag.value = {}'.format(b.value))
    print('Bag.array = {}'.format(b.array))
    b.increment(3)
    print('Bag.array = {}'.format(b.array))
    b.increment(6)
    print('Bag.array = {}'.format(b.array))
