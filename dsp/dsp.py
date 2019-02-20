#!/usr/bin/env python


import numpy as np


def mean(a):
    """Compute mean
    Args:
        a: array_like
    """
    sum = 0
    for i in a:
        sum += i
    return sum/len(a)


if __name__ == '__main__':
    print("Hello dsp book")
