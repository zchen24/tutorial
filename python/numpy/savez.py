#!/usr/bin/env python3

"""
Demo how to save and load numpy data 
savez and load

Author: Zihan Chen
Date: 2019-07-04
"""

import numpy as np

# save m0, m1 to file
m0 = np.zeros(5)
m1 = np.ones(5)
np.savez('tmp.npz', m0=m0, m1=m1)

# remove m0, m1
del m0
del m1

# make m0, m1 no longer exist
try:
    print('m0 = {}  m1 = {}'.format(m0, m1))
except NameError as e:
    print(e)
    

# now load m0, m1 from file
tmp = np.load('tmp.npz')
m0 = tmp['m0']
m1 = tmp['m1']
print('m0 = {}\nm1 = {}'.format(m0, m1))
