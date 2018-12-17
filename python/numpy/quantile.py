#!/usr/bin/env python

import numpy as np

data = range(1000)
q = [0.01, 0.99]
res = np.quantile(data, q)
print('res = {}'.format(res))
