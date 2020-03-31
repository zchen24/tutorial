#!/usr/bin/env python3

# See: https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file

import numpy as np

filename = "foo.csv"
a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ], dtype=np.float)
print("a = \n {}\n".format(a))
np.savetxt(filename, a, delimiter=',')

a_csv = np.loadtxt(filename, delimiter=',')
print("a_csv = \n {}".format(a_csv))
