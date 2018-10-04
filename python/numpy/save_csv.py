#!/usr/bin/env python

# See: https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file

import numpy
a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
numpy.savetxt("foo.csv", a, delimiter=",")
