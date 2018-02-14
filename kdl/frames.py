#!/usr/bin/env python

from PyKDL import *

# ---------------------
# vector
# ---------------------
v1 = Vector(1, 2, 3)

# get vector norm
v1_norm = v1.Norm()

# normalize vector v
v1_normalized = v1.Normalized()

# dot product
v2 = Vector(1, 1, 1)
a = dot(v1, v2)

# cross product
v3 = v1 * v2


