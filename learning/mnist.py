#!/usr/bin/env python

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, interpolation="nearest")
plt.axis("off")
plt.show()


# create test set 


import ipdb; ipdb.set_trace()