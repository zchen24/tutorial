#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


state = 0
val = 0.0
alpha = 0.9
input = np.array([1] * 100)
input[0:40] = 0
input[40:60] = np.random.randint(0, 2, 20)
input[60:-1] = 1
all_states = [0] * 100
thresh_high = 0.8
thresh_low = 0.4

for i in range(100):
    val = val * alpha + input[i] * (1 - alpha)
    if state == 0 and val > thresh_high:
        state = 1
    elif state == 1 and val < thresh_low:
        state = 0
    all_states[i] = state

plt.plot(input, 'b')
plt.plot(all_states, 'r')
plt.show()