#!/usr/bin/env python3
'''Plot a cubic function'''
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)
plt.xlim(0, 10)
plt.plot(x, y, color='red')
plt.show()
