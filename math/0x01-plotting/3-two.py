#!/usr/bin/env python3
'''Plot two functions'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)
plt.title('Exponential Decay of Radioactive Elements')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.plot(x, y1, 'r--', color='red', label="C-14")
plt.plot(x, y2, color='green', label="Ra-226")
plt.legend()
ax = plt.gca()
ax.set_xlim([0, 20000])
ax.set_ylim([0, 1])
plt.show()
