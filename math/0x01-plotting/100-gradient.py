#!/usr/bin/env python3
'''Plot gradient #D in a 2D way'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# your code here
fig, ax = plt.subplots()

plt.title('Mountain Elevation')
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
cs = plt.scatter(x, y, c=z, cmap='viridis')
cbar = fig.colorbar(cs, label="elevation (m)")

plt.show()
