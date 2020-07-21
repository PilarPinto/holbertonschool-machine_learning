#!/usr/bin/env python3
'''Plot 5 graphs in one plot'''
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
fig, axs = plt.subplots(3, 2)
fig.tight_layout(h_pad=2.3, w_pad=3.5)
fig.suptitle('All in One')
fig.subplots_adjust(top=0.92)


''' Figure Line figure '''
x0 = np.arange(0, 11)
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
# ax1 = plt.subplot(321)
ax1.plot(x0, y0, color='red')
ax1.set_xlim([0, 10])
ax1.set_yticks(np.arange(0, 1001, 500))

'''Figure Scatter '''
ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
ax2.plot(x1, y1, 'ro', color='magenta')
ax2.set_xlabel('Height (in)', fontsize='x-small')
ax2.set_ylabel('Weight (lbs)', fontsize='x-small')
ax2.set_title('Men\'s Height vs Weight', fontsize='x-small')


''' Exponential decay'''

ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
ax3.plot(x2, y2)
ax3.set_yscale("log")
ax3.set_xlim([0, 28651])
ax3.set_xlabel('Time (years)', fontsize='x-small')
ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
ax3.set_title('Exponential Decay of C-14', fontsize='x-small')


'''Exponential decay of reactive'''

ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
ax4.plot(x3, y31, 'r--', color='red', label="C-14")
ax4.plot(x3, y32, color='green', label="Ra-226")
ax4.set_xlim([0, 20000])
ax4.set_ylim([0, 1])
ax4.legend(prop={'size': 7})
ax4.set_xlabel('Time (years)', fontsize='x-small')
ax4.set_ylabel('Fraction Remaining', fontsize='x-small')
ax4.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')

'''Histogram'''
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
ax5.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
ax5.set_xlabel('Grades', fontsize='x-small')
ax5.set_ylabel('Number of Students', fontsize='x-small')
ax5.set_title('Project A', fontsize='x-small')
ax5.set_xlim([0, 100])
ax5.set_ylim([0, 30])
ax5.set_xticks(np.arange(0, 101, 10))
ax5.set_yticks(np.arange(0, 31, 10))

plt.show()
