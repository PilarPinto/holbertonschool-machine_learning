#!/usr/bin/env python3
'''Plot stacking bars for four fruits'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
list_by_fruit = np.array(fruit)
apples = list_by_fruit[0, :]
bananas = list_by_fruit[1, :]
oranges = list_by_fruit[2, :]
peaches = list_by_fruit[3, :]
names = ['Farrah', 'Fred', 'Felicia']
x = np.arange(3)
barWidth = 0.5

plt.bar(x, apples, color='red', width=barWidth, label="apples")
plt.bar(x, bananas, bottom=apples, color='yellow', width=barWidth,
        label="bananas")
plt.bar(x, oranges, bottom=apples+bananas, color='#ff8000',
        width=barWidth, label="oranges")
plt.bar(x, peaches, bottom=apples+bananas+oranges, color='#ffe5b4',
        width=barWidth, label="peaches")

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 81, 10))
plt.xticks(x, names)
plt.legend()
plt.show()
