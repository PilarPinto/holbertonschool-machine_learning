#!/usr/bin/env python3
'''File where is comute the cost NN'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''Cost of NN with L2 regularization'''
    sum = 0
    for layer in range(1, L+1):
        k = 'W' + str(layer)
        sum += np.linalg.norm(weights[k])
    cost_l2_reg = cost + ((lambtha / (2 * m)) * sum)

    return cost_l2_reg
