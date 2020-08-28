#!/usr/bin/env python3
'''File for update w and b of a NN'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''Update w and b of NN with gradient descent'''

    Z = cache['A' + str(L)]
    dZ = Z - Y
    m = Y.shape[1]

    for layer in reversed(range(1, L + 1)):
        Z = cache['A' + str(layer)]
        dz2 = 1 - (Z**2)
        if layer == L:
            dZ = dZ
        else:
            dZ = dZ * dz2

        Wgt = weights['W' + str(layer)]
        Z1 = cache['A' + str(layer - 1)]
        dWgt = (1 / m) * np.matmul(dZ, Z1.T) + ((lambtha / m) * Wgt)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.matmul(Wgt.T, dZ)
        weights['W' + str(layer)] = weights['W' + str(layer)] - alpha * dWgt
        weights['b' + str(layer)] = weights['b' + str(layer)] - alpha * db
