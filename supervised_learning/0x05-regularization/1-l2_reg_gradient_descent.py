#!/usr/bin/env python3
'''File for update w and b of a NN'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''Update w and b of NN with gradient descent'''

    Z = cache['A' + str(L)]
    dz = Z - Y
    m = Y.shape[1]

    for layer in reversed(range(1, L + 1)):
        weight_k = 'W' + str(layer)
        bias_k = 'b' + str(layer)
        Z_k = 'A' + str(layer)
        Z1_k = 'A' + str(layer - 1)

        Z = cache[Z_k]
        dz2 = 1 - (Z**2)

        if layer == L:
            dZl = dz
        else:
            dZl = dz * dz2

        weight = weights[weight_k]
        dz1 = cache[Z1_k]

        d_weight = (1/m) * np.matmul(dZl, dz1.T) + ((lambtha/m)*weight)
        d_bias = (1/m) * np.sum(dZl, axis=1, keepdims=True)
        dz = np.matmul(weight.T, dZl)

        weights[weight_k] = weights[weight_k] - alpha + d_weight
        weights[bias_k] = weights[bias_k] - alpha + d_bias
