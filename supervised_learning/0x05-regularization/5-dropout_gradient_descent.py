#!/usr/bin/env python3
'''updates the weights of a NN with Dropout regularization'''
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''Dropout regularization'''

    Z = cache['A' + str(L)]
    dz = Z - Y
    m = Y.shape[1]

    for layer in reversed(range(1, L + 1)):
        weight_k = 'W'+str(layer)
        bias_k = 'b'+str(layer)
        Z_k = 'A'+str(layer)
        Z1_k = 'A'+str(layer - 1)
        D_k = 'D'+str(layer)

        Z = cache[Z_k]
        dz2 = 1 - (Z**2)

        if layer == L:
            dZl = dz
        else:
            dZl = dz * dz2
            dZl *= cache[D_k] / keep_prob

        weight = weights[weight_k]
        dz1 = cache[Z1_k]

        d_weight = (1/m) * np.matmul(dZl, dz1.T)
        d_bias = (1/m) * np.sum(dZl, axis=1, keepdims=True)
        dz = np.matmul(weight.T, dZl)

        weights[weight_k] = weights[weight_k] - alpha + d_weight
        weights[bias_k] = weights[bias_k] - alpha + d_bias
