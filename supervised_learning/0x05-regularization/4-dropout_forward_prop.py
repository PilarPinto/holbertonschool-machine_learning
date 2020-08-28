#!/usr/bin/env python3
'''File forward propagation using Dropout'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''forward propagation using Dropout'''
    cache = {}
    cache['A0'] = X

    for layer in range(1, L+1):
        weight = weights['W'+str(layer)]
        bias = weights['b'+str(layer)]
        activation = cache['A'+str(layer-1)]

        z = (np.matmul(weight, activation)) + bias
        drop = np.random.binomial(1, keep_prob, size=z.shape)

        if layer == L:
            t = np.exp(z)
            cache['A'+str(layer)] = t/np.sum(t, axis=0, keepdims=True)
        else:
            cache['A'+str(layer)] = np.tanh(z)
            cache['D'+str(layer)] = drop
            cache['A'+str(layer)] *= drop
            cache['A'+str(layer)] /= keep_prob
    return cache
