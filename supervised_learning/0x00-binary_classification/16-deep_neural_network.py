#!/usr/bin/env python3
'''Deep Neural Network file'''

import numpy as np


class DeepNeuralNetwork:
    '''neural network performing binary classification:'''
    def __init__(self, nx, layers):
        '''class constructor with features of the neuron'''

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) is 0:
            raise TypeError('layers must be a list of positive integers')

        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        self.nx = nx

        for l in range(self.L):
            if type(layers[l]) is not int or layers[l] <= 0:
                raise ValueError('layers must be a list of positive integers')

            he_al = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)
            self.weights['W' + str(l+1)] = he_al
            self.weights['b' + str(l+1)] = np.zeros((layers[l], 1))
            nx = layers[l]
