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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            if type(layers[l]) is not int or layers[l] <= 0:
                raise TypeError('layers must be a list of positive integers')

            he_al = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)
            self.__weights['W' + str(l+1)] = he_al
            self.__weights['b' + str(l+1)] = np.zeros((layers[l], 1))
            nx = layers[l]

    @property
    def L(self):
        '''Getter of L value'''
        return self.__L

    @property
    def cache(self):
        '''Getter of cache value'''
        return self.__cache

    @property
    def weights(self):
        '''Getter of weights value'''
        return self.__weights
