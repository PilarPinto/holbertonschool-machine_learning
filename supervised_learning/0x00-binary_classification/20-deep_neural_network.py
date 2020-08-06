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

    def forward_prop(self, X):
        '''Foward propagation for a deep neural network'''
        self.__cache['A0'] = X
        mul_Z1 = (np.matmul(self.__weights['W1'],
                            self.__cache['A0'])) + self.__weights['b1']
        self.__cache['A1'] = 1 / (1 + np.exp(- mul_Z1))

        for l in range(self.L-1):
            mul_Z = (np.matmul(self.__weights['W' + str(l+2)], self.__cache[
                'A' + str(l+1)])) + self.__weights['b' + str(l+2)]
            self.__cache['A' + str(l+2)] = 1 / (1 + np.exp(- mul_Z))

        return self.__cache['A' + str(self.L)], self.__cache

    def cost(self, Y, A):
        '''Logisting regression cost'''
        m = Y.shape[1]
        q = 1 - Y
        cost = (-1 / m) * np.sum(Y * np.log(A) + q * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions'''
        A, self.__cache = self.forward_prop(X)
        ev = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return (ev, cost)
