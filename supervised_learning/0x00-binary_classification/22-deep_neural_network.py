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

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''Neural network Gradient Descent
        '''
        m = Y.shape[1]
        for layer in reversed(range(1, self.L + 1)):

            Anc = self.__cache['A' + str(layer)]
            Ancp = self.__cache['A' + str(layer-1)]
            Wn = self.__weights['W' + str(layer)]

            if layer == self.L:
                dZc = self.__cache['A' + str(self.L)] - Y
            if layer < self.L:
                dZc = dZcur * (Anc * (1-Anc))

            dw_c = (1 / m) * np.matmul(dZc, Ancp.T)
            der_bc = (1 / m) * np.sum(dZc, axis=1, keepdims=True)
            dZcur = np.matmul(Wn.T, dZc)

            self.__weights['W' + str(layer)] = self.__weights[
                'W' + str(layer)] - alpha * dw_c
            self.__weights['b' + str(layer)] = self.__weights[
                'b' + str(layer)] - alpha * der_bc

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''Train deep neural network'''

        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for ind in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

        return (self.evaluate(X, Y))
