#!/usr/bin/env python3
'''Neuron file'''

import numpy as np


class Neuron:
    '''Class Neuron definition'''
    def __init__(self, nx):
        '''class constructor with features of the neuron'''

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Getter of weight value'''
        return self.__W

    @property
    def b(self):
        '''Getter of bias value'''
        return self.__b

    @property
    def A(self):
        '''Getter of activation value'''
        return self.__A

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron'''
        mul_Z = (np.matmul(self.__W, X)) + self.__b
        self.__A = 1 / (1 + np.exp(- mul_Z))
        return self.__A

    def cost(self, Y, A):
        '''Logisting regression cost'''
        label = Y.shape[1]
        q = 1 - Y
        cost = (-1 / label) * np.sum(Y * np.log(A) + q * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions'''
        A = self.forward_prop(X)
        ev = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return (ev, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Neuron Gradient Descent'''
        m = Y.shape[1]
        loss = A - Y
        gradient = (1/m) * np.matmul(X, loss.T)
        der_b = (1/m) * np.sum(loss)
        self.__W = self.__W - (alpha * gradient).T
        self.__b = self.__b - alpha * der_b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''Train Neuron'''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for ind in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

        return (self.evaluate(X, Y))
