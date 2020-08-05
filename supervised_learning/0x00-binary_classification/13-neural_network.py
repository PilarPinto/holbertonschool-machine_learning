#!/usr/bin/env python3
'''Neural Network file'''

import numpy as np


class NeuralNetwork:
    '''Class Neuronal network definition'''
    def __init__(self, nx, nodes):
        '''class constructor with features of the neuronal network'''

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''Getter of weight value 1'''
        return self.__W1

    @property
    def b1(self):
        '''Getter of bias value 1'''
        return self.__b1

    @property
    def A1(self):
        '''Getter of activation value 1'''
        return self.__A1

    @property
    def W2(self):
        '''Getter of weight value 2'''
        return self.__W2

    @property
    def b2(self):
        '''Getter of bias value 2'''
        return self.__b2

    @property
    def A2(self):
        '''Getter of activation value 2'''
        return self.__A2

    def forward_prop(self, X):
        '''Foward propagation for a neural network'''
        mul_Z1 = (np.matmul(self.__W1, X)) + self.__b1
        self.__A1 = 1 / (1 + np.exp(- mul_Z1))
        mul_Z2 = (np.matmul(self.__W2, self.__A1)) + self.__b2
        self.__A2 = 1 / (1 + np.exp(- mul_Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''Logisting regression cost'''
        m = Y.shape[1]
        q = 1 - Y
        cost = (-1 / m) * np.sum(Y * np.log(A) + q * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions'''
        A = self.forward_prop(X)
        ev = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return (ev, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''Neural network Gradient Descent'''
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        der_b2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1-A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        der_b1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * der_b1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * der_b2
