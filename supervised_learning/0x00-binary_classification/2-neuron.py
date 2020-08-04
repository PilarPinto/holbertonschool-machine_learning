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
