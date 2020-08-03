#!/usr/bin/env python3
'''Neuron file'''

import numpy as np


class Neuron:
    '''Class Neuron definition'''
    def __init__(self, nx):
        '''class constructor with features of the neuron'''

        if nx % 1 != 0:
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
